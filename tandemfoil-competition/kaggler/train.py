"""Train Transolver on TandemFoilSet (frieren, apr27 iter2: warmstart).

Strategy:
  - Warmstart from prior best checkpoint (matches current 42.11 leader arch).
  - bf16 AMP for speed; full 30-min budget.
  - Low LR fine-tuning with cosine decay (no warmup).
  - Combined L1 + L2 loss (L1 aligns with MAE eval metric).
  - EMA weights for validation/checkpoint selection.

Run:
  python train.py --agent frieren --wandb_name "frieren/<desc>"
"""

import copy
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from viz import visualize


# ---------------------------------------------------------------------------
# Transolver model
# ---------------------------------------------------------------------------

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        if self.unified_pos:
            self.preprocess = MLP(fun_dim + ref**3, n_hidden * 2, n_hidden,
                                   n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden,
                                   n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx)
        return {"preds": fx}


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def state_dict(self, model: nn.Module) -> dict:
        out = {k: v.clone() for k, v in model.state_dict().items()}
        for k, v in self.shadow.items():
            out[k] = v.clone()
        return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))
# Iter13: warm-restart cycle from iter11 best (ii1fhq90, val=44.32) with a
# bigger LR + single_boost=4 to push harder on the weakest split.
WARMSTART_PATH = "/mnt/new-pvc/kagent/apr27/frieren/checkpoints/model-ii1fhq90/checkpoint.pt"


@dataclass
class Config:
    lr: float = 2e-5
    min_lr: float = 1e-7
    weight_decay: float = 1e-4
    batch_size: int = 2
    surf_weight: float = 10.0
    epochs: int = 14
    warmup_epochs: int = 0  # no warmup; chain step
    grad_clip: float = 1.0
    l1_weight: float = 1.0
    l2_weight: float = 1.0  # restore L2 — earlier soup analysis shows L1-only is correlated with iter7 anyway
    p_weight: float = 5.0
    ema_decay: float = 0.99
    # Strongly boost the racecar_single domain — biggest test gap is here.
    single_boost: float = 4.0
    warmstart: str = WARMSTART_PATH
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


# Architecture matches WARMSTART_PATH (s8nqhr0q: hid=192 L=6 S=64)
MODEL_CONFIG = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=192,
    n_layers=6,
    n_head=6,
    slice_num=64,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    if cfg.debug:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, **loader_kwargs)
    else:
        # Optionally upweight the racecar_single domain (our weakest val split).
        if cfg.single_boost != 1.0:
            import json as _json
            with open(Path(cfg.splits_dir) / "meta.json") as f:
                _meta = _json.load(f)
            single_idxs = set(_meta["domain_groups"].get("racecar_single", []))
            boosted = sample_weights.clone()
            for i in single_idxs:
                boosted[i] = boosted[i] * cfg.single_boost
            print(f"Sampler: single_boost={cfg.single_boost} applied to {len(single_idxs)} samples")
            sample_weights = boosted
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  sampler=sampler, **loader_kwargs)

    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = Transolver(**MODEL_CONFIG).to(device)
    if cfg.warmstart and Path(cfg.warmstart).exists():
        sd = torch.load(cfg.warmstart, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Warmstart from {cfg.warmstart}  missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"WARNING: warmstart path {cfg.warmstart} not found; training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Cosine decay from cfg.lr → cfg.min_lr (no warmup; we're fine-tuning)
    def lr_lambda(epoch):
        progress = epoch / max(1, MAX_EPOCHS - 1)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cos + (cfg.min_lr / cfg.lr) * (1 - cos)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = EMA(model, decay=cfg.ema_decay)

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v2"),
        group=cfg.wandb_group,
        name=cfg.wandb_name,
        tags=[cfg.agent] if cfg.agent else [],
        config={
            **asdict(cfg),
            "model_config": MODEL_CONFIG,
            "n_params": n_params,
            "train_samples": len(train_ds),
            "val_samples": {k: len(v) for k, v in val_splits.items()},
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    for _name in VAL_SPLIT_NAMES:
        wandb.define_metric(f"{_name}/*", step_metric="global_step")
    wandb.define_metric("lr", step_metric="global_step")

    model_dir = Path(f"models/model-{run.id}")
    model_dir.mkdir(parents=True)
    model_path = model_dir / "checkpoint.pt"
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(MODEL_CONFIG, f)

    best_val = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()
    amp_dtype = torch.bfloat16

    chan_w = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)

    def compute_loss(pred, y_norm, vol_mask, surf_mask):
        err = pred - y_norm
        sq = err * err * chan_w
        abs_e = err.abs() * chan_w

        vol_w = vol_mask.unsqueeze(-1).float()
        vol_n = vol_w.sum().clamp(min=1) * chan_w.mean()
        vol_l2 = (sq * vol_w).sum() / vol_n
        vol_l1 = (abs_e * vol_w).sum() / vol_n
        vol_loss = cfg.l2_weight * vol_l2 + cfg.l1_weight * vol_l1

        sf_w = surf_mask.unsqueeze(-1).float()
        sf_n = sf_w.sum().clamp(min=1) * chan_w.mean()
        surf_l2 = (sq * sf_w).sum() / sf_n
        surf_l1 = (abs_e * sf_w).sum() / sf_n
        surf_loss = cfg.l2_weight * surf_l2 + cfg.l1_weight * surf_l1

        return vol_loss, surf_loss, vol_l2.detach(), surf_l2.detach()

    # Initial validation (warmstart sanity check) -----
    model.eval()
    with torch.no_grad():
        warmstart_surf_p = 0.0
        for split_name, vloader in val_loaders.items():
            mae_surf_p = 0.0
            n_surf = 0
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device)
                mask = mask.to(device)
                xn = (x - stats["x_mean"]) / stats["x_std"]
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model({"x": xn})["preds"].float()
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                surf_mask = mask & is_surface
                err = (pred_orig[..., 2] - y[..., 2]).abs()
                mae_surf_p += (err * surf_mask.float()).sum().item()
                n_surf += surf_mask.sum().item()
            sp_split = mae_surf_p / max(n_surf, 1)
            warmstart_surf_p += sp_split
            print(f"  warmstart {split_name}/mae_surf_p = {sp_split:.2f}")
        warmstart_surf_p /= 4
        print(f"  WARMSTART avg_surf_p = {warmstart_surf_p:.2f}")
        wandb.log({"val/warmstart_avg_surf_p": warmstart_surf_p, "global_step": 0})

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT:
            print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
            break

        t0 = time.time()
        model.train()
        epoch_vol = epoch_surf = 0.0
        n_batches = 0

        for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                pred = model({"x": x})["preds"]
                vol_loss, surf_loss, vol_l2, surf_l2 = compute_loss(pred, y_norm, vol_mask, surf_mask)
                loss = vol_loss + cfg.surf_weight * surf_loss

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            ema.update(model)

            global_step += 1
            wandb.log({
                "train/loss": loss.item(),
                "train/vol_l2": vol_l2.item(),
                "train/surf_l2": surf_l2.item(),
                "global_step": global_step,
            })

            epoch_vol += vol_l2.item()
            epoch_surf += surf_l2.item()
            n_batches += 1

        scheduler.step()
        epoch_vol /= max(n_batches, 1)
        epoch_surf /= max(n_batches, 1)

        # --- Validate (use EMA weights) ---
        eval_model = copy.deepcopy(model)
        eval_model.load_state_dict(ema.state_dict(model))
        eval_model.eval()

        val_loss_sum = 0.0
        split_metrics: dict[str, dict] = {}

        for split_name, vloader in val_loaders.items():
            val_vol_l2 = val_surf_l2 = 0.0
            mae_surf = torch.zeros(3, device=device)
            mae_vol = torch.zeros(3, device=device)
            n_surf = n_vol = n_vb = 0

            with torch.no_grad():
                for x, y, is_surface, mask in vloader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    is_surface = is_surface.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)

                    x = (x - stats["x_mean"]) / stats["x_std"]
                    y_norm = (y - stats["y_mean"]) / stats["y_std"]

                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = eval_model({"x": x})["preds"]
                    pred = pred.float()
                    sq_err = (pred - y_norm) ** 2

                    vol_mask = mask & ~is_surface
                    surf_mask = mask & is_surface
                    val_vol_l2 += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                    val_surf_l2 += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                    n_vb += 1

                    pred_orig = pred * stats["y_std"] + stats["y_mean"]
                    err = (pred_orig - y).abs()
                    mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                    mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                    n_surf += surf_mask.sum().item()
                    n_vol += vol_mask.sum().item()

            val_vol_l2 /= max(n_vb, 1)
            val_surf_l2 /= max(n_vb, 1)
            split_loss = val_vol_l2 + cfg.surf_weight * val_surf_l2
            mae_surf /= max(n_surf, 1)
            mae_vol /= max(n_vol, 1)

            split_metrics[split_name] = {
                f"{split_name}/vol_loss": val_vol_l2,
                f"{split_name}/surf_loss": val_surf_l2,
                f"{split_name}/loss": split_loss,
                f"{split_name}/mae_vol_Ux": mae_vol[0].item(),
                f"{split_name}/mae_vol_Uy": mae_vol[1].item(),
                f"{split_name}/mae_vol_p": mae_vol[2].item(),
                f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
                f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
                f"{split_name}/mae_surf_p": mae_surf[2].item(),
            }
            val_loss_sum += split_loss

        mean_val_loss = val_loss_sum / len(val_loaders)
        avg_surf_p = sum(split_metrics[s][f"{s}/mae_surf_p"] for s in VAL_SPLIT_NAMES) / 4.0
        dt = time.time() - t0

        metrics = {
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "val/loss": mean_val_loss,
            "val/avg_mae_surf_p": avg_surf_p,
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": dt,
            "global_step": global_step,
        }
        for sm in split_metrics.values():
            metrics.update(sm)
        wandb.log(metrics)

        tag = ""
        # Track best by avg_surf_p (the actual leaderboard metric).
        if avg_surf_p < best_val:
            best_val = avg_surf_p
            best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_surf_p": avg_surf_p}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(eval_model.state_dict(), model_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        split_summary = "  ".join(
            f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
        )
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  lr={scheduler.get_last_lr()[0]:.2e}  "
            f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
            f"val[{split_summary}]  surf_p={avg_surf_p:.2f}{tag}"
        )

        del eval_model

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}, "
              f"avg_surf_p={best_metrics['avg_surf_p']:.2f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        plot_dir = Path("plots") / run.id
        n = 1 if cfg.debug else 4
        for split_name, split_ds in val_splits.items():
            images = visualize(model, split_ds, stats, device, n_samples=n,
                               out_dir=plot_dir / split_name)
            if images:
                wandb.log({
                    f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                    "global_step": global_step,
                })

    if best_metrics and not cfg.debug:
        import subprocess
        print("\nGenerating test predictions...")
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()


if __name__ == "__main__":
    main()
