"""Train Transolver on TandemFoilSet with bf16 + subsample + L1 loss.

Recipe (proven from prior frieren runs that hit val/loss ~1.0 / surf_p ~40):
  - Transolver 192h x 6L, slice_num=64
  - bf16 autocast on forward+loss
  - Subsample to N_SUB nodes (keeping all surface nodes) for memory headroom
  - L1 loss on volume + L1 loss on surface * surf_weight
  - Extra pressure-channel weight (pressure is the primary metric)
  - Cosine LR with warmup, AdamW, gradient clip 1.0
  - Optional resume from a previous checkpoint for chained finetuning

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
  python train.py --agent <name> --wandb_name "<name>/<desc>" --resume <ckpt.pt>
"""

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from viz import visualize


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
    def __init__(self, space_dim=2, n_layers=6, n_hidden=192, dropout=0.0,
                 n_head=6, act="gelu", mlp_ratio=2, fun_dim=22, out_dim=3,
                 slice_num=64, ref=8, unified_pos=False,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

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
# Subsampling wrapper: reduces N_max for batching but keeps all surface nodes
# ---------------------------------------------------------------------------
class SubsampleDataset(Dataset):
    def __init__(self, base, n_sub: int):
        self.base = base
        self.n_sub = n_sub

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, sf = self.base[idx]
        N = x.shape[0]
        if self.n_sub <= 0 or N <= self.n_sub:
            return x, y, sf
        surf_idx = torch.nonzero(sf, as_tuple=False).squeeze(-1)
        vol_idx = torch.nonzero(~sf, as_tuple=False).squeeze(-1)
        n_keep_vol = max(self.n_sub - surf_idx.numel(), 0)
        if n_keep_vol > 0 and n_keep_vol < vol_idx.numel():
            perm = torch.randperm(vol_idx.numel())[:n_keep_vol]
            vol_idx = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_idx])
        return x[keep], y[keep], sf[keep]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    surf_weight: float = 10.0
    p_weight: float = 3.0  # extra weight on pressure channel
    epochs: int = 60
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    n_sub: int = 30000  # subsample size during training (0 = no subsample)
    val_n_sub: int = 0  # no subsample at val
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    resume: str | None = None
    debug: bool = False
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    slice_num: int = 64
    mlp_ratio: int = 2


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds_full, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

train_ds = SubsampleDataset(train_ds_full, cfg.n_sub) if cfg.n_sub > 0 else train_ds_full
val_dss = {k: (SubsampleDataset(v, cfg.val_n_sub) if cfg.val_n_sub > 0 else v) for k, v in val_splits.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_dss.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=cfg.n_hidden,
    n_layers=cfg.n_layers,
    n_head=cfg.n_head,
    slice_num=cfg.slice_num,
    mlp_ratio=cfg.mlp_ratio,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.1f}M params")

if cfg.resume:
    print(f"Resuming from {cfg.resume}")
    state = torch.load(cfg.resume, map_location=device, weights_only=True)
    model.load_state_dict(state)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def lr_lambda(epoch):
    # warmup then cosine
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, MAX_EPOCHS - cfg.warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    project=os.environ.get("WANDB_PROJECT", "kagent-v2"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={
        **asdict(cfg),
        "model_config": model_config,
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

# Save under PVC for durability + local symlink for git
pvc_dir = Path(f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG','default')}/{cfg.agent or 'unknown'}/checkpoints/model-{run.id}")
pvc_dir.mkdir(parents=True, exist_ok=True)
model_path = pvc_dir / "checkpoint.pt"
with open(pvc_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

# Per-channel weight tensor (shape [3]) — pressure gets extra
chan_weights = torch.tensor([1.0, 1.0, cfg.p_weight], device=device).view(1, 1, 3)


def loss_fn(pred, y_norm, surf_mask, vol_mask):
    abs_err = (pred - y_norm).abs() * chan_weights
    vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1) / chan_weights.mean()
    surf_loss = (abs_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1) / chan_weights.mean()
    loss = vol_loss + cfg.surf_weight * surf_loss
    return loss, vol_loss, surf_loss


best_val = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()
amp_dtype = torch.bfloat16

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT:
        print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = epoch_loss = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x_n = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            pred = model({"x": x_n})["preds"]
            loss, vol_loss, surf_loss = loss_fn(pred, y_norm, surf_mask, vol_mask)

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(),
                   "train/vol_loss_step": vol_loss.item(),
                   "train/surf_loss_step": surf_loss.item(),
                   "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches
    epoch_loss /= n_batches

    # --- Validate ---
    model.eval()
    val_loss_sum = 0.0
    split_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = n_vb = 0

        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x_n = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]
                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model({"x": x_n})["preds"]
                pred = pred.float()

                _, vol_l, surf_l = loss_fn(pred, y_norm, surf_mask, vol_mask)
                val_vol += vol_l.item()
                val_surf += surf_l.item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vb, 1)
        val_surf /= max(n_vb, 1)
        split_loss = val_vol + cfg.surf_weight * val_surf
        mae_surf /= max(n_surf, 1)
        mae_vol /= max(n_vol, 1)

        split_metrics[split_name] = {
            f"{split_name}/vol_loss": val_vol,
            f"{split_name}/surf_loss": val_surf,
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
    avg_surf_p = sum(split_metrics[k][f"{k}/mae_surf_p"] for k in split_metrics) / len(split_metrics)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/loss_epoch": epoch_loss,
        "val/loss": mean_val_loss,
        "val/avg_surf_p": avg_surf_p,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    if mean_val_loss < best_val:
        best_val = mean_val_loss
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_surf_p": avg_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        # Also mirror to local for git tracking
        local_ckpt = Path("checkpoints/best.pt")
        local_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), local_ckpt)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}]  surf_p={avg_surf_p:.2f}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}, avg_surf_p={best_metrics['avg_surf_p']:.2f}")
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

# --- Auto-submit predictions ---
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
