"""Train a 3D airflow velocity predictor.

Transolver: per-point encoder -> L Transolver blocks with soft slice attention
-> per-point decoder that predicts a residual delta from the last input step.
Zero velocity at airfoil (no-slip BC) is enforced.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import math
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------


def fourier_features(x: torch.Tensor, num_freqs: int, scale: float = 1.0) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi * scale
    xf = x.unsqueeze(-1) * freqs  # [..., C, F]
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1)  # [..., C, 2F]
    enc = enc.reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


# ---------------------------------------------------------------------------
# Transolver block
# ---------------------------------------------------------------------------


class TransolverBlock(nn.Module):
    """Soft-slice attention:
       1) Point -> M slices via softmax(Wx)
       2) Aggregate points into slice features
       3) Self-attention over M slices
       4) Broadcast slice features back to points using same assignments
    """

    def __init__(self, d, heads=8, slices=64, mlp_mult=2, dropout=0.0):
        super().__init__()
        self.d = d
        self.heads = heads
        self.slices = slices
        self.head_dim = d // heads
        assert d % heads == 0

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Per-head slice assignment logits.
        self.to_slice_logits = nn.Linear(d, heads * slices)
        self.temp = nn.Parameter(torch.tensor(1.0))

        self.to_v = nn.Linear(d, d)

        # Slice self-attention: each head runs its own QKV over its slice set.
        self.to_qkv = nn.Linear(self.head_dim, self.head_dim * 3)

        self.out = nn.Linear(d, d)

        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d * mlp_mult, d),
        )

    def forward(self, x):
        # x: [B, N, d]
        B, N, d = x.shape
        H, M = self.heads, self.slices

        h0 = self.ln1(x)
        logits = self.to_slice_logits(h0) / self.temp.clamp(min=0.1)
        logits = rearrange(logits, 'b n (h m) -> b h n m', h=H, m=M)
        assign = logits.softmax(dim=-1)  # point -> slice, sums to 1 per point

        v = self.to_v(h0)
        v = rearrange(v, 'b n (h c) -> b h n c', h=H)  # [B, H, N, C]

        # Aggregate points into slices (weighted mean).
        # slice_j = sum_i (w_ij * v_i) / sum_i w_ij
        num = torch.einsum('bhnm,bhnc->bhmc', assign, v)                      # [B, H, M, C]
        denom = assign.sum(dim=2).unsqueeze(-1).clamp(min=1e-4)               # [B, H, M, 1]
        slices = num / denom

        # Self-attention among M slices (per head).
        qkv = self.to_qkv(slices)  # [B, H, M, 3C]
        q, k, v2 = qkv.chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        slices_out = attn @ v2  # [B, H, M, C]

        # Broadcast slice -> points using same assignment.
        out = torch.einsum('bhnm,bhmc->bhnc', assign, slices_out)
        out = rearrange(out, 'b h n c -> b n (h c)')
        out = self.out(out)

        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TransolverModel(nn.Module):
    def __init__(self, hidden=256, n_blocks=6, heads=8, slices=64,
                 num_pos_freqs=10, num_vel_freqs=3, dropout=0.0,
                 vel_mean=None, vel_std=None):
        super().__init__()
        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vin_dim = T_IN * 3
        vin_fourier_dim = 3 * 2 * num_vel_freqs
        in_dim = pos_dim + vin_dim + vin_fourier_dim + 1

        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([
            TransolverBlock(hidden, heads=heads, slices=slices, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, T_OUT * 3),
        )
        nn.init.zeros_(self.proj_out[1].weight)
        nn.init.zeros_(self.proj_out[1].bias)

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        last_v = velocity_in[:, -1]

        v_norm = (velocity_in - self.vel_mean) / self.vel_std

        pos_feat = fourier_features(pos, self.num_pos_freqs, scale=1.0)
        vin_flat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T_IN * 3)
        last_norm = v_norm[:, -1]
        freqs = 2.0 ** torch.arange(
            self.num_vel_freqs, device=pos.device, dtype=pos.dtype
        ) * math.pi
        vf = last_norm.unsqueeze(-1) * freqs
        vin_fourier = torch.cat([vf.sin(), vf.cos()], dim=-1).reshape(B, N, -1)

        airfoil_ind = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            airfoil_ind[b, idx.to(pos.device).long(), 0] = 1.0

        x = torch.cat([pos_feat, vin_flat, vin_fourier, airfoil_ind], dim=-1)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)

        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_v.unsqueeze(1) + delta

        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx.to(pred.device).long(), :] = 0.0

        return pred


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
    model.eval()
    val_metrics: dict[str, dict] = {}
    for split_name, vloader in val_loaders.items():
        total_l2 = 0.0
        total_mae = torch.zeros(3, device=device, dtype=torch.float64)
        n_samples = 0
        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(v_in, pos, t, idcs)
                pred = pred.float()
                l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
                total_l2 += l2_err.sum().item()
                mae = (pred - v_out).abs().mean(dim=(1, 2))
                total_mae += mae.double().sum(dim=0)
                n_samples += v_in.shape[0]
        mean_l2 = total_l2 / max(n_samples, 1)
        mean_mae = total_mae / max(n_samples, 1)
        val_metrics[split_name] = {
            f"{split_name}/l2_error": mean_l2,
            f"{split_name}/mae_Ux": mean_mae[0].item(),
            f"{split_name}/mae_Uy": mean_mae[1].item(),
            f"{split_name}/mae_Uz": mean_mae[2].item(),
        }
    mean_val = sum(m[f"{k}/l2_error"] for k, m in val_metrics.items()) / len(val_metrics)
    metrics = {"val/l2_error": mean_val, "global_step": global_step}
    for sm in val_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)
    return mean_val, val_metrics


# ---------------------------------------------------------------------------
# Config + training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 7e-4
    weight_decay: float = 1e-5
    batch_size: int = 1
    epochs: int = 90
    # Subsample points during training for speed + regularization.
    subsample_points: int = 24000
    hidden: int = 384
    n_blocks: int = 8
    heads: int = 8
    slices: int = 128
    num_pos_freqs: int = 10
    num_vel_freqs: int = 3
    dropout: float = 0.05
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


def subsample_batch(v_in, v_out, pos, idcs_airfoil, k):
    B, T, N, C = v_in.shape
    if k is None or k >= N:
        return v_in, v_out, pos, idcs_airfoil
    new_idcs = []
    perm = torch.stack([torch.randperm(N, device=v_in.device)[:k] for _ in range(B)])
    v_in_s = torch.gather(v_in, 2, perm[:, None, :, None].expand(-1, T, -1, C))
    v_out_s = torch.gather(v_out, 2, perm[:, None, :, None].expand(-1, T_OUT, -1, C))
    pos_s = torch.gather(pos, 1, perm[:, :, None].expand(-1, -1, C))
    for b in range(B):
        af = idcs_airfoil[b].to(v_in.device).long()
        mask_full = torch.zeros(N, dtype=torch.bool, device=v_in.device)
        mask_full[af] = True
        mask_sub = mask_full[perm[b]]
        new_idcs.append(mask_sub.nonzero(as_tuple=False).squeeze(-1))
    return v_in_s, v_out_s, pos_s, new_idcs


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = TransolverModel(
        hidden=cfg.hidden, n_blocks=cfg.n_blocks, heads=cfg.heads, slices=cfg.slices,
        num_pos_freqs=cfg.num_pos_freqs, num_vel_freqs=cfg.num_vel_freqs,
        dropout=cfg.dropout,
        vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-gram"),
        group=cfg.wandb_group or RESEARCH_TAG,
        name=cfg.wandb_name,
        tags=[t for t in [cfg.agent, RESEARCH_TAG] if t],
        config={**asdict(cfg), "n_params": n_params,
                "train_samples": len(train_ds),
                "val_samples": {k: len(v) for k, v in val_splits.items()}},
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    KAGGLER_NAME = os.environ.get("KAGGLER_NAME", cfg.agent or "local")
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}/{KAGGLER_NAME}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    model_path = pvc_dir / "checkpoint.pt"
    git_ckpt_path = Path("checkpoints/best.pt")
    git_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT:
            print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
            break

        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for v_in, v_out, pos, t, idcs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            v_in_s, v_out_s, pos_s, idcs_s = subsample_batch(
                v_in, v_out, pos, idcs, cfg.subsample_points
            )

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(v_in_s, pos_s, t, idcs_s)
                loss = (pred.float() - v_out_s).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": scheduler.get_last_lr()[0],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(model.state_dict(), model_path)
            shutil.copyfile(model_path, git_ckpt_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
            f"train={epoch_loss:.4f}  val/l2={mean_val:.4f}{tag}"
        )

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/l2_error={best_metrics['val_l2_error']:.4f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

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
