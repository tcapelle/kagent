"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from flash_attn import flash_attn_func
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Transformer model with flash attention for point cloud
# ---------------------------------------------------------------------------

class PointTransformerBlock(nn.Module):
    """Transformer block using flash attention for point-to-point interaction."""

    def __init__(self, dim, n_heads=8, mlp_ratio=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim % n_heads == 0

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        # Self-attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, N, H, D_head]

        # Flash attention expects [B, N, H, D_head]
        attn_out = flash_attn_func(q, k, v)  # [B, N, H, D_head]
        attn_out = attn_out.reshape(B, N, D)
        x = x + self.proj(attn_out)

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class AirflowTransformer(nn.Module):
    """
    Point cloud transformer for airflow prediction.
    - Projects per-point features to hidden dim
    - Applies transformer blocks with flash attention for spatial interaction
    - Residual prediction from last input timestep
    - No-slip boundary enforcement
    """

    def __init__(self, hidden=256, n_heads=8, n_transformer_blocks=4, n_mlp_blocks=2,
                 vel_mean=None, vel_std=None):
        super().__init__()
        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

        # Input: pos(3) + vel_in_norm(15) + vel_diff(12) + vel_accel(9) = 39
        in_dim = 3 + T_IN * 3 + (T_IN - 1) * 3 + (T_IN - 2) * 3
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)

        # Pre-transformer MLP blocks (per-point feature extraction)
        self.pre_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_mlp_blocks)])

        # Transformer blocks (spatial interaction)
        self.transformer_blocks = nn.Sequential(
            *[PointTransformerBlock(hidden, n_heads=n_heads) for _ in range(n_transformer_blocks)]
        )

        # Post-transformer MLP blocks (per-point prediction)
        self.post_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_mlp_blocks)])

        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # Zero-init for residual
        nn.init.zeros_(self.proj_out[-1].weight)
        nn.init.zeros_(self.proj_out[-1].bias)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocity
        vel_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)
        vel_flat = vel_norm.reshape(B, N, T * C)

        # Temporal derivatives
        vel_diff = vel_norm[:, 1:, :, :] - vel_norm[:, :-1, :, :]
        vel_diff_flat = vel_diff.reshape(B, N, (T - 1) * C)

        vel_accel = vel_diff[:, 1:, :, :] - vel_diff[:, :-1, :, :]
        vel_accel_flat = vel_accel.reshape(B, N, (T - 2) * C)

        # Build input features
        x = torch.cat([pos, vel_flat, vel_diff_flat, vel_accel_flat], dim=-1)

        x = self.proj_in(x)
        x = self.pre_blocks(x)
        x = self.transformer_blocks(x)  # spatial interaction via attention
        x = self.post_blocks(x)

        delta_norm = self.proj_out(x).reshape(B, T_OUT, N, 3)

        # Residual: add last normalized input
        last_norm = vel_norm[:, -1:, :, :]
        out_norm = delta_norm + last_norm

        # Denormalize
        out = out_norm * (self.vel_std + 1e-8) + self.vel_mean

        # No-slip boundary condition
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                out[i, :, idcs_airfoil[i].to(out.device), :] = 0.0

        return out


# Alias for predict.py compatibility
AirflowModel = AirflowTransformer


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

                with torch.cuda.amp.autocast():
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
# Point subsampling
# ---------------------------------------------------------------------------

def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points=8192):
    B, T, N, C = v_in.shape
    device = v_in.device

    v_in_sub, v_out_sub, pos_sub, idcs_sub = [], [], [], []

    for i in range(B):
        weights = torch.ones(N, device=device)
        airfoil_idx = idcs_airfoil[i]
        if airfoil_idx is not None and len(airfoil_idx) > 0:
            weights[airfoil_idx.to(device)] = 3.0

        idx = torch.multinomial(weights, n_points, replacement=False).sort().values

        v_in_sub.append(v_in[i, :, idx, :])
        v_out_sub.append(v_out[i, :, idx, :])
        pos_sub.append(pos[i, idx, :])

        if airfoil_idx is not None and len(airfoil_idx) > 0:
            mask = torch.isin(idx, airfoil_idx.to(device))
            idcs_sub.append(torch.where(mask)[0])
        else:
            idcs_sub.append(torch.tensor([], dtype=torch.long, device=device))

    return torch.stack(v_in_sub), torch.stack(v_out_sub), torch.stack(pos_sub), idcs_sub


# ---------------------------------------------------------------------------
# Config + main
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    subsample_train: int = 8192
    hidden: int = 256
    n_heads: int = 8
    n_transformer_blocks: int = 4
    n_mlp_blocks: int = 2
    n_blocks: int = 8  # kept for predict.py compat


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=1, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    vel_mean = stats["vel_mean"].to(device)
    vel_std = stats["vel_std"].to(device)

    model = AirflowTransformer(
        hidden=cfg.hidden, n_heads=cfg.n_heads,
        n_transformer_blocks=cfg.n_transformer_blocks,
        n_mlp_blocks=cfg.n_mlp_blocks,
        vel_mean=vel_mean, vel_std=vel_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS,
        pct_start=0.1, anneal_strategy='cos',
    )
    scaler = torch.cuda.amp.GradScaler()

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

    model_dir = Path(f"models/model-{run.id}")
    model_dir.mkdir(parents=True)
    model_path = model_dir / "checkpoint.pt"

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

            if cfg.subsample_train and cfg.subsample_train < N_POINTS:
                v_in_s, v_out_s, pos_s, idcs_s = subsample_batch(
                    v_in, v_out, pos, idcs, n_points=cfg.subsample_train
                )
            else:
                v_in_s, v_out_s, pos_s, idcs_s = v_in, v_out, pos, idcs

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = model(v_in_s, pos_s, t, idcs_s)
                # L2 competition loss
                loss = (pred - v_out_s).norm(dim=3).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": optimizer.param_groups[0]['lr'],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(model.state_dict(), model_path)
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

    return cfg, model_path


if __name__ == "__main__":
    main()
