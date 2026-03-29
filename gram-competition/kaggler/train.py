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
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip(x)


class VoxelConvMLP(nn.Module):
    """Hybrid model: voxel 3D convolutions for spatial context + per-point MLP.

    1. Voxelize points into a 3D grid
    2. Scatter-mean velocity features into grid
    3. Apply 3D conv network for spatial interaction
    4. Read conv features back at each point (trilinear interpolation)
    5. Concatenate with per-point features
    6. Per-point MLP to predict velocity delta
    7. Add to last input (residual prediction)
    8. Zero out airfoil surface (no-slip BC)
    """

    def __init__(self, hidden=256, n_blocks=6, grid_size=(32, 12, 16), conv_ch=64):
        super().__init__()
        self.grid_size = grid_size  # (Gx, Gy, Gz)
        self.conv_ch = conv_ch

        # Per-point feature dim for voxel grid: velocity(5*3=15) + airfoil_indicator(1) = 16
        voxel_in_ch = T_IN * 3 + 1

        # 3D conv network on voxel grid
        self.voxel_conv = nn.Sequential(
            ConvBlock3D(voxel_in_ch, conv_ch),
            ConvBlock3D(conv_ch, conv_ch),
            ConvBlock3D(conv_ch, conv_ch),
        )

        # Per-point MLP: pos(3) + velocity(15) + airfoil(1) + conv_features(conv_ch) = 19 + conv_ch
        in_dim = 3 + T_IN * 3 + 1 + conv_ch
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # Domain bounds (will be set from data)
        self.register_buffer("domain_min", torch.tensor([0.0, -0.45, 0.0]))
        self.register_buffer("domain_max", torch.tensor([2.15, 0.45, 1.25]))

    def voxelize_and_conv(self, pos, features):
        """Voxelize features, run 3D conv, read back at point locations.

        Args:
            pos: [B, N, 3]
            features: [B, N, C]

        Returns:
            conv_features: [B, N, conv_ch]
        """
        B, N, C = features.shape
        Gx, Gy, Gz = self.grid_size
        device = pos.device

        # Normalize positions to [0, 1] within domain
        pos_norm = (pos - self.domain_min) / (self.domain_max - self.domain_min + 1e-8)
        pos_norm = pos_norm.clamp(0, 1 - 1e-6)

        # Compute grid indices
        ix = (pos_norm[..., 0] * Gx).long().clamp(0, Gx - 1)  # [B, N]
        iy = (pos_norm[..., 1] * Gy).long().clamp(0, Gy - 1)
        iz = (pos_norm[..., 2] * Gz).long().clamp(0, Gz - 1)

        # Scatter features into grid (mean aggregation)
        grid = torch.zeros(B, C, Gx, Gy, Gz, device=device)
        count = torch.zeros(B, 1, Gx, Gy, Gz, device=device)

        flat_idx = ix * Gy * Gz + iy * Gz + iz  # [B, N]

        for b in range(B):
            # Scatter add features
            idx = flat_idx[b]  # [N]
            for c in range(C):
                grid[b, c].view(-1).scatter_add_(0, idx, features[b, :, c])
            count[b, 0].view(-1).scatter_add_(0, idx, torch.ones(N, device=device))

        # Average (avoid div by zero)
        grid = grid / (count + 1e-8)

        # 3D convolution
        conv_out = self.voxel_conv(grid)  # [B, conv_ch, Gx, Gy, Gz]

        # Read back features at each point using grid_sample (trilinear interpolation)
        # grid_sample expects input in [-1, 1] range
        grid_coords = pos_norm * 2 - 1  # [B, N, 3] in [-1, 1]
        # grid_sample expects [B, D, H, W, 3] grid, with (x,y,z) -> (W,H,D) mapping
        # Our grid is [B, C, Gx, Gy, Gz] where Gx=depth, Gy=height, Gz=width
        # So grid_sample coords should be: (z->W, y->H, x->D)
        grid_coords_5d = torch.stack([grid_coords[..., 2], grid_coords[..., 1], grid_coords[..., 0]], dim=-1)
        grid_coords_5d = grid_coords_5d.reshape(B, 1, 1, N, 3)  # [B, 1, 1, N, 3]

        conv_features = F.grid_sample(
            conv_out, grid_coords_5d,
            mode='bilinear', padding_mode='border', align_corners=False,
        )  # [B, conv_ch, 1, 1, N]
        conv_features = conv_features.reshape(B, self.conv_ch, N).permute(0, 2, 1)  # [B, N, conv_ch]

        return conv_features

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        v_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]

        # Binary airfoil indicator
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                idx = idcs_airfoil[i].to(pos.device)
                airfoil_mask[i, idx, 0] = 1.0

        # Voxel features for conv
        voxel_features = torch.cat([v_flat, airfoil_mask], dim=-1)  # [B, N, 16]
        conv_features = self.voxelize_and_conv(pos, voxel_features)  # [B, N, conv_ch]

        # Concatenate all per-point features
        x = torch.cat([pos, v_flat, airfoil_mask, conv_features], dim=-1)

        x = self.proj_in(x)
        x = self.blocks(x)
        delta = self.proj_out(x)  # [B, N, 15]
        delta = delta.reshape(B, T_OUT, N, 3)

        # Residual: add last input timestep
        last_v = velocity_in[:, -1:, :, :]
        pred = last_v + delta

        # No-slip BC
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                idx = idcs_airfoil[i].to(pred.device)
                pred[i, :, idx, :] = 0.0

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

                with torch.amp.autocast("cuda"):
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
# Config
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))  # minutes


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    hidden: int = 256
    n_blocks: int = 6
    conv_ch: int = 64


if __name__ == "__main__":
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

    model = VoxelConvMLP(
        hidden=cfg.hidden, n_blocks=cfg.n_blocks, conv_ch=cfg.conv_ch,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

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

            with torch.amp.autocast("cuda"):
                pred = model(v_in, pos, t, idcs)
                loss = (pred - v_out).pow(2).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

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
