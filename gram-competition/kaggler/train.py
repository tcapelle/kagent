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
# Voxel-based spatial interaction
# ---------------------------------------------------------------------------

class VoxelEncoder(nn.Module):
    """
    Bin point features into a 3D voxel grid, run 3D CNN, interpolate back.
    This gives spatial context to each point efficiently.
    """
    def __init__(self, in_dim, voxel_dim=64, grid_size=(32, 8, 16), n_conv_blocks=3):
        super().__init__()
        self.grid_size = grid_size  # (Gx, Gy, Gz)
        Gx, Gy, Gz = grid_size

        # Project per-point features to voxel features
        self.point_to_voxel = nn.Linear(in_dim, voxel_dim)

        # 3D CNN for spatial interaction
        layers = []
        for i in range(n_conv_blocks):
            layers.append(nn.Conv3d(voxel_dim, voxel_dim, kernel_size=3, padding=1))
            layers.append(nn.GroupNorm(8, voxel_dim))
            layers.append(nn.GELU())
        self.cnn = nn.Sequential(*layers)

        # The output is voxel_dim features per point (interpolated from grid)

    def forward(self, point_features, pos, pos_min, pos_max):
        """
        point_features: [B, N, in_dim]
        pos: [B, N, 3]
        pos_min, pos_max: [3] bounds for the grid
        Returns: [B, N, voxel_dim] spatial features for each point
        """
        B, N, D = point_features.shape
        Gx, Gy, Gz = self.grid_size
        device = point_features.device

        # Project features
        feats = self.point_to_voxel(point_features)  # [B, N, voxel_dim]
        voxel_dim = feats.shape[-1]

        # Normalize positions to [0, 1] range
        pos_range = (pos_max - pos_min).clamp(min=1e-6)
        pos_norm = (pos - pos_min.view(1, 1, 3)) / pos_range.view(1, 1, 3)  # [B, N, 3]
        pos_norm = pos_norm.clamp(0, 1 - 1e-6)

        # Compute voxel indices
        vx = (pos_norm[:, :, 0] * Gx).long().clamp(0, Gx - 1)
        vy = (pos_norm[:, :, 1] * Gy).long().clamp(0, Gy - 1)
        vz = (pos_norm[:, :, 2] * Gz).long().clamp(0, Gz - 1)
        voxel_idx = vx * (Gy * Gz) + vy * Gz + vz  # [B, N]

        # Scatter features into voxel grid (mean pooling)
        grid_flat = torch.zeros(B, Gx * Gy * Gz, voxel_dim, device=device)
        count = torch.zeros(B, Gx * Gy * Gz, 1, device=device)

        # Use scatter_add for efficiency
        idx_expanded = voxel_idx.unsqueeze(-1).expand(B, N, voxel_dim)  # [B, N, voxel_dim]
        grid_flat.scatter_add_(1, idx_expanded, feats)
        count.scatter_add_(1, voxel_idx.unsqueeze(-1), torch.ones(B, N, 1, device=device))

        grid_flat = grid_flat / count.clamp(min=1)  # mean pooling

        # Reshape to 3D grid: [B, voxel_dim, Gx, Gy, Gz]
        grid_3d = grid_flat.permute(0, 2, 1).reshape(B, voxel_dim, Gx, Gy, Gz)

        # Apply 3D CNN
        grid_3d = grid_3d + self.cnn(grid_3d)  # residual connection

        # Interpolate back to each point using trilinear interpolation
        # grid_sample needs [B, C, D, H, W] and grid in [-1, 1]
        grid_coords = pos_norm * 2 - 1  # [B, N, 3] in [-1, 1]
        # grid_sample expects [B, 1, 1, N, 3] with (x, y, z) ordering
        grid_coords = grid_coords.view(B, 1, 1, N, 3)

        spatial_feats = F.grid_sample(
            grid_3d, grid_coords,
            mode='bilinear', padding_mode='border', align_corners=True,
        )  # [B, voxel_dim, 1, 1, N]
        spatial_feats = spatial_feats.squeeze(2).squeeze(2).permute(0, 2, 1)  # [B, N, voxel_dim]

        return spatial_feats


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


class AirflowModel(nn.Module):
    """
    Per-point MLP with voxel-based spatial interaction.
    Predicts residual from temporal mean of input velocities.
    """

    def __init__(self, hidden=384, n_blocks=8, voxel_dim=64,
                 grid_size=(32, 8, 16), n_conv_blocks=3,
                 vel_mean=None, vel_std=None):
        super().__init__()
        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

        # Domain bounds (fixed for this dataset)
        self.register_buffer("pos_min", torch.tensor([0.0, -0.41, 0.0]))
        self.register_buffer("pos_max", torch.tensor([2.1, 0.41, 1.22]))

        # Per-point input: pos(3) + vel_in_norm(15) + vel_dev(15) = 33
        in_dim = 3 + T_IN * 3 + T_IN * 3
        out_dim = T_OUT * 3

        # Voxel encoder for spatial features
        self.voxel_encoder = VoxelEncoder(
            in_dim=in_dim, voxel_dim=voxel_dim,
            grid_size=grid_size, n_conv_blocks=n_conv_blocks,
        )

        # Main MLP: per-point features + spatial features
        self.proj_in = nn.Linear(in_dim + voxel_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # Zero-init for residual learning
        nn.init.zeros_(self.proj_out[-1].weight)
        nn.init.zeros_(self.proj_out[-1].bias)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocity
        vel_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)

        # Temporal mean
        vel_mean_t = vel_norm.mean(dim=1, keepdim=True)

        # Deviation from mean
        vel_dev = vel_norm - vel_mean_t

        vel_flat = vel_norm.reshape(B, N, T * C)
        dev_flat = vel_dev.reshape(B, N, T * C)

        point_feats = torch.cat([pos, vel_flat, dev_flat], dim=-1)  # [B, N, 33]

        # Get spatial features from voxel encoder
        spatial_feats = self.voxel_encoder(point_feats, pos, self.pos_min, self.pos_max)

        # Combine per-point and spatial features
        x = torch.cat([point_feats, spatial_feats], dim=-1)  # [B, N, 33 + voxel_dim]

        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x).reshape(B, T_OUT, N, 3)

        # Residual from temporal mean
        out_norm = delta_norm + vel_mean_t

        # Denormalize
        out = out_norm * (self.vel_std + 1e-8) + self.vel_mean

        # No-slip boundary condition
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                out[i, :, idcs_airfoil[i].to(out.device), :] = 0.0

        return out


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

def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points=30000):
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
    batch_size: int = 1
    epochs: int = 100
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    subsample_train: int = 0  # 0 = use all points
    hidden: int = 384
    n_blocks: int = 8
    voxel_dim: int = 64


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

    model = AirflowModel(
        hidden=cfg.hidden, n_blocks=cfg.n_blocks,
        voxel_dim=cfg.voxel_dim,
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
                loss = (pred - v_out_s).pow(2).mean()

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
