"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Grid-based 3D spatial mixing
# ---------------------------------------------------------------------------

class PointToGrid(nn.Module):
    """Scatter point features onto a regular 3D grid using trilinear splatting."""

    def __init__(self, grid_size=(32, 16, 24)):
        super().__init__()
        self.grid_size = grid_size  # (Gx, Gy, Gz)

    def forward(self, features, pos):
        """
        features: [B, N, C]
        pos: [B, N, 3] — raw positions
        Returns: grid [B, C, Gx, Gy, Gz]
        """
        B, N, C = features.shape
        Gx, Gy, Gz = self.grid_size

        # Normalize positions to [0, 1] per sample
        pos_min = pos.min(dim=1, keepdim=True).values  # [B, 1, 3]
        pos_max = pos.max(dim=1, keepdim=True).values  # [B, 1, 3]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-6)  # [B, N, 3] in [0,1]

        # Map to grid coordinates [0, G-1]
        gx = (pos_norm[..., 0] * (Gx - 1)).clamp(0, Gx - 1)  # [B, N]
        gy = (pos_norm[..., 1] * (Gy - 1)).clamp(0, Gy - 1)
        gz = (pos_norm[..., 2] * (Gz - 1)).clamp(0, Gz - 1)

        # Trilinear splatting
        gx0 = gx.long().clamp(0, Gx - 2)
        gy0 = gy.long().clamp(0, Gy - 2)
        gz0 = gz.long().clamp(0, Gz - 2)
        gx1, gy1, gz1 = gx0 + 1, gy0 + 1, gz0 + 1

        wx = (gx - gx0.float()).unsqueeze(-1)  # [B, N, 1]
        wy = (gy - gy0.float()).unsqueeze(-1)
        wz = (gz - gz0.float()).unsqueeze(-1)

        # Use fp32 for scatter operations (AMP compat)
        feat_f32 = features.float()
        wx_f32, wy_f32, wz_f32 = wx.float(), wy.float(), wz.float()

        grid = torch.zeros(B, Gx, Gy, Gz, C, device=features.device, dtype=torch.float32)
        count = torch.zeros(B, Gx, Gy, Gz, 1, device=features.device, dtype=torch.float32)

        for dx, dwx in [(gx0, 1 - wx_f32), (gx1, wx_f32)]:
            for dy, dwy in [(gy0, 1 - wy_f32), (gy1, wy_f32)]:
                for dz, dwz in [(gz0, 1 - wz_f32), (gz1, wz_f32)]:
                    w = dwx * dwy * dwz  # [B, N, 1]
                    idx = (dx * Gy * Gz + dy * Gz + dz).unsqueeze(-1).expand(-1, -1, C)
                    grid.view(B, -1, C).scatter_add_(1, idx, feat_f32 * w)
                    count.view(B, -1, 1).scatter_add_(1, idx[..., :1], w)

        grid = grid / count.clamp(min=1e-6)
        return grid.permute(0, 4, 1, 2, 3), pos_min, pos_max  # [B, C, Gx, Gy, Gz]


class GridToPoint(nn.Module):
    """Gather features from 3D grid back to points using grid_sample."""

    def forward(self, grid, pos, pos_min, pos_max):
        """
        grid: [B, C, Gx, Gy, Gz]
        pos: [B, N, 3]
        Returns: [B, N, C]
        """
        B, C, Gx, Gy, Gz = grid.shape
        N = pos.shape[1]

        # Normalize to [-1, 1] for grid_sample
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-6)  # [0, 1]
        pos_grid = pos_norm * 2 - 1  # [-1, 1]

        # grid_sample expects [B, C, D, H, W] and grid [B, D_out, H_out, W_out, 3]
        # We sample at N points, so reshape to [B, 1, 1, N, 3]
        sample_grid = pos_grid.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]

        sampled = F.grid_sample(grid, sample_grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        # sampled: [B, C, 1, 1, N]
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # [B, N, C]


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.GELU(),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class Grid3DNet(nn.Module):
    """3D U-Net operating on voxelized point cloud."""

    def __init__(self, in_ch, out_ch, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock3D(in_ch, base_ch)
        self.enc2 = ConvBlock3D(base_ch, base_ch * 2)
        self.enc3 = ConvBlock3D(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock3D(base_ch * 4, base_ch * 2)  # concat skip
        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock3D(base_ch * 2, base_ch)  # concat skip

        self.out_conv = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        e3 = self.enc3(F.avg_pool3d(e2, 2))

        # Decoder with skip connections
        d2 = self.up2(e3)
        # Handle size mismatch from non-power-of-2 grids
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class VelocityPredictor(nn.Module):
    """Hybrid grid + pointwise model for velocity prediction.

    1. Encode per-point features (MLP)
    2. Voxelize onto 3D grid
    3. Apply 3D U-Net for spatial mixing
    4. Interpolate back to points
    5. Combine with pointwise features
    6. Predict residual delta
    7. Enforce no-slip BC
    """

    def __init__(self, hidden=128, n_blocks=4, grid_size=(32, 16, 24),
                 grid_ch=64, dropout=0.1):
        super().__init__()
        in_dim = 3 + T_IN * 3 + (T_IN - 1) * 3  # 30
        out_dim = T_OUT * 3  # 15

        # Pointwise feature encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            *[ResBlock(hidden, dropout) for _ in range(n_blocks)],
        )

        # Grid branch
        self.point_to_grid = PointToGrid(grid_size)
        self.grid_net = Grid3DNet(hidden, grid_ch, base_ch=grid_ch)
        self.grid_to_point = GridToPoint()

        # Combiner: merge pointwise + grid features
        self.combiner = nn.Sequential(
            nn.Linear(hidden + grid_ch, hidden),
            ResBlock(hidden, dropout),
            ResBlock(hidden, dropout),
        )

        # Output head — zero-initialized for copy-last starting point
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        last_vel = velocity_in[:, -1]

        # Temporal differences
        vel_diff = velocity_in[:, 1:] - velocity_in[:, :-1]
        vel_flat = velocity_in.reshape(B, N, T * C)
        diff_flat = vel_diff.reshape(B, N, (T - 1) * C)
        x_in = torch.cat([pos, vel_flat, diff_flat], dim=-1)

        # Pointwise encoding
        point_feat = self.point_encoder(x_in)  # [B, N, hidden]

        # Grid-based spatial mixing
        grid, pos_min, pos_max = self.point_to_grid(point_feat, pos)
        grid = self.grid_net(grid)
        grid_feat = self.grid_to_point(grid, pos, pos_min, pos_max)  # [B, N, grid_ch]

        # Combine
        combined = torch.cat([point_feat, grid_feat], dim=-1)
        x = self.combiner(combined)

        # Predict delta
        delta = self.proj_out(self.norm_out(x)).reshape(B, T_OUT, N, 3)
        out = delta + last_vel.unsqueeze(1)

        # No-slip BC
        for i in range(B):
            out[i, :, idcs_airfoil[i], :] = 0.0

        return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
    import wandb
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import simple_parsing as sp
    import wandb
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))

    @dataclass
    class Config:
        lr: float = 3e-4
        weight_decay: float = 0.01
        batch_size: int = 1
        epochs: int = 60
        splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
        wandb_group: str | None = None
        wandb_name: str | None = None
        agent: str | None = None
        debug: bool = False
        hidden: int = 128
        n_blocks: int = 4
        grid_ch: int = 64
        dropout: float = 0.1

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

    model = VelocityPredictor(
        hidden=cfg.hidden, n_blocks=cfg.n_blocks,
        grid_size=(32, 16, 24), grid_ch=cfg.grid_ch, dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_epochs = 3
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, MAX_EPOCHS - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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

    model_cfg_dict = {"hidden": cfg.hidden, "n_blocks": cfg.n_blocks,
                      "grid_ch": cfg.grid_ch, "dropout": cfg.dropout}
    torch.save(model_cfg_dict, model_dir / "config.pt")

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
                # L2 norm loss — matches competition metric
                loss = (pred - v_out).norm(dim=3).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path),
                    "--config", str(model_dir / "config.pt")]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()
