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
# Model: Voxel U-Net
# ---------------------------------------------------------------------------

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


class VoxelUNet(nn.Module):
    """3D U-Net operating on voxelized velocity field.

    Architecture:
    1. Voxelize input velocity + airfoil mask into 3D grid
    2. U-Net encoder-decoder with skip connections
    3. Output: correction field in voxel space
    4. Trilinear interpolate correction to each point
    5. Add to copy baseline (residual prediction)
    6. Zero out airfoil surface (no-slip BC)

    The voxel grid enforces spatial smoothness by construction —
    the 3D convolutions capture local spatial patterns that per-point MLPs miss.
    """

    def __init__(self, grid_size=(64, 24, 32), base_ch=48):
        super().__init__()
        self.grid_size = grid_size
        in_ch = T_IN * 3 + 1  # velocity(15) + airfoil_mask(1) = 16
        out_ch = T_OUT * 3  # predict 5*3=15 channels

        # Encoder
        self.enc1 = ConvBlock3D(in_ch, base_ch)      # 64×24×32
        self.enc2 = ConvBlock3D(base_ch, base_ch*2)   # 32×12×16
        self.enc3 = ConvBlock3D(base_ch*2, base_ch*4) # 16×6×8

        # Bottleneck
        self.bottleneck = ConvBlock3D(base_ch*4, base_ch*4)  # 8×3×4

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch*4, base_ch*4, 2, stride=2)
        self.dec3 = ConvBlock3D(base_ch*8, base_ch*2)  # concat with enc3

        self.up2 = nn.ConvTranspose3d(base_ch*2, base_ch*2, 2, stride=2)
        self.dec2 = ConvBlock3D(base_ch*4, base_ch)    # concat with enc2

        self.up1 = nn.ConvTranspose3d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock3D(base_ch*2, base_ch)    # concat with enc1

        self.head = nn.Conv3d(base_ch, out_ch, 1)
        # Zero-init head so model starts from copy baseline
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        # Domain bounds
        self.register_buffer("domain_min", torch.tensor([0.0, -0.45, 0.0]))
        self.register_buffer("domain_max", torch.tensor([2.15, 0.45, 1.25]))

        # Per-point refinement MLP (small, operates on concat of voxel features + local features)
        refine_in = out_ch + 3 + T_IN * 3 + 1  # correction(15) + pos(3) + vel(15) + airfoil(1) = 34
        self.refine = nn.Sequential(
            nn.Linear(refine_in, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, out_ch),
        )
        # Zero-init last layer of refinement MLP too
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def points_to_grid(self, pos, features):
        """Scatter-mean points into voxel grid."""
        B, N, C = features.shape
        Gx, Gy, Gz = self.grid_size
        device = pos.device

        pos_norm = (pos - self.domain_min) / (self.domain_max - self.domain_min + 1e-8)
        pos_norm = pos_norm.clamp(0, 1 - 1e-6)

        ix = (pos_norm[..., 0] * Gx).long().clamp(0, Gx - 1)
        iy = (pos_norm[..., 1] * Gy).long().clamp(0, Gy - 1)
        iz = (pos_norm[..., 2] * Gz).long().clamp(0, Gz - 1)

        grid = torch.zeros(B, C, Gx * Gy * Gz, device=device)
        count = torch.zeros(B, 1, Gx * Gy * Gz, device=device)
        flat_idx = ix * Gy * Gz + iy * Gz + iz  # [B, N]

        for b in range(B):
            idx_b = flat_idx[b].unsqueeze(0).expand(C, -1)  # [C, N]
            grid[b].scatter_add_(1, idx_b, features[b].t())
            count[b, 0].scatter_add_(0, flat_idx[b], torch.ones(N, device=device))

        grid = grid / (count + 1e-8)
        return grid.reshape(B, C, Gx, Gy, Gz)

    def grid_to_points(self, grid, pos):
        """Trilinear interpolate grid values to point locations."""
        B, C, Gx, Gy, Gz = grid.shape
        N = pos.shape[1]

        pos_norm = (pos - self.domain_min) / (self.domain_max - self.domain_min + 1e-8)
        pos_norm = pos_norm.clamp(0, 1)

        # grid_sample coords in [-1, 1], format: [B, 1, 1, N, 3] with (z, y, x) ordering
        coords = pos_norm * 2 - 1
        coords = torch.stack([coords[..., 2], coords[..., 1], coords[..., 0]], dim=-1)
        coords = coords.reshape(B, 1, 1, N, 3)

        out = F.grid_sample(grid, coords, mode='bilinear', padding_mode='border', align_corners=False)
        return out.reshape(B, C, N).permute(0, 2, 1)  # [B, N, C]

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        v_flat = velocity_in.reshape(B, N, T * C)

        # Airfoil mask
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                idx = idcs_airfoil[i].to(pos.device)
                airfoil_mask[i, idx, 0] = 1.0

        # Voxelize
        features = torch.cat([v_flat, airfoil_mask], dim=-1)  # [B, N, 16]
        grid = self.points_to_grid(pos, features)  # [B, 16, Gx, Gy, Gz]

        # U-Net
        e1 = self.enc1(grid)           # [B, ch, 64, 24, 32]
        e2 = self.enc2(F.avg_pool3d(e1, 2))  # [B, 2ch, 32, 12, 16]
        e3 = self.enc3(F.avg_pool3d(e2, 2))  # [B, 4ch, 16, 6, 8]
        b = self.bottleneck(F.avg_pool3d(e3, 2))  # [B, 4ch, 8, 3, 4]

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # [B, 2ch, 16, 6, 8]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, ch, 32, 12, 16]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, ch, 64, 24, 32]

        correction_grid = self.head(d1)  # [B, 15, 64, 24, 32]

        # Interpolate correction to point locations
        correction = self.grid_to_points(correction_grid, pos)  # [B, N, 15]

        # Per-point refinement
        refine_in = torch.cat([correction, pos, v_flat, airfoil_mask], dim=-1)  # [B, N, 34]
        refinement = self.refine(refine_in)  # [B, N, 15]
        correction = correction + refinement

        correction = correction.reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)  # [B, 5, N, 3]

        # Residual: add last input
        pred = velocity_in[:, -1:, :, :] + correction

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
    batch_size: int = 2
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    base_ch: int = 48


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

    model = VoxelUNet(base_ch=cfg.base_ch).to(device)

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
