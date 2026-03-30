"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import time
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Grid operations
# ---------------------------------------------------------------------------

def splat_to_grid(features, pos, grid_size):
    """Trilinear splatting of point features onto a 3D grid.
    features: [B, N, C], pos: [B, N, 3]
    Returns: grid [B, C, Gx, Gy, Gz], pos_min [B,1,3], pos_max [B,1,3]
    """
    B, N, C = features.shape
    Gx, Gy, Gz = grid_size

    pos_min = pos.min(dim=1, keepdim=True).values
    pos_max = pos.max(dim=1, keepdim=True).values
    pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-6)

    gcoords = [
        (pos_norm[..., i] * (g - 1)).clamp(0, g - 1) for i, g in enumerate([Gx, Gy, Gz])
    ]
    g0 = [gc.long().clamp(0, g - 2) for gc, g in zip(gcoords, [Gx, Gy, Gz])]
    g1 = [gc + 1 for gc in g0]

    weights = [(gc - gc0.float()).unsqueeze(-1) for gc, gc0 in zip(gcoords, g0)]

    feat_f32 = features.float()
    grid = torch.zeros(B, Gx, Gy, Gz, C, device=features.device, dtype=torch.float32)
    count = torch.zeros(B, Gx, Gy, Gz, 1, device=features.device, dtype=torch.float32)

    for ix, (dx, dwx) in enumerate([(g0[0], 1 - weights[0]), (g1[0], weights[0])]):
        for iy, (dy, dwy) in enumerate([(g0[1], 1 - weights[1]), (g1[1], weights[1])]):
            for iz, (dz, dwz) in enumerate([(g0[2], 1 - weights[2]), (g1[2], weights[2])]):
                w = (dwx * dwy * dwz).float()
                idx = (dx * Gy * Gz + dy * Gz + dz).unsqueeze(-1).expand(-1, -1, C)
                grid.view(B, -1, C).scatter_add_(1, idx, feat_f32 * w)
                count.view(B, -1, 1).scatter_add_(1, idx[..., :1], w)

    grid = grid / count.clamp(min=1e-6)
    return grid.permute(0, 4, 1, 2, 3), pos_min, pos_max


def sample_from_grid(grid, pos, pos_min, pos_max):
    """Sample grid features at point positions using trilinear interpolation.
    grid: [B, C, Gx, Gy, Gz], pos: [B, N, 3]
    Returns: [B, N, C]
    """
    pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-6)
    pos_grid = (pos_norm * 2 - 1).float()
    sample_pts = pos_grid.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N, 3]
    sampled = F.grid_sample(grid.float(), sample_pts, mode='bilinear',
                            padding_mode='border', align_corners=True)
    return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)  # [B, N, C]


# ---------------------------------------------------------------------------
# 3D U-Net
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.gelu(self.gn1(self.conv1(x)))
        h = F.gelu(self.gn2(self.conv2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    """3-level U-Net for 3D grid processing."""

    def __init__(self, in_ch, out_ch, base_ch=64):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.enc1 = ConvBlock3D(in_ch, c1)
        self.enc2 = ConvBlock3D(c1, c2)
        self.bottleneck = ConvBlock3D(c2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock3D(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock3D(c1 * 2, c1)

        self.out_conv = nn.Conv3d(c1, out_ch, 1)
        # Zero-init output so residual starts at 0
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        bn = self.bottleneck(F.avg_pool3d(e2, 2))

        d2 = self.up2(bn)
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
    """Pure grid-based velocity field prediction.

    1. Splat velocity features onto 3D grid
    2. Process with 3D U-Net for spatial mixing
    3. Sample back to original point positions
    4. Combine with point-level MLP for fine details
    5. Residual prediction from last timestep
    """

    def __init__(self, grid_size=(64, 32, 48), base_ch=96,
                 point_hidden=256, n_point_blocks=4, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        in_ch = T_IN * 3 + (T_IN - 1) * 3  # velocity(15) + diffs(12) = 27
        out_ch = T_OUT * 3  # 15

        # Grid U-Net: spatial mixing on the velocity field
        self.grid_unet = UNet3D(in_ch, out_ch, base_ch=base_ch)

        # Pointwise refinement MLP
        point_in = 3 + in_ch + out_ch  # pos(3) + input_features(27) + grid_output(15) = 45
        self.point_mlp = nn.Sequential(
            nn.Linear(point_in, point_hidden),
            *[ResBlock(point_hidden, dropout) for _ in range(n_point_blocks)],
            nn.LayerNorm(point_hidden),
        )
        self.point_out = nn.Linear(point_hidden, out_ch)
        nn.init.zeros_(self.point_out.weight)
        nn.init.zeros_(self.point_out.bias)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        last_vel = velocity_in[:, -1]

        vel_diff = velocity_in[:, 1:] - velocity_in[:, :-1]
        vel_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]
        diff_flat = vel_diff.reshape(B, N, (T - 1) * C)  # [B, N, 12]
        features = torch.cat([vel_flat, diff_flat], dim=-1)  # [B, N, 27]

        # Grid processing
        grid, pos_min, pos_max = splat_to_grid(features, pos, self.grid_size)
        grid_delta = self.grid_unet(grid)  # [B, 15, Gx, Gy, Gz]
        grid_output = sample_from_grid(grid_delta, pos, pos_min, pos_max)  # [B, N, 15]

        # Pointwise refinement
        point_input = torch.cat([pos, features, grid_output], dim=-1)
        point_feat = self.point_mlp(point_input)
        point_delta = self.point_out(point_feat).reshape(B, T_OUT, N, 3)

        # Combine: grid prediction + point refinement + residual
        delta = grid_output.reshape(B, T_OUT, N, 3) + point_delta
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


def augment_batch(v_in, v_out, pos):
    """Random spatial flipping."""
    if torch.rand(1).item() < 0.5:
        pos = pos.clone()
        pos[..., 1] = -pos[..., 1]
        v_in = v_in.clone()
        v_in[..., 1] = -v_in[..., 1]
        v_out = v_out.clone()
        v_out[..., 1] = -v_out[..., 1]
    if torch.rand(1).item() < 0.5:
        pos = pos.clone()
        pos[..., 2] = -pos[..., 2]
        v_in = v_in.clone()
        v_in[..., 2] = -v_in[..., 2]
        v_out = v_out.clone()
        v_out[..., 2] = -v_out[..., 2]
    return v_in, v_out, pos


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
        lr: float = 5e-4
        weight_decay: float = 0.01
        batch_size: int = 1
        epochs: int = 60
        splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
        wandb_group: str | None = None
        wandb_name: str | None = None
        agent: str | None = None
        debug: bool = False
        base_ch: int = 96
        point_hidden: int = 256
        n_point_blocks: int = 4
        dropout: float = 0.1
        augment: bool = True

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
        grid_size=(64, 32, 48), base_ch=cfg.base_ch,
        point_hidden=cfg.point_hidden, n_point_blocks=cfg.n_point_blocks,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_epochs = 3
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

    model_cfg_dict = {
        "base_ch": cfg.base_ch, "point_hidden": cfg.point_hidden,
        "n_point_blocks": cfg.n_point_blocks, "dropout": cfg.dropout,
    }
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

            if cfg.augment:
                v_in, v_out, pos = augment_batch(v_in, v_out, pos)

            with torch.amp.autocast("cuda"):
                pred = model(v_in, pos, t, idcs)
                loss = (pred - v_out).pow(2).mean()

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
