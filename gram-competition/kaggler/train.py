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
# Fourier Neural Operator layer
# ---------------------------------------------------------------------------

class SpectralConv3d(nn.Module):
    """3D spectral convolution: learn weights in Fourier space."""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, 2))
            for _ in range(4)  # 4 corners of the Fourier modes
        ])

    def complex_mul(self, a, b):
        # a: [..., 2], b: [..., 2] → [..., 2]
        return torch.stack([
            a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
            a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
        ], dim=-1)

    def forward(self, x):
        # x: [B, C, X, Y, Z]
        B = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])  # [B, C, X, Y, Z//2+1] complex
        x_ft = torch.view_as_real(x_ft)  # [B, C, X, Y, Z//2+1, 2]

        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft = torch.zeros(B, self.out_channels, x.size(2), x.size(3), x.size(4) // 2 + 1, 2,
                            device=x.device, dtype=x.dtype)

        # 4 corners of the mode space
        out_ft[:, :, :m1, :m2, :m3] = torch.einsum(
            "bcxyz,coxyz->boxyz", x_ft[:, :, :m1, :m2, :m3], self.weights[0])
        out_ft[:, :, -m1:, :m2, :m3] = torch.einsum(
            "bcxyz,coxyz->boxyz", x_ft[:, :, -m1:, :m2, :m3], self.weights[1])
        out_ft[:, :, :m1, -m2:, :m3] = torch.einsum(
            "bcxyz,coxyz->boxyz", x_ft[:, :, :m1, -m2:, :m3], self.weights[2])
        out_ft[:, :, -m1:, -m2:, :m3] = torch.einsum(
            "bcxyz,coxyz->boxyz", x_ft[:, :, -m1:, -m2:, :m3], self.weights[3])

        out_ft = torch.view_as_complex(out_ft.contiguous())
        return torch.fft.irfftn(out_ft, s=[x.size(2), x.size(3), x.size(4)])


class FNOBlock(nn.Module):
    """FNO block: spectral conv + local conv + residual."""
    def __init__(self, channels, modes1, modes2, modes3):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, modes1, modes2, modes3)
        self.local_conv = nn.Conv3d(channels, channels, 1)
        self.norm = nn.InstanceNorm3d(channels)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.spectral(x) + self.local_conv(x)))


# ---------------------------------------------------------------------------
# Model: MLP per point + FNO on voxel grid
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
    Hybrid MLP + FNO model:
    1. Per-point MLP encodes features
    2. Voxelize to 3D grid
    3. FNO learns spatial interactions in Fourier space
    4. Interpolate back to points
    5. Combine with per-point features for prediction
    """

    def __init__(self, hidden=256, n_mlp_blocks=6, fno_channels=32,
                 grid_size=(48, 12, 24), n_fno_blocks=4, fno_modes=(12, 4, 8),
                 vel_mean=None, vel_std=None):
        super().__init__()
        self.grid_size = grid_size

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

        self.register_buffer("pos_min", torch.tensor([0.0, -0.41, 0.0]))
        self.register_buffer("pos_max", torch.tensor([2.1, 0.41, 1.22]))

        # Per-point input: pos(3) + vel_norm(15) + vel_dev(15) = 33
        in_dim = 3 + T_IN * 3 + T_IN * 3
        out_dim = T_OUT * 3

        # MLP feature encoder
        self.point_encoder = nn.Linear(in_dim, hidden)
        self.point_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_mlp_blocks // 2)])

        # Point-to-voxel projection
        self.to_voxel = nn.Linear(hidden, fno_channels)

        # FNO on voxel grid
        self.fno_blocks = nn.Sequential(
            *[FNOBlock(fno_channels, *fno_modes) for _ in range(n_fno_blocks)]
        )

        # Voxel-to-point interpolation already gives fno_channels features
        # Combine per-point features + FNO features
        self.combiner = nn.Linear(hidden + fno_channels, hidden)
        self.post_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_mlp_blocks - n_mlp_blocks // 2)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        nn.init.zeros_(self.proj_out[-1].weight)
        nn.init.zeros_(self.proj_out[-1].bias)

    def voxelize_and_fno(self, features, pos):
        """Voxelize features, apply FNO, interpolate back."""
        B, N, D = features.shape
        Gx, Gy, Gz = self.grid_size
        device = features.device

        # Normalize positions to [0, 1]
        pos_range = (self.pos_max - self.pos_min).clamp(min=1e-6)
        pos_norm = (pos - self.pos_min.view(1, 1, 3)) / pos_range.view(1, 1, 3)
        pos_norm = pos_norm.clamp(0, 1 - 1e-6)

        # Voxel indices
        vx = (pos_norm[:, :, 0] * Gx).long().clamp(0, Gx - 1)
        vy = (pos_norm[:, :, 1] * Gy).long().clamp(0, Gy - 1)
        vz = (pos_norm[:, :, 2] * Gz).long().clamp(0, Gz - 1)
        voxel_idx = vx * (Gy * Gz) + vy * Gz + vz

        # Scatter into grid
        grid_flat = torch.zeros(B, Gx * Gy * Gz, D, device=device, dtype=features.dtype)
        count = torch.zeros(B, Gx * Gy * Gz, 1, device=device, dtype=features.dtype)
        idx_exp = voxel_idx.unsqueeze(-1).expand(B, N, D)
        grid_flat.scatter_add_(1, idx_exp, features)
        count.scatter_add_(1, voxel_idx.unsqueeze(-1), torch.ones(B, N, 1, device=device, dtype=features.dtype))
        grid_flat = grid_flat / count.clamp(min=1)

        # Reshape to 3D: [B, D, Gx, Gy, Gz]
        grid_3d = grid_flat.permute(0, 2, 1).reshape(B, D, Gx, Gy, Gz)

        # Apply FNO
        grid_3d = self.fno_blocks(grid_3d)

        # Interpolate back to points
        grid_coords = pos_norm * 2 - 1
        grid_coords = grid_coords.view(B, 1, 1, N, 3)
        spatial_feats = F.grid_sample(
            grid_3d, grid_coords, mode='bilinear', padding_mode='border', align_corners=True,
        ).squeeze(2).squeeze(2).permute(0, 2, 1)

        return spatial_feats

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        vel_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)
        vel_mean_t = vel_norm.mean(dim=1, keepdim=True)
        vel_dev = vel_norm - vel_mean_t

        vel_flat = vel_norm.reshape(B, N, T * C)
        dev_flat = vel_dev.reshape(B, N, T * C)

        point_input = torch.cat([pos, vel_flat, dev_flat], dim=-1)

        # Per-point encoding
        x = self.point_encoder(point_input)
        x = self.point_blocks(x)

        # FNO spatial processing
        voxel_feats = self.to_voxel(x)
        spatial_feats = self.voxelize_and_fno(voxel_feats, pos)

        # Combine
        x = self.combiner(torch.cat([x, spatial_feats], dim=-1))
        x = self.post_blocks(x)
        delta_norm = self.proj_out(x).reshape(B, T_OUT, N, 3)

        out_norm = delta_norm + vel_mean_t
        out = out_norm * (self.vel_std + 1e-8) + self.vel_mean

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
# Config + main
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 200
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    hidden: int = 256
    n_mlp_blocks: int = 6
    fno_channels: int = 32
    n_fno_blocks: int = 4


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
        hidden=cfg.hidden, n_mlp_blocks=cfg.n_mlp_blocks,
        fno_channels=cfg.fno_channels, n_fno_blocks=cfg.n_fno_blocks,
        vel_mean=vel_mean, vel_std=vel_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS,
        pct_start=0.05, anneal_strategy='cos',
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

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred = model(v_in, pos, t, idcs)
                loss = (pred - v_out).pow(2).mean()

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
