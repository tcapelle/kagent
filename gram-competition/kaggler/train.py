"""Train a 3D airflow velocity predictor.

Template — fill in your model architecture.
The training loop, loss, validation, and W&B logging are provided.

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
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class FourierFeatures(nn.Module):
    """Random Fourier features for positional encoding."""
    def __init__(self, in_dim, n_freqs=64):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * 2.0)

    def forward(self, x):
        proj = x @ self.B  # [..., n_freqs]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [..., 2*n_freqs]


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """
    Improved MLP with:
    - Fourier positional features
    - Time conditioning
    - Residual prediction (predict delta from last input timestep)
    - No-slip boundary enforcement
    - Input normalization
    """

    def __init__(self, hidden=512, n_blocks=8, n_freqs=64, dropout=0.05):
        super().__init__()
        self.n_freqs = n_freqs

        # Fourier features for position
        self.pos_ff = FourierFeatures(3, n_freqs)
        pos_dim = 2 * n_freqs  # 128

        # Time embedding
        self.time_ff = FourierFeatures(1, 32)
        time_dim = 2 * 32  # 64
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim * 10, 128),  # 10 timesteps
            nn.GELU(),
            nn.Linear(128, hidden),
        )

        # Input: pos_fourier(128) + pos_raw(3) + velocity_in(5*3=15) + velocity_stats(6: mean+std of last timestep)
        in_dim = pos_dim + 3 + T_IN * 3 + 6
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Fourier position features
        pos_feat = self.pos_ff(pos)  # [B, N, 128]

        # Velocity features: flatten all input timesteps
        vel_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]

        # Local velocity statistics from last timestep (per-point context)
        last_vel = velocity_in[:, -1]  # [B, N, 3]
        vel_mean = velocity_in.mean(dim=1)  # [B, N, 3] mean over time
        vel_stats = torch.cat([last_vel, vel_mean], dim=-1)  # [B, N, 6]

        # Combine inputs
        x = torch.cat([pos, pos_feat, vel_flat, vel_stats], dim=-1)  # [B, N, in_dim]
        x = self.proj_in(x)  # [B, N, hidden]

        # Time conditioning (global, broadcast to all points)
        t_feat = self.time_ff(t.unsqueeze(-1))  # [B, 10, 64]
        t_feat = t_feat.reshape(B, -1)  # [B, 640]
        t_cond = self.time_mlp(t_feat)  # [B, hidden]
        x = x + t_cond.unsqueeze(1)  # broadcast to [B, N, hidden]

        x = self.blocks(x)
        delta = self.proj_out(x)  # [B, N, T_OUT*3]
        delta = delta.reshape(B, T_OUT, N, 3)

        # Residual prediction: add last input timestep
        pred = last_vel.unsqueeze(1) + delta  # [B, 5, N, 3]

        # No-slip boundary condition: zero velocity at airfoil surface
        for i, idcs in enumerate(idcs_airfoil):
            pred[i, :, idcs, :] = 0.0

        return pred


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
    """Run validation, log to W&B. Returns mean val metric (L2 velocity error)."""
    model.eval()
    val_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        total_l2 = 0.0
        total_mae = torch.zeros(3, device=device, dtype=torch.float64)
        n_samples = 0

        with torch.no_grad(), torch.cuda.amp.autocast():
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs)  # [B, 5, N, 3]

                # L2 velocity error (competition hint metric)
                l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))  # [B]
                total_l2 += l2_err.sum().item()

                # Per-component MAE
                mae = (pred - v_out).abs().mean(dim=(1, 2))  # [B, 3]
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
# Config + data loading
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))  # minutes


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 50
    subsample_train: int = 25000  # subsample points during training
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


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

model = ResidualMLP(hidden=512, n_blocks=8, n_freqs=64, dropout=0.05).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
scaler = torch.amp.GradScaler("cuda")

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

# ---------------------------------------------------------------------------
# W&B setup (do not remove)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Point subsampling helper
# ---------------------------------------------------------------------------

def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points):
    """Subsample points during training, preserving airfoil indices."""
    B, T, N, C = v_in.shape
    if n_points >= N:
        return v_in, v_out, pos, idcs_airfoil

    new_v_in = []
    new_v_out = []
    new_pos = []
    new_idcs = []

    for i in range(B):
        # Ensure airfoil points are included
        airfoil = idcs_airfoil[i]
        n_airfoil = len(airfoil)

        # Random sample from non-airfoil points
        all_idx = torch.arange(N, device=v_in.device)
        mask = torch.ones(N, dtype=torch.bool, device=v_in.device)
        mask[airfoil] = False
        non_airfoil = all_idx[mask]

        n_random = min(n_points - n_airfoil, len(non_airfoil))
        if n_random > 0:
            perm = torch.randperm(len(non_airfoil), device=v_in.device)[:n_random]
            selected_non_airfoil = non_airfoil[perm]
        else:
            selected_non_airfoil = non_airfoil[:0]

        # Combine: airfoil first, then random
        selected = torch.cat([airfoil.to(v_in.device), selected_non_airfoil])

        new_v_in.append(v_in[i, :, selected, :])
        new_v_out.append(v_out[i, :, selected, :])
        new_pos.append(pos[i, selected, :])
        # Airfoil indices are now 0..n_airfoil-1
        new_idcs.append(torch.arange(n_airfoil, device=v_in.device))

    return torch.stack(new_v_in), torch.stack(new_v_out), torch.stack(new_pos), new_idcs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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

        # Subsample points during training
        v_in_sub, v_out_sub, pos_sub, idcs_sub = subsample_batch(
            v_in, v_out, pos, idcs, cfg.subsample_train
        )

        with torch.cuda.amp.autocast():
            pred = model(v_in_sub, pos_sub, t, idcs_sub)  # [B, 5, n_sub, 3]
            loss = (pred - v_out_sub).pow(2).mean()

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

    # --- Validate ---
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

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, val/l2_error={best_metrics['val_l2_error']:.4f}")
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

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
