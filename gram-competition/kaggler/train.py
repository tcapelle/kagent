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
    def __init__(self, in_dim=3, n_freqs=64):
        super().__init__()
        # Fixed random projection
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * 10.0)

    def forward(self, x):
        # x: [..., 3]
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


class AirflowModel(nn.Module):
    """
    Residual prediction model with:
    - Fourier positional encoding
    - Normalized velocity inputs
    - Predicts delta from last input timestep
    - No-slip boundary enforcement
    """

    def __init__(self, hidden=512, n_blocks=8, n_freqs=64, dropout=0.05):
        super().__init__()
        self.n_freqs = n_freqs

        self.pos_enc = FourierFeatures(in_dim=3, n_freqs=n_freqs)
        pos_dim = 2 * n_freqs  # 128

        # Input: pos_encoding(128) + velocity_in(5*3=15) + time_features(10)
        in_dim = pos_dim + T_IN * 3 + 10
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Fourier encode positions
        pos_feat = self.pos_enc(pos)  # [B, N, 128]

        # Flatten velocity input
        vel_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]

        # Time features - broadcast to all points
        t_feat = t.unsqueeze(1).expand(B, N, 10)  # [B, N, 10]

        # Concatenate all features
        x = torch.cat([pos_feat, vel_flat, t_feat], dim=-1)  # [B, N, 153]

        x = self.proj_in(x)
        x = self.blocks(x)
        delta = self.proj_out(x)  # [B, N, 15]
        delta = delta.reshape(B, T_OUT, N, 3)

        # Residual: add last input timestep
        last_vel = velocity_in[:, -1:, :, :]  # [B, 1, N, 3]
        out = delta + last_vel  # [B, 5, N, 3]

        # No-slip boundary condition: zero velocity on airfoil surface
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                out[i, :, idcs_airfoil[i], :] = 0.0

        return out


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

        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    pred = model(v_in, pos, t, idcs)  # [B, 5, N, 3]

                pred = pred.float()
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
# Point subsampling for training efficiency
# ---------------------------------------------------------------------------

def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points=20000):
    """Subsample points during training. Importance-weight near airfoil."""
    B, T, N, C = v_in.shape
    device = v_in.device

    v_in_sub = []
    v_out_sub = []
    pos_sub = []
    idcs_sub = []

    for i in range(B):
        # Create importance weights: higher near airfoil
        weights = torch.ones(N, device=device)
        airfoil_idx = idcs_airfoil[i]
        if airfoil_idx is not None and len(airfoil_idx) > 0:
            weights[airfoil_idx] = 3.0  # oversample airfoil points

        # Sample indices
        idx = torch.multinomial(weights, n_points, replacement=False)
        idx = idx.sort().values

        v_in_sub.append(v_in[i, :, idx, :])
        v_out_sub.append(v_out[i, :, idx, :])
        pos_sub.append(pos[i, idx, :])

        # Remap airfoil indices
        if airfoil_idx is not None and len(airfoil_idx) > 0:
            # Find which airfoil points are in our subsample
            mask = torch.isin(idx, airfoil_idx.to(device))
            new_idcs = torch.where(mask)[0]
            idcs_sub.append(new_idcs)
        else:
            idcs_sub.append(torch.tensor([], dtype=torch.long, device=device))

    return (
        torch.stack(v_in_sub),
        torch.stack(v_out_sub),
        torch.stack(pos_sub),
        idcs_sub,
    )


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
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    subsample_train: int = 20000  # points to subsample during training
    hidden: int = 512
    n_blocks: int = 8


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

model = AirflowModel(hidden=cfg.hidden, n_blocks=cfg.n_blocks).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
scaler = torch.cuda.amp.GradScaler()

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

        # Subsample points for training efficiency
        if cfg.subsample_train and cfg.subsample_train < N_POINTS:
            v_in_s, v_out_s, pos_s, idcs_s = subsample_batch(
                v_in, v_out, pos, idcs, n_points=cfg.subsample_train
            )
        else:
            v_in_s, v_out_s, pos_s, idcs_s = v_in, v_out, pos, idcs

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            pred = model(v_in_s, pos_s, t, idcs_s)  # [B, 5, N_sub, 3]
            loss = (pred - v_out_s).pow(2).mean()

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
