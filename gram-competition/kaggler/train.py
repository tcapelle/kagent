"""Train a 3D airflow velocity predictor.

Architecture: Residual MLP with Fourier position encoding, temporal derivatives,
surface indicator, no-slip enforcement, velocity normalization, AMP.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


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


class AirflowPredictor(nn.Module):
    """Residual MLP with physics-informed design and Fourier features."""

    def __init__(self, hidden=512, n_blocks=10, n_fourier=64, dropout=0.05,
                 vel_mean=None, vel_std=None):
        super().__init__()
        self.n_fourier = n_fourier

        # Input features:
        # fourier_pos(n_fourier*2*3=384) + vel_in_norm(15) + vel_derivatives(4*3=12) + time(5) + surface_flag(1) = 417
        pos_feat_dim = n_fourier * 2 * 3  # sin + cos for each of 3 coords
        vel_feat_dim = T_IN * 3  # flattened velocity
        deriv_feat_dim = (T_IN - 1) * 3  # temporal derivatives
        time_feat_dim = T_IN
        surface_dim = 1
        in_dim = pos_feat_dim + vel_feat_dim + deriv_feat_dim + time_feat_dim + surface_dim

        out_dim = T_OUT * 3

        # Random Fourier feature frequencies (fixed, not learned)
        # Multiple scales for multi-resolution spatial encoding
        freqs = torch.randn(3, n_fourier) * 2.0  # scale factor
        self.register_buffer("fourier_freqs", freqs)  # [3, n_fourier]

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

    def _fourier_encode(self, pos):
        """Random Fourier features for position. pos: [B, N, 3]"""
        # pos @ freqs: [B, N, n_fourier] for each coord
        proj = pos @ self.fourier_freqs  # [B, N, n_fourier] (broadcasting over 3 coords summed)
        # Actually we want per-coord Fourier features
        # pos: [B, N, 3], freqs: [3, n_fourier]
        # For each coord, compute sin/cos
        B, N, _ = pos.shape
        feats = []
        for i in range(3):
            p = pos[:, :, i:i+1] * self.fourier_freqs[i:i+1, :]  # [B, N, n_fourier]
            feats.append(torch.sin(p))
            feats.append(torch.cos(p))
        return torch.cat(feats, dim=-1)  # [B, N, n_fourier*6]

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocities
        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std

        # Fourier position encoding
        pos_feat = self._fourier_encode(pos)  # [B, N, n_fourier*6]

        # Flattened velocity features
        v_flat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)  # [B, N, 15]

        # Temporal derivatives: differences between consecutive timesteps
        v_deriv = v_in_norm[:, 1:] - v_in_norm[:, :-1]  # [B, 4, N, 3]
        v_deriv_flat = v_deriv.permute(0, 2, 1, 3).reshape(B, N, (T-1) * C)  # [B, N, 12]

        # Time features
        t_in = t[:, :T_IN]
        t_range = t_in[:, -1:] - t_in[:, :1] + 1e-6
        t_features = (t_in - t_in[:, :1]) / t_range
        t_features = t_features.unsqueeze(1).expand(B, N, T_IN)  # [B, N, 5]

        # Surface indicator
        surface_flag = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                surface_flag[i, idcs_airfoil[i], 0] = 1.0

        # Concatenate all features
        x = torch.cat([pos_feat, v_flat, v_deriv_flat, t_features, surface_flag], dim=-1)

        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x)
        delta_norm = delta_norm.reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)

        # Residual prediction from last input timestep
        last_in_norm = v_in_norm[:, -1:, :, :]
        pred_norm = last_in_norm + delta_norm

        # Denormalize
        pred = pred_norm * self.vel_std + self.vel_mean

        # No-slip enforcement
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                pred[i, :, idcs_airfoil[i], :] = 0.0

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
# Config + data loading
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 50
    subsample_points: int = 25000
    hidden: int = 512
    n_blocks: int = 10
    n_fourier: int = 64
    dropout: float = 0.05
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
    name: DataLoader(ds, batch_size=1, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model = AirflowPredictor(
    hidden=cfg.hidden, n_blocks=cfg.n_blocks, n_fourier=cfg.n_fourier,
    dropout=cfg.dropout,
    vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# Warmup + cosine annealing
warmup_epochs = 3
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, MAX_EPOCHS - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler("cuda")

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

# ---------------------------------------------------------------------------
# W&B setup
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

        B = v_in.shape[0]
        N = v_in.shape[2]

        # Subsample points during training
        if cfg.subsample_points and cfg.subsample_points < N:
            sub_idx = torch.randperm(N, device=device)[:cfg.subsample_points]
            sub_idx = sub_idx.sort().values
            v_in_sub = v_in[:, :, sub_idx, :]
            v_out_sub = v_out[:, :, sub_idx, :]
            pos_sub = pos[:, sub_idx, :]

            sub_set = set(sub_idx.cpu().tolist())
            idx_map = {old: new for new, old in enumerate(sub_idx.cpu().tolist())}
            idcs_sub = []
            for i in range(B):
                if idcs[i] is not None:
                    airfoil_set = set(idcs[i].tolist())
                    kept = airfoil_set & sub_set
                    idcs_sub.append(torch.tensor([idx_map[k] for k in kept], dtype=torch.long))
                else:
                    idcs_sub.append(torch.tensor([], dtype=torch.long))
        else:
            v_in_sub, v_out_sub, pos_sub, idcs_sub = v_in, v_out, pos, idcs

        with torch.amp.autocast("cuda"):
            pred = model(v_in_sub, pos_sub, t, idcs_sub)
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
