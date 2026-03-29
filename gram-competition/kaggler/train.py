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
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model: Enhanced Residual MLP with time embedding and no-slip enforcement
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


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        # t: [B, 10]
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device).float() * (torch.log(torch.tensor(10000.0)) / half))
        # Use mean of input times and output times
        t_in_mean = t[:, :5].mean(dim=1, keepdim=True)  # [B, 1]
        t_out_mean = t[:, 5:].mean(dim=1, keepdim=True)  # [B, 1]
        dt = t_out_mean - t_in_mean  # [B, 1]
        args = dt * freqs[None, :]  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        return self.mlp(emb)


class FrierenModel(nn.Module):
    """Residual prediction MLP with normalization, time embedding, and no-slip BC."""

    def __init__(self, hidden=512, n_blocks=8, dropout=0.05, n_fourier=32):
        super().__init__()
        self.n_fourier = n_fourier
        # Input: pos(3) + fourier_pos(n_fourier*6) + velocity_in(5*3=15) + velocity_stats(3: mean of last step)
        pos_dim = 3 + n_fourier * 6  # raw pos + fourier features
        vel_dim = T_IN * 3  # 15
        in_dim = pos_dim + vel_dim

        self.proj_in = nn.Linear(in_dim, hidden)
        self.time_emb = TimeEmbedding(hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, T_OUT * 3))

        # Learnable Fourier frequency matrix for positional encoding
        self.register_buffer('fourier_freqs', torch.randn(3, n_fourier) * 10.0)

    def _fourier_pos(self, pos):
        # pos: [B, N, 3]
        proj = pos @ self.fourier_freqs  # [B, N, n_fourier]
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # [B, N, n_fourier*2]

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Fourier position encoding
        pos_feat = torch.cat([pos, self._fourier_pos(pos)], dim=-1)  # [B, N, 3+n_fourier*2]

        # Flatten velocity input
        vel_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]

        # Concatenate all features
        x = torch.cat([pos_feat, vel_flat], dim=-1)  # [B, N, in_dim]
        x = self.proj_in(x)  # [B, N, hidden]

        # Add time embedding (broadcast to all points)
        t_emb = self.time_emb(t)  # [B, hidden]
        x = x + t_emb.unsqueeze(1)  # [B, N, hidden]

        x = self.blocks(x)
        delta = self.proj_out(x)  # [B, N, T_OUT*3]
        delta = delta.reshape(B, T_OUT, N, 3)

        # Residual prediction: add last input timestep
        last_vel = velocity_in[:, -1:, :, :]  # [B, 1, N, 3]
        out = last_vel + delta  # [B, T_OUT, N, 3]

        # No-slip boundary condition: zero velocity at airfoil surface
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
    subsample_train: int = 20000  # subsample points during training
    grad_accum: int = 1


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

model = FrierenModel(hidden=512, n_blocks=8, dropout=0.05, n_fourier=32).to(device)

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
# Training loop with AMP and point subsampling
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

    for batch_idx, (v_in, v_out, pos, t, idcs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)):
        v_in = v_in.to(device, non_blocking=True)
        v_out = v_out.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)

        B = v_in.shape[0]

        # Subsample points during training for speed/memory
        if cfg.subsample_train > 0 and cfg.subsample_train < N_POINTS:
            idx = torch.randperm(N_POINTS, device=device)[:cfg.subsample_train]
            v_in_sub = v_in[:, :, idx, :]
            v_out_sub = v_out[:, :, idx, :]
            pos_sub = pos[:, idx, :]
            # Remap airfoil indices
            idx_set = set(idx.cpu().tolist())
            new_idcs = []
            for i in range(B):
                if idcs[i] is not None:
                    old_idcs = idcs[i].tolist()
                    # Find which airfoil indices are in our subsample
                    idx_list = idx.cpu().tolist()
                    idx_to_new = {old: new for new, old in enumerate(idx_list)}
                    mapped = [idx_to_new[o] for o in old_idcs if o in idx_to_new]
                    new_idcs.append(torch.tensor(mapped, dtype=torch.long))
                else:
                    new_idcs.append(torch.tensor([], dtype=torch.long))
        else:
            v_in_sub, v_out_sub, pos_sub, new_idcs = v_in, v_out, pos, idcs

        with torch.cuda.amp.autocast():
            pred = model(v_in_sub, pos_sub, t, new_idcs)
            loss = (pred - v_out_sub).pow(2).mean()
            loss = loss / cfg.grad_accum

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        global_step += 1
        wandb.log({"train/loss": loss.item() * cfg.grad_accum, "global_step": global_step})

        epoch_loss += loss.item() * cfg.grad_accum
        n_batches += 1

    # Handle remaining gradients
    if n_batches % cfg.grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

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
