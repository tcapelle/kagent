"""Train a 3D airflow velocity predictor.

Template — fill in your model architecture.
The training loop, loss, validation, and W&B logging are provided.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import shutil
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
# Baseline MLP — replace with your own architecture
#
# Model contract:
#   Input:  velocity_in [B, 5, N, 3], pos [B, N, 3], t [B, 10], idcs_airfoil list[tensor]
#   Output: velocity_out [B, 5, N, 3]  (predicted future velocity field)
#
# Note: the real competition uses model(t, pos, idcs_airfoil, velocity_in) —
#       different arg order. If you submit to the real comp, wrap accordingly.
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim, film_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim * 2)
        self.lin2 = nn.Linear(dim * 2, dim)
        self.film = nn.Linear(film_dim, dim * 2) if film_dim else None

    def forward(self, x, cond=None):
        h = self.norm(x)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).chunk(2, dim=-1)
            h = h * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)
        h = self.lin1(h)
        h = torch.nn.functional.gelu(h)
        h = self.lin2(h)
        return x + h


def sinusoidal_embed(x, dim):
    """Sinusoidal embedding for a scalar per-sample tensor x of shape [B, K]."""
    device = x.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=device, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = x.unsqueeze(-1) * freqs  # [B, K, half]
    emb = torch.cat([args.sin(), args.cos()], dim=-1)  # [B, K, dim]
    return emb


class ResMLP(nn.Module):
    """Pointwise residual MLP with physics priors.

    - Normalizes velocity (standardize by train stats) and position (per-sample center + scale).
    - Predicts residual delta per output timestep relative to the last input timestep.
    - FiLM-conditions each block on a per-sample time embedding (encodes the 10 absolute t values).
    - Enforces no-slip BC: zero velocity at airfoil indices.
    """

    def __init__(self, hidden=384, n_blocks=8, time_dim=64, vel_mean=None, vel_std=None):
        super().__init__()
        in_dim = 3 + T_IN * 3 + 1   # pos(3) + velocity_in (normalized, 5*3=15) + airfoil mask(1)
        out_dim = T_OUT * 3          # residual delta per-timestep
        self.proj_in = nn.Linear(in_dim, hidden)
        # Per-sample time embedding dim: 10 absolute times * time_dim, projected to `hidden`.
        t_enc_dim = 10 * time_dim
        self.time_proj = nn.Sequential(
            nn.Linear(t_enc_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.blocks = nn.ModuleList([ResBlock(hidden, film_dim=hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
        self.time_dim = time_dim

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocity with dataset stats.
        v_norm = (velocity_in - self.vel_mean) / self.vel_std  # [B, T, N, 3]

        # Center and scale pos per sample (makes model invariant to absolute coordinate).
        pos_mean = pos.mean(dim=1, keepdim=True)
        pos_scale = pos.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        pos_norm = (pos - pos_mean) / pos_scale  # [B, N, 3]

        # Airfoil mask as a per-point feature.
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            airfoil_mask[b, idcs.to(pos.device), 0] = 1.0

        x = torch.cat([pos_norm, v_norm.reshape(B, N, T * C), airfoil_mask], dim=-1)
        x = self.proj_in(x)

        # Time conditioning: embed each of the 10 timestep values, flatten, project.
        t_emb = sinusoidal_embed(t, self.time_dim).reshape(B, -1)  # [B, 10*time_dim]
        cond = self.time_proj(t_emb)  # [B, hidden]

        for blk in self.blocks:
            x = blk(x, cond)

        out = self.proj_out(x)  # [B, N, T_OUT * 3]
        delta_norm = out.reshape(B, T_OUT, N, 3)

        # Residual prediction: add delta to last input timestep (both in normalized space).
        v_last_norm = v_norm[:, -1:, :, :]  # [B, 1, N, 3]
        pred_norm = v_last_norm + delta_norm  # [B, T_OUT, N, 3]

        # Denormalize back to m/s.
        pred = pred_norm * self.vel_std + self.vel_mean

        # No-slip BC: zero velocity at airfoil surface points.
        for b, idcs in enumerate(idcs_airfoil):
            pred[b, :, idcs.to(pos.device), :] = 0.0
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

        with torch.no_grad():
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
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 50
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

model = ResMLP(
    hidden=384,
    n_blocks=8,
    time_dim=64,
    vel_mean=stats["vel_mean"],
    vel_std=stats["vel_std"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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

KAGGLER_NAME = os.environ.get("KAGGLER_NAME", cfg.agent or "local")
pvc_dir = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}/{KAGGLER_NAME}/checkpoints/model-{run.id}")
pvc_dir.mkdir(parents=True, exist_ok=True)
model_path = pvc_dir / "checkpoint.pt"

# Git-tracked mirror of the best checkpoint (un-ignored in .gitignore).
git_ckpt_path = Path("checkpoints/best.pt")
git_ckpt_path.parent.mkdir(parents=True, exist_ok=True)


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

        pred = model(v_in, pos, t, idcs)  # [B, 5, N, 3]
        loss = (pred - v_out).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
        shutil.copyfile(model_path, git_ckpt_path)
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
