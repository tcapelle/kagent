"""Train a CFD surrogate model.

FiLM-conditioned ResMLP with Fourier positional features and separate output heads.

Run:
  uv run train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import copy
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from viz import visualize


# --- Model ---

class MultiscaleFourierFeatures(nn.Module):
    """Multi-scale random Fourier features for positional encoding."""

    def __init__(self, in_dim, n_freq=64, scales=(1.0, 5.0, 25.0)):
        super().__init__()
        freqs_per_scale = n_freq // len(scales)
        parts = []
        for s in scales:
            parts.append(torch.randn(in_dim, freqs_per_scale) * s)
        self.register_buffer("B", torch.cat(parts, dim=1))

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FiLMResBlock(nn.Module):
    """ResBlock with FiLM conditioning from global features."""

    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.ones_(self.film.bias[:dim])
        nn.init.zeros_(self.film.bias[dim:])

    def forward(self, x, cond):
        h = self.norm(x)
        gamma, beta = self.film(cond).unsqueeze(1).chunk(2, dim=-1)
        h = gamma * h + beta
        h = self.linear2(torch.nn.functional.gelu(self.linear1(h)))
        return x + h


class CFDModel(nn.Module):
    def __init__(self, in_dim=24, out_dim=3, hidden=512, n_blocks=8,
                 n_fourier=64, cond_dim=128):
        super().__init__()
        # Input splits: local (dims 0-12) and global (dims 13-23)
        self.local_dim = 13
        self.global_dim = in_dim - 13  # 11

        # Multi-scale Fourier on position (2d) and on saf+dsdf (10d)
        self.fourier_pos = MultiscaleFourierFeatures(2, n_fourier, scales=(1.0, 5.0, 25.0))
        self.fourier_geom = MultiscaleFourierFeatures(10, n_fourier, scales=(1.0, 5.0, 25.0))

        # Input projection: local features + fourier features (pos + geom)
        n_fourier_actual = (n_fourier // 3) * 3
        proj_in_dim = self.local_dim + 2 * n_fourier_actual * 2
        self.proj_in = nn.Linear(proj_in_dim, hidden)

        # Global condition encoder
        self.cond_enc = nn.Sequential(
            nn.Linear(self.global_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # FiLM-conditioned ResBlocks
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, cond_dim) for _ in range(n_blocks)])

        # Separate heads for velocity and pressure
        self.head_vel = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.head_p = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, data, **kwargs):
        x = data["x"]  # [B, N, 24]

        # Split local (per-node) and global (per-sample) features
        local = x[:, :, :self.local_dim]
        global_feat = x[:, 0, self.local_dim:]  # same for all nodes

        # Fourier features on position and geometry
        ff_pos = self.fourier_pos(x[:, :, :2])
        ff_geom = self.fourier_geom(x[:, :, 2:12])

        # Combine and project
        h = torch.cat([local, ff_pos, ff_geom], dim=-1)
        h = self.proj_in(h)

        # Condition encoding
        cond = self.cond_enc(global_feat)

        # FiLM blocks
        for block in self.blocks:
            h = block(h, cond)

        # Separate heads
        vel = self.head_vel(h)
        p = self.head_p(h)
        return {"preds": torch.cat([vel, p], dim=-1)}


# --- EMA ---

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = 30.0  # minutes


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 5e-4
    batch_size: int = 2
    surf_weight: float = 50.0
    epochs: int = 30
    resume: str | None = None  # path to checkpoint for warm restart
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

HIDDEN = 768
N_BLOCKS = 6
N_FOURIER = 66  # divisible by 3 for multi-scale
COND_DIM = 192

model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=HIDDEN, n_blocks=N_BLOCKS,
                 n_fourier=N_FOURIER, cond_dim=COND_DIM).to(device)

if cfg.resume:
    state = torch.load(cfg.resume, map_location=device, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    print(f"Resumed from {cfg.resume}")

model = torch.compile(model)
ema = EMA(model, decay=0.999)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
warmup_epochs = 2
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS - warmup_epochs),
    ],
    milestones=[warmup_epochs],
)
scaler = GradScaler()

# --- W&B ---
run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    project=os.environ.get("WANDB_PROJECT", "kagent-v1"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={**asdict(cfg), "n_params": n_params,
            "train_samples": len(train_ds),
            "val_samples": {k: len(v) for k, v in val_splits.items()}},
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump({"n_params": n_params, "hidden": HIDDEN, "n_blocks": N_BLOCKS,
               "n_fourier": N_FOURIER, "cond_dim": COND_DIM}, f)

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
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Normalize inputs and targets
        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        # Mask out nodes with inf/nan targets and sanitize
        finite_mask = y.isfinite().all(dim=-1) & y_norm.isfinite().all(dim=-1)
        mask = mask & finite_mask
        y_norm = y_norm.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        # Forward pass — L1 loss (directly optimizes MAE metric)
        with autocast("cuda"):
            pred = model({"x": x})["preds"]
            abs_err = (pred - y_norm).abs()
            channel_w = torch.tensor([1.0, 1.0, 3.0], device=device)
            abs_err = abs_err * channel_w

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (abs_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (abs_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # --- Validate using EMA model ---
    eval_model = ema.shadow
    eval_model.eval()
    val_loss_sum = 0.0
    n_valid_splits = 0
    split_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = n_vb = 0

        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                finite_mask = y.isfinite().all(dim=-1) & y_norm.isfinite().all(dim=-1)
                mask = mask & finite_mask
                y_norm = y_norm.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

                with autocast("cuda"):
                    pred = eval_model({"x": x})["preds"]
                pred = pred.float()
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                err = err.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vb, 1)
        val_surf /= max(n_vb, 1)
        split_loss = val_vol + cfg.surf_weight * val_surf
        mae_surf /= max(n_surf, 1)
        mae_vol /= max(n_vol, 1)

        split_metrics[split_name] = {
            f"{split_name}/vol_loss": val_vol,
            f"{split_name}/surf_loss": val_surf,
            f"{split_name}/loss": split_loss,
            f"{split_name}/mae_vol_Ux": mae_vol[0].item(),
            f"{split_name}/mae_vol_Uy": mae_vol[1].item(),
            f"{split_name}/mae_vol_p": mae_vol[2].item(),
            f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
            f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
            f"{split_name}/mae_surf_p": mae_surf[2].item(),
        }
        if not math.isnan(split_loss):
            val_loss_sum += split_loss
            n_valid_splits += 1

    mean_val_loss = val_loss_sum / max(n_valid_splits, 1)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    if mean_val_loss < best_val:
        best_val = mean_val_loss
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        # Save EMA weights
        torch.save(eval_model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}]{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}")
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    eval_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    plot_dir = Path("plots") / run.id
    n = 1 if cfg.debug else 4
    for split_name, split_ds in val_splits.items():
        images = visualize(eval_model, split_ds, stats, device, n_samples=n,
                           out_dir=plot_dir / split_name)
        if images:
            wandb.log({
                f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                "global_step": global_step,
            })

wandb.finish()
