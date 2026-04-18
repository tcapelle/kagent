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
from model import BaselineMLP, VoxelFlowNet


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
            for v_in, v_out, pos, t, idcs, sdf in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                sdf = sdf.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs, sdf)  # [B, 5, N, 3]

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
    lr: float = 2.5e-3
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 28
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

model = VoxelFlowNet(
    vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
    grid_res=80, grid_ch=32, n_grid_blocks=4,
    point_hidden=384, n_point_blocks=6, point_dropout=0.15,
).to(device)
vel_std_gpu = stats["vel_std"].to(device).view(1, 1, 1, 3)

n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)


class EMA:
    """Exponential moving average of model weights. Validate/save the EMA copy."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def swap_in(self, model):
        self.backup = {k: model.state_dict()[k].detach().clone() for k in self.shadow}
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def swap_out(self, model):
        sd = model.state_dict()
        for k in self.shadow:
            sd[k].copy_(self.backup[k])


ema = EMA(model, decay=0.999)

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

    for v_in, v_out, pos, t, idcs, sdf in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        v_in = v_in.to(device, non_blocking=True)
        v_out = v_out.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        sdf = sdf.to(device, non_blocking=True)

        pred = model(v_in, pos, t, idcs, sdf)  # [B, 5, N, 3]
        # L2 velocity loss, matches the val metric exactly.
        loss = (pred - v_out).norm(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    epoch_loss /= n_batches

    # --- Validate EMA weights (what we'd ship) ---
    ema.swap_in(model)
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
        # model is currently holding EMA weights — save those (they're what won val).
        torch.save(model.state_dict(), model_path)
        shutil.copyfile(model_path, git_ckpt_path)
        tag = " *"

    # Restore training weights for the next epoch.
    ema.swap_out(model)

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
