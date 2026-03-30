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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data
from model import ResidualMLP


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

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs)

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
# Point subsampling
# ---------------------------------------------------------------------------

def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points):
    B, T, N, C = v_in.shape
    if n_points >= N:
        return v_in, v_out, pos, idcs_airfoil

    new_v_in = []
    new_v_out = []
    new_pos = []
    new_idcs = []

    for i in range(B):
        airfoil = idcs_airfoil[i].to(v_in.device)
        n_airfoil = len(airfoil)

        # Cap airfoil to half budget
        max_airfoil = n_points // 2
        if n_airfoil > max_airfoil:
            perm_a = torch.randperm(n_airfoil, device=v_in.device)[:max_airfoil]
            airfoil_sel = airfoil[perm_a]
            n_airfoil = max_airfoil
        else:
            airfoil_sel = airfoil

        all_idx = torch.arange(N, device=v_in.device)
        mask = torch.ones(N, dtype=torch.bool, device=v_in.device)
        mask[idcs_airfoil[i].to(v_in.device)] = False
        non_airfoil = all_idx[mask]

        n_random = min(n_points - n_airfoil, len(non_airfoil))
        perm = torch.randperm(len(non_airfoil), device=v_in.device)[:n_random]
        selected = torch.cat([airfoil_sel, non_airfoil[perm]])

        new_v_in.append(v_in[i, :, selected, :])
        new_v_out.append(v_out[i, :, selected, :])
        new_pos.append(pos[i, selected, :])
        new_idcs.append(torch.arange(n_airfoil, device=v_in.device))

    return torch.stack(new_v_in), torch.stack(new_v_out), torch.stack(new_pos), new_idcs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 100
    subsample_train: int = 50000
    hidden: int = 640
    n_blocks: int = 12
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

loader_kwargs = dict(collate_fn=collate_fn, num_workers=4, pin_memory=True)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model = ResidualMLP(
    hidden=cfg.hidden, n_blocks=cfg.n_blocks, n_freqs=128,
    vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

warmup_epochs = 3
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(MAX_EPOCHS - warmup_epochs, 1)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
# Training loop — metric-aligned loss
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

        v_in_sub, v_out_sub, pos_sub, idcs_sub = subsample_batch(
            v_in, v_out, pos, idcs, cfg.subsample_train
        )

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
