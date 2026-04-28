"""Train Transolver on TandemFoilSet — askeladd's recipe.

Based on thorfinn's apr23 iter-4 recipe: bigger Transolver (256x6, slice_num=96,
mlp_ratio=4), bf16 AMP, point subsampling (keep all surface + 20k volume),
warmup+cosine, channel-weighted surface loss with extra weight on pressure
(the scored metric).

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
"""

import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from model import Transolver
from viz import visualize


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-5
    batch_size: int = 4
    surf_weight: float = 20.0
    surf_p_weight: float = 48.0  # extra multiplier on surface pressure (primary metric)
    epochs: int = 50
    train_subsample: int = 60000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    resume_from: str | None = None
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

# Resume mode: when continuing from a checkpoint, default to a lower LR and a
# tiny warmup (the model is already trained — no need to ramp from zero).
# 5e-5 is appropriate for repeat-resume fine-tuning; first-resume can override.
if cfg.resume_from is not None:
    if cfg.lr == 5e-4:
        cfg.lr = 5e-5
    if cfg.warmup_steps == 1000:
        cfg.warmup_steps = 100
    print(f"Resume mode: lr={cfg.lr}, warmup={cfg.warmup_steps}")

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

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,
    n_layers=6,
    n_head=8,
    slice_num=96,
    mlp_ratio=4,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params")

if cfg.resume_from:
    state = torch.load(cfg.resume_from, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Resumed from {cfg.resume_from}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

steps_per_epoch = max(1, len(train_loader))
total_steps = steps_per_epoch * MAX_EPOCHS

def lr_lambda(step):
    if step < cfg.warmup_steps:
        return (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(progress * math.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32
print(f"AMP: {use_amp} ({amp_dtype})")

run = wandb.init(
    entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
    project=os.environ.get("WANDB_PROJECT", "kagent-v2"),
    group=cfg.wandb_group,
    name=cfg.wandb_name,
    tags=[cfg.agent] if cfg.agent else [],
    config={
        **asdict(cfg),
        "model_config": model_config,
        "n_params": n_params,
        "train_samples": len(train_ds),
        "val_samples": {k: len(v) for k, v in val_splits.items()},
    },
    mode=os.environ.get("WANDB_MODE", "online"),
)

wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for _name in VAL_SPLIT_NAMES:
    wandb.define_metric(f"{_name}/*", step_metric="global_step")
wandb.define_metric("lr", step_metric="global_step")

model_dir = Path(f"models/model-{run.id}")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)


def subsample_batch(x, y, is_surface, mask, target_n):
    """Keep all surface points, randomly sample volume points to ~target_n total."""
    B = x.shape[0]
    new_x, new_y, new_surf = [], [], []
    max_n = 0

    for b in range(B):
        m = mask[b]
        s = is_surface[b] & m
        v = (~is_surface[b]) & m
        n_surf = int(s.sum())
        n_vol = int(v.sum())
        n_vol_keep = min(n_vol, max(0, target_n - n_surf))

        surf_idx = torch.nonzero(s, as_tuple=False).squeeze(-1)
        vol_idx = torch.nonzero(v, as_tuple=False).squeeze(-1)
        perm = torch.randperm(n_vol, device=x.device)[:n_vol_keep]
        vol_keep = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_keep], dim=0)

        new_x.append(x[b, keep])
        new_y.append(y[b, keep])
        new_surf.append(is_surface[b, keep])
        max_n = max(max_n, keep.shape[0])

    X = torch.zeros(B, max_n, x.shape[-1], dtype=x.dtype, device=x.device)
    Y = torch.zeros(B, max_n, y.shape[-1], dtype=y.dtype, device=y.device)
    S = torch.zeros(B, max_n, dtype=torch.bool, device=x.device)
    M = torch.zeros(B, max_n, dtype=torch.bool, device=x.device)
    for b in range(B):
        n = new_x[b].shape[0]
        X[b, :n] = new_x[b]
        Y[b, :n] = new_y[b]
        S[b, :n] = new_surf[b]
        M[b, :n] = True
    return X, Y, S, M


best_val = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()
ch_w = torch.tensor([1.0, 1.0, cfg.surf_p_weight], device=device)

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT:
        print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if cfg.train_subsample > 0:
            x, y, is_surface, mask = subsample_batch(x, y, is_surface, mask, cfg.train_subsample)

        x_n = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            pred = model({"x": x_n})["preds"]
            sq_err = (pred.float() - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1) / 3.0
            surf_loss = (sq_err * surf_mask.unsqueeze(-1) * ch_w).sum() / (surf_mask.sum().clamp(min=1) * ch_w.sum())
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % 20 == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/vol_loss_step": vol_loss.item(),
                "train/surf_loss_step": surf_loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "global_step": global_step,
            })

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate (full-mesh) ---
    model.eval()
    val_loss_sum = 0.0
    split_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = 0
        n_vb = 0

        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x_n = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred = model({"x": x_n})["preds"].float()
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
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
        val_loss_sum += split_loss

    mean_val_loss = val_loss_sum / len(val_loaders)
    avg_surf_p = sum(sm[f"{n}/mae_surf_p"] for n, sm in zip(VAL_SPLIT_NAMES, split_metrics.values())) / len(VAL_SPLIT_NAMES)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "val/avg_surf_p": avg_surf_p,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    if avg_surf_p < best_val:
        best_val = avg_surf_p
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_surf_p": avg_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}_p={split_metrics[name][f'{name}/mae_surf_p']:.2f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"avg_surf_p={avg_surf_p:.2f}  {split_summary}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, avg_surf_p={best_metrics['avg_surf_p']:.4f}")
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    plot_dir = Path("plots") / run.id
    n = 1 if cfg.debug else 2
    for split_name, split_ds in val_splits.items():
        images = visualize(model, split_ds, stats, device, n_samples=n,
                           out_dir=plot_dir / split_name)
        if images:
            wandb.log({
                f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                "global_step": global_step,
            })

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
        print(f"predict.py failed:\n{result.stderr[-1000:]}")

wandb.finish()
