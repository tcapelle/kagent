"""Train Transolver on TandemFoilSet.

Edward's iter1 — proven frieren recipe (L1 loss, p_weight=3, train_subsample=40k,
warmup+cosine, 35 epochs) plus Fourier features + attention masking.
"""

import math
import os
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from einops import rearrange
from timm.layers import trunc_normal_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from model import Transolver
from viz import visualize


# ---------------------------------------------------------------------------
# Subsample volume nodes (keep all surface nodes)
# ---------------------------------------------------------------------------


class SubsampledDataset(Dataset):
    """Wrap a dataset and subsample volume nodes per __getitem__.

    All surface nodes are always kept; the requested number of volume nodes
    is sampled uniformly without replacement. Subsample disabled when n_vol_keep<=0.
    """

    def __init__(self, base, n_vol_keep: int):
        self.base = base
        self.n_vol_keep = n_vol_keep

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, is_surface = self.base[idx]
        if self.n_vol_keep <= 0:
            return x, y, is_surface
        vol_idx = (~is_surface).nonzero(as_tuple=True)[0]
        if vol_idx.numel() <= self.n_vol_keep:
            return x, y, is_surface
        keep = torch.randperm(vol_idx.numel())[: self.n_vol_keep]
        keep_vol = vol_idx[keep]
        surf_idx = is_surface.nonzero(as_tuple=True)[0]
        order = torch.cat([surf_idx, keep_vol])
        # Shuffle so order isn't surface-then-vol
        order = order[torch.randperm(order.numel())]
        return x[order], y[order], is_surface[order]



# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    p_weight: float = 3.0   # extra weighting on pressure channel
    epochs: int = 35
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    train_subsample: int = 40000  # subsample volume nodes (keeps all surface). 0 = full mesh
    loss_type: str = "l1"   # l1 or l2
    warm_start: str | None = None
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    slice_num: int = 64
    mlp_ratio: int = 2
    fourier_scales: int = 8
    fourier_max_freq: float = 16.0
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds_raw, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}
train_ds = SubsampledDataset(train_ds_raw, cfg.train_subsample) if cfg.train_subsample > 0 else train_ds_raw

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds_raw), replacement=True)
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
    n_hidden=cfg.n_hidden,
    n_layers=cfg.n_layers,
    n_head=cfg.n_head,
    slice_num=cfg.slice_num,
    mlp_ratio=cfg.mlp_ratio,
    fourier_scales=cfg.fourier_scales,
    fourier_max_freq=cfg.fourier_max_freq,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e6:.2f}M")

if cfg.warm_start:
    state = torch.load(cfg.warm_start, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Warm-started from {cfg.warm_start} (missing={len(missing)} unexpected={len(unexpected)})")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
# Warmup + cosine
def lr_lambda(epoch):
    if epoch < cfg.warmup_epochs:
        return float(epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, MAX_EPOCHS - cfg.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
        "train_samples": len(train_ds_raw),
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
model_dir.mkdir(parents=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

research_tag = os.environ.get("RESEARCH_TAG", "default")
kaggler_name = os.environ.get("KAGGLER_NAME") or cfg.agent or "unknown"
pvc_dir = Path(f"/mnt/new-pvc/kagent/{research_tag}/{kaggler_name}/checkpoints/model-{run.id}")
pvc_dir.mkdir(parents=True, exist_ok=True)
pvc_path = pvc_dir / "checkpoint.pt"
with open(pvc_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

# Channel weighting tensor: [1, 1, 1, 3] for [Ux, Uy, p] -> upweight p
chan_w = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)


def channel_loss(pred, y_norm, mask_pts):
    diff = pred - y_norm
    if cfg.loss_type == "l1":
        err = diff.abs()
    else:
        err = diff ** 2
    err = err * chan_w  # [..., 3]
    masked = err * mask_pts.unsqueeze(-1)
    return masked.sum() / mask_pts.sum().clamp(min=1) / chan_w.sum()


best_val = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()
amp_dtype = torch.bfloat16

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT - 1.5:
        print(f"Approaching timeout. Stopping at epoch {epoch}.")
        break

    t0 = time.time()
    model.train()
    epoch_vol = epoch_surf = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            pred = model({"x": x, "mask": mask})["preds"]
        pred = pred.float()

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = channel_loss(pred, y_norm, vol_mask)
        surf_loss = channel_loss(pred, y_norm, surf_mask)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # --- Validate ---
    model.eval()
    val_loss_sum = 0.0
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

                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = model({"x": x, "mask": mask})["preds"]
                pred = pred.float()

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += channel_loss(pred, y_norm, vol_mask).item()
                val_surf += channel_loss(pred, y_norm, surf_mask).item()
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
    avg_surf_p = sum(m[f"{n}/mae_surf_p"] for n, m in split_metrics.items()) / len(split_metrics)
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
    if mean_val_loss < best_val:
        best_val = mean_val_loss
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_surf_p": avg_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        shutil.copyfile(model_path, pvc_path)
        Path("checkpoints").mkdir(exist_ok=True)
        shutil.copyfile(model_path, "checkpoints/best.pt")
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}] avg_surf_p={avg_surf_p:.2f}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}, "
          f"avg_surf_p={best_metrics['avg_surf_p']:.2f}")
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    plot_dir = Path("plots") / run.id
    n = 1 if cfg.debug else 4
    for split_name, split_ds in val_splits.items():
        images = visualize(model, split_ds, stats, device, n_samples=n,
                           out_dir=plot_dir / split_name)
        if images:
            wandb.log({
                f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                "global_step": global_step,
            })

# Free GPU memory before launching predict subprocess
del model, optimizer, scheduler
if torch.cuda.is_available():
    torch.cuda.empty_cache()

wandb.finish()

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
