"""Train Transolver on TandemFoilSet.

Structured benchmark with four validation tracks:
  val_in_dist          — interpolation (raceCar single holdout)
  val_tandem_transfer  — unseen tandem front foil (Part2)
  val_ood_cond         — extreme conditions (frontier 20%)
  val_ood_re           — OOD Reynolds number (cruise Part2)

Run:
  uv run train.py [--debug]
"""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from models import Transolver
from viz import visualize


class SubsampleDataset(Dataset):
    """Wrapper that randomly subsamples volume nodes; surface nodes are always kept.

    Speeds up training on large meshes (cruise ~210K nodes) without losing the
    surface, where most of the loss weight is. The full mesh is still used at
    val/test time."""

    def __init__(self, base, n_volume: int):
        self.base = base
        self.n_volume = n_volume

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y, sf = self.base[idx]
        if sf.dtype != torch.bool:
            sf = sf.bool()
        surf_idx = torch.nonzero(sf, as_tuple=False).squeeze(-1)
        vol_idx = torch.nonzero(~sf, as_tuple=False).squeeze(-1)
        if vol_idx.numel() > self.n_volume:
            perm = torch.randperm(vol_idx.numel())[: self.n_volume]
            vol_idx = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_idx])
        return x[keep], y[keep], sf[keep]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = 30.0  # minutes


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    surf_weight: float = 20.0
    p_weight: float = 4.0
    epochs: int = 60
    train_n_volume: int = 32000
    val_every: int = 2  # validate every N epochs (saves time -> more train epochs)
    bf16: bool = True
    huber_beta: float = 0.05  # small beta -> nearly pure L1 (metric is L1 MAE);
    # quadratic only in tiny region near zero for gradient stability
    log_pressure: bool = True  # predict signed_log of normalized pressure
    ema_decay: float = 0.995  # exponential moving average; eval and predict on EMA
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


class EMA:
    """Exponential moving average of model parameters.
    Eval on the EMA copy — usually 1-3% smoother val/test errors with no extra train time."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if torch.is_floating_point(v):
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
                else:
                    self.shadow[k].copy_(v)

    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)


def signed_log(x):
    return torch.sign(x) * torch.log1p(x.abs())


def signed_log_inv(z):
    return torch.sign(z) * torch.expm1(z.abs())


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

# Wrap training dataset with mesh subsampling for speed (val/test stay full).
train_ds_sub = SubsampleDataset(train_ds, n_volume=cfg.train_n_volume)

loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds_sub, batch_size=cfg.batch_size,
                              shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds_sub, batch_size=cfg.batch_size,
                              sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=192,
    n_layers=7,
    n_head=8,
    slice_num=96,
    mlp_ratio=4,
    ff_n_freqs=32,
    ff_sigmas=[0.5, 1.0, 2.0],
    dropout=0.1,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

# Per-channel weights (Ux, Uy, p) — emphasize pressure since the leaderboard
# ranks by surface-pressure MAE.
ch_weights = torch.tensor([1.0, 1.0, cfg.p_weight], device=device).view(1, 1, 3)
amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float32

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
model_dir.mkdir(parents=True)
model_path = model_dir / "checkpoint.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

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

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        # Build target. If log_pressure, the network learns signed_log of the
        # normalized pressure channel (compresses heavy-tailed Cp values).
        if cfg.log_pressure:
            y_target = y_norm.clone()
            y_target[..., 2] = signed_log(y_norm[..., 2])
        else:
            y_target = y_norm

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.bf16):
            pred = model({"x": x})["preds"]
            err = (pred.float() - y_target)
            abs_err = err.abs()
            beta = cfg.huber_beta
            # smooth L1 / Huber: 0.5*x^2/beta for |x|<beta, |x|-0.5*beta otherwise
            huber = torch.where(abs_err < beta, 0.5 * err * err / beta, abs_err - 0.5 * beta)
            huber_w = huber * ch_weights  # weight pressure channel
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (huber_w * vol_mask.unsqueeze(-1)).sum() / (vol_mask.sum().clamp(min=1) * 3)
            surf_loss = (huber_w * surf_mask.unsqueeze(-1)).sum() / (surf_mask.sum().clamp(min=1) * 3)
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # Skip validation on most epochs to fit more training in 30 min.
    do_val = ((epoch + 1) % cfg.val_every == 0) or (epoch == 0) or (epoch == MAX_EPOCHS - 1)
    if not do_val:
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(
            f"Epoch {epoch+1:3d} ({time.time()-t0:.0f}s) [{peak_gb:.1f}GB]  "
            f"train[v={epoch_vol:.3f} s={epoch_surf:.3f}]  (val skipped)"
        )
        wandb.log({
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "lr": scheduler.get_last_lr()[0],
            "global_step": global_step,
        })
        continue

    # --- Validate (on EMA copy if available) ---
    if ema is not None:
        ema.apply_to(model)
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

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.bf16):
                    pred = model({"x": x})["preds"]
                pred = pred.float()

                # Build target matching training (apply signed_log to p channel)
                if cfg.log_pressure:
                    y_target = y_norm.clone()
                    y_target[..., 2] = signed_log(y_norm[..., 2])
                else:
                    y_target = y_norm

                err = pred - y_target
                abs_err = err.abs()
                beta = cfg.huber_beta
                huber = torch.where(abs_err < beta, 0.5 * err * err / beta, abs_err - 0.5 * beta)
                huber_w = huber * ch_weights

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (huber_w * vol_mask.unsqueeze(-1)).sum().item() / (vol_mask.sum().clamp(min=1).item() * 3)
                val_surf += (huber_w * surf_mask.unsqueeze(-1)).sum().item() / (surf_mask.sum().clamp(min=1).item() * 3)
                n_vb += 1

                # Invert log on pressure channel before denormalizing
                pred_norm = pred.clone()
                if cfg.log_pressure:
                    pred_norm[..., 2] = signed_log_inv(pred[..., 2])
                pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
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

    # Select checkpoint by surface-pressure MAE (the leaderboard metric)
    # rather than the combined val/loss. They usually agree but not always.
    surf_p_avg = sum(split_metrics[name][f"{name}/mae_surf_p"] for name in VAL_SPLIT_NAMES) / len(VAL_SPLIT_NAMES)
    tag = ""
    if surf_p_avg < best_val:
        best_val = surf_p_avg
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "surf_p_avg": surf_p_avg}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        # Save EMA weights (already loaded into model for val); they are what we
        # want to submit at inference time.
        torch.save(model.state_dict(), model_path)
        tag = " *"

    # Restore the trainable (non-EMA) weights so optimization continues normally.
    if ema is not None:
        ema.restore(model)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/mae_surf_p']:.1f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[v={epoch_vol:.3f} s={epoch_surf:.3f}]  "
        f"val_loss={mean_val_loss:.3f}  surf_p={surf_p_avg:.1f}  "
        f"surf_p[{split_summary}]{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}")
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

# --- Auto-submit predictions ---
if best_metrics and not cfg.debug:
    import subprocess
    # Free GPU memory before launching predict.py subprocess (otherwise OOM).
    del model
    torch.cuda.empty_cache()
    print("\nGenerating test predictions...")
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-500:]}")

wandb.finish()
