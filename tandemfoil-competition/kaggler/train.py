"""Train Transolver on TandemFoilSet.

Optimised for the **avg surface pressure MAE** leaderboard: surface loss is
weighted heavily, with the surface-pressure channel given even more weight.

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
"""

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


MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    grad_clip: float = 1.0
    amp: bool = True
    # Heavy weighting on surface (the leaderboard scores surface pressure MAE).
    surf_weight: float = 30.0
    # L1 loss on surface pressure — directly aligned with the metric.
    surf_p_l1_weight: float = 0.0
    # Channel weights inside vol/surf losses.
    # Surface-pressure dominates the leaderboard so we bias the channel weights
    # toward p, but still keep a useful Ux/Uy signal so attention learns flow.
    surf_w_Ux: float = 0.3
    surf_w_Uy: float = 0.1
    surf_w_p: float = 1.0
    vol_w_Ux: float = 1.0
    vol_w_Uy: float = 0.3
    vol_w_p: float = 0.5
    # Training-time random subsampling per sample. 0 = full mesh.
    train_max_points: int = 80_000
    ema_decay: float = 0.999
    # Warm-start: load weights from this checkpoint before training.
    resume: str | None = None
    # Effective epochs for cosine schedule (>= actual epochs in the budget).
    effective_epochs: int = 16
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    # Model hyperparameters
    n_hidden: int = 256
    n_layers: int = 8
    n_head: int = 8
    slice_num: int = 64
    mlp_ratio: int = 2


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def store_and_swap(self, model: torch.nn.Module):
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(self.shadow[n])
        return backup

    @staticmethod
    def restore(model: torch.nn.Module, backup: dict):
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad and n in backup:
                    p.data.copy_(backup[n])


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}


def train_collate(batch, max_points=cfg.train_max_points):
    """Subsample each sample to at most `max_points` then pad."""
    new_batch = []
    for x, y, sf in batch:
        n = x.shape[0]
        if max_points > 0 and n > max_points:
            idx = torch.randperm(n)[:max_points]
            x, y, sf = x[idx], y[idx], sf[idx]
        new_batch.append((x, y, sf))
    return pad_collate(new_batch)


loader_common = dict(num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)
train_kwargs = dict(collate_fn=train_collate, **loader_common)
val_kwargs = dict(collate_fn=pad_collate, **loader_common)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **train_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **train_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=2, shuffle=False, **val_kwargs)
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
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
if cfg.resume:
    print(f"Resuming from: {cfg.resume}")
    state = torch.load(cfg.resume, map_location=device, weights_only=True)
    model.load_state_dict(state)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=cfg.effective_epochs, eta_min=cfg.lr * 0.02,
)
ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

surf_cw = torch.tensor([cfg.surf_w_Ux, cfg.surf_w_Uy, cfg.surf_w_p], device=device)
vol_cw = torch.tensor([cfg.vol_w_Ux, cfg.vol_w_Uy, cfg.vol_w_p], device=device)

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
print(f"Params: {n_params/1e6:.1f}M")

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

# Track best by avg surface-p MAE across the 4 splits — that's the leaderboard.
best_avg_surf_p = float("inf")
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

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
            pred = model({"x": x})["preds"]
            err = pred.float() - y_norm
            sq_err = err ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_cw * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_cw * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            # L1 on the surface pressure channel — directly aligned with metric.
            l1_p = err[..., 2].abs() * surf_mask
            surf_p_l1 = l1_p.sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss + cfg.surf_p_l1_weight * surf_p_l1

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        global_step += 1
        if global_step % 20 == 0:
            wandb.log({"train/loss_step": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # --- Validate using EMA weights when available ---
    ema_backup = ema.store_and_swap(model) if ema is not None else None
    model.eval()
    val_loss_sum = 0.0
    avg_surf_p = 0.0
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

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp):
                    pred = model({"x": x})["preds"].float()
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
        avg_surf_p += mae_surf[2].item()

    mean_val_loss = val_loss_sum / len(val_loaders)
    avg_surf_p /= len(val_loaders)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "train/loss": epoch_vol + cfg.surf_weight * epoch_surf,
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
    if avg_surf_p < best_avg_surf_p:
        best_avg_surf_p = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_loss": mean_val_loss,
            "avg_surf_p": avg_surf_p,
        }
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        tag = " *"

    if ema_backup is not None:
        EMA.restore(model, ema_backup)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    surf_p_summary = "  ".join(
        f"{name.replace('val_','')}_p={split_metrics[name][f'{name}/mae_surf_p']:.1f}"
        for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"avg_surf_p={avg_surf_p:.2f}  ({surf_p_summary}){tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, avg_surf_p={best_metrics['avg_surf_p']:.2f}")
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
    print("\nGenerating test predictions...")
    # Free GPU memory before subprocess so predict.py can allocate.
    del model, optimizer, scheduler
    if ema is not None:
        del ema
    torch.cuda.empty_cache()
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-500:]}")

wandb.finish()
