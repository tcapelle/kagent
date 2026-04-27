"""Train Transolver on TandemFoilSet.

Recipe: 192/6/6 Transolver, L1 loss with p_weight=3 + surf_weight, train_subsample
to 40k pts/sample (all surface + random volume), bf16 autocast, warmup + cosine LR,
grad_clip=1.0. Validation runs full mesh in no_grad.

Run:
  uv run train.py [--debug] [--warm_start path/to/checkpoint.pt]
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


# ---------------------------------------------------------------------------
# Subsampling collate: training only — keeps all surface nodes + random volume
# ---------------------------------------------------------------------------

# Per-sample geometry params we may augment. dim 15 = foil1 NACA camber (M digit /9).
# val_geom_camber_rc tests OOD camber 0.78/0.89 not seen in training tandem set.
GEOM_AUG_DIMS = [15, 16, 17]  # foil1 camber (M), position (P), thickness (TT)


def make_subsample_collate(n_keep: int, camber_noise: float = 0.0,
                            aoa_noise: float = 0.0):
    """If n_keep <= 0, no subsampling (full mesh)."""
    def collate(batch):
        new_batch = []
        for x, y, is_surface in batch:
            if camber_noise > 0 or aoa_noise > 0:
                # Shared per-sample noise on geometry features (raw scale).
                noise = torch.zeros(x.shape[1])
                if camber_noise > 0:
                    # foil1 camber/position/thickness (dims 15/16/17)
                    noise[15] = torch.randn(1).item() * camber_noise
                    noise[16] = torch.randn(1).item() * camber_noise * 0.3
                    noise[17] = torch.randn(1).item() * camber_noise * 0.3
                if aoa_noise > 0:
                    # foil1 AoA (dim 14) and foil2 AoA (dim 18)
                    noise[14] = torch.randn(1).item() * aoa_noise
                    noise[18] = torch.randn(1).item() * aoa_noise
                x = x + noise.unsqueeze(0)
            n = x.shape[0]
            if n_keep <= 0 or n <= n_keep:
                new_batch.append((x, y, is_surface))
                continue
            surf_idx = torch.where(is_surface)[0]
            vol_idx = torch.where(~is_surface)[0]
            n_surf = surf_idx.numel()
            n_vol_keep = max(0, n_keep - n_surf)
            if 0 < n_vol_keep < vol_idx.numel():
                perm = torch.randperm(vol_idx.numel())[:n_vol_keep]
                vol_idx = vol_idx[perm]
            elif n_vol_keep == 0:
                vol_idx = vol_idx[:0]
            keep = torch.cat([surf_idx, vol_idx])
            keep, _ = torch.sort(keep)
            new_batch.append((x[keep], y[keep], is_surface[keep]))
        return pad_collate(new_batch)
    return collate


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    # Separate weights for surface vs volume, pressure vs U,V channels.
    # The leaderboard metric is surface pressure MAE — surf_p_weight dominates.
    surf_p_weight: float = 6.0
    surf_uv_weight: float = 1.0
    vol_p_weight: float = 0.5
    vol_uv_weight: float = 0.5
    epochs: int = 200
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    train_subsample: int = 40000
    camber_noise: float = 0.0  # Gaussian noise std on foil1 camber (raw NACA-M units)
    aoa_noise: float = 0.0  # Gaussian noise std on AoA foil1/foil2 (radians, raw scale)
    ema_decay: float = 0.0  # 0 = disabled. Common values: 0.99, 0.995, 0.999.
    fourier_freqs: int = 0  # 0 = disabled. Number of log-spaced freqs for position encoding.
    fourier_max_freq: float = 32.0
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    slice_num: int = 64
    mlp_ratio: int = 2
    dropout: float = 0.0
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    warm_start: str | None = None  # path to checkpoint.pt to warm-start from
    debug: bool = False


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}

train_collate = make_subsample_collate(
    cfg.train_subsample, camber_noise=cfg.camber_noise, aoa_noise=cfg.aoa_noise)
loader_kwargs = dict(num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, collate_fn=train_collate, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, collate_fn=train_collate, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, **loader_kwargs)
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
    dropout=cfg.dropout,
    fourier_freqs=cfg.fourier_freqs,
    fourier_max_freq=cfg.fourier_max_freq,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.2f}M")

if cfg.warm_start:
    print(f"Warm-starting from {cfg.warm_start}")
    state = torch.load(cfg.warm_start, map_location=device, weights_only=True)
    # strict=False: lets us load Fourier-adapted checkpoints into a Fourier
    # model where `fourier_freqs_buf` is registered fresh on the new model.
    model.load_state_dict(state, strict=False)

# EMA shadow weights — initialised from current model. Updated each opt step.
ema_state: dict | None = None
if cfg.ema_decay > 0:
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"EMA enabled (decay={cfg.ema_decay})")


def _update_ema():
    if ema_state is None:
        return
    d = cfg.ema_decay
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                ema_state[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                ema_state[k].copy_(v.detach())


def _swap_in_ema():
    """Swap model.state_dict() with EMA. Returns the saved (live) state for restore."""
    if ema_state is None:
        return None
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_state)
    return live


def _restore_live(saved):
    if saved is None:
        return
    model.load_state_dict(saved)


optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# Warmup + cosine schedule
if cfg.warm_start:
    # Skip warmup when warm-starting; LR is already a sensible value.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(MAX_EPOCHS, 1))
else:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(cfg.warmup_epochs, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(MAX_EPOCHS - cfg.warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])

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

# Per-region / per-channel weights for the loss
# Channels are [Ux, Uy, p]. We split them into uv (0,1) and p (2).
def _l1_uv_p(pred, target, mask):
    """Returns (mean L1 over uv channels, mean L1 over p channel) within masked region."""
    err = (pred - target).abs()  # [B, N, 3]
    m = mask.unsqueeze(-1).float()  # [B, N, 1]
    denom = m.sum().clamp(min=1.0)
    err_uv = (err[..., :2] * m).sum() / (denom * 2.0)
    err_p = (err[..., 2:3] * m).sum() / denom
    return err_uv, err_p


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

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model({"x": x})["preds"]
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_uv, vol_p = _l1_uv_p(pred, y_norm, vol_mask)
            surf_uv, surf_p = _l1_uv_p(pred, y_norm, surf_mask)
            vol_loss = cfg.vol_uv_weight * vol_uv + cfg.vol_p_weight * vol_p
            surf_loss = cfg.surf_uv_weight * surf_uv + cfg.surf_p_weight * surf_p
            loss = vol_loss + surf_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        _update_ema()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # --- Validate (use EMA weights if enabled) ---
    model.eval()
    saved_live = _swap_in_ema()
    val_loss_sum = 0.0
    val_surf_p_sum = 0.0
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

                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model({"x": x})["preds"]
                pred = pred.float()

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                vu, vp = _l1_uv_p(pred, y_norm, vol_mask)
                su, sp = _l1_uv_p(pred, y_norm, surf_mask)
                val_vol += (cfg.vol_uv_weight * vu + cfg.vol_p_weight * vp).item()
                val_surf += (cfg.surf_uv_weight * su + cfg.surf_p_weight * sp).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vb, 1)
        val_surf /= max(n_vb, 1)
        split_loss = val_vol + val_surf
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
        val_surf_p_sum += mae_surf[2].item()

    mean_val_loss = val_loss_sum / len(val_loaders)
    mean_surf_p = val_surf_p_sum / len(val_loaders)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "val/avg_surf_p": mean_surf_p,
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    # Use surf_p as the primary checkpoint criterion (matches the leaderboard metric)
    tag = ""
    if mean_surf_p < best_val:
        best_val = mean_surf_p
        best_metrics = {"epoch": epoch + 1, "val_surf_p": mean_surf_p, "val_loss": mean_val_loss}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        # Saves EMA-weighted state if EMA enabled (we're currently swapped in).
        torch.save(model.state_dict(), model_path)
        tag = " *"

    # Restore live (non-EMA) weights for the next training step.
    _restore_live(saved_live)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}] "
        f"val/avg_surf_p={mean_surf_p:.2f}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, val/avg_surf_p={best_metrics['val_surf_p']:.4f}")
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
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-500:]}")

# Mirror best checkpoint to PVC for durability
if best_metrics:
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG', 'default')}/"
                   f"{cfg.agent or 'unknown'}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(model_path, pvc_dir / "checkpoint.pt")
    shutil.copy(model_dir / "config.yaml", pvc_dir / "config.yaml")
    print(f"Mirrored checkpoint to {pvc_dir}")

wandb.finish()
