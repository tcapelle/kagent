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
from collections.abc import Mapping
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
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

MAX_TIMEOUT = 30.0  # minutes


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    grad_accum: int = 1
    surf_weight: float = 30.0
    p_channel_weight: float = 2.0  # channel weight for pressure (metric)
    huber_delta: float = 0.1
    grad_clip: float = 1.0
    warmup_epochs: int = 1
    epochs: int = 80
    use_amp: bool = True
    vol_subsample: int = 20000  # max volume nodes per sample at training time
    cp_normalize: bool = True  # divide pressure target by exp(2*(log_re - LOG_RE_REF))
    velocity_norm: bool = False  # also divide Ux/Uy by Re-linear factor (iter 9 found this hurts)
    log_re_ref: float = 14.0  # reference log(Re) ~1.2M for Cp normalization
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
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

def subsample_collate(batch):
    """Surface-aware subsampling: keep all surface nodes, subsample volume to vol_subsample."""
    K = cfg.vol_subsample
    new = []
    for x, y, sf in batch:
        surf_idx = sf.nonzero(as_tuple=True)[0]
        vol_idx = (~sf).nonzero(as_tuple=True)[0]
        if K < vol_idx.numel():
            perm = torch.randperm(vol_idx.numel())[:K]
            vol_idx = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_idx])
        new.append((x[keep], y[keep], sf[keep]))
    return pad_collate(new)


train_loader_kwargs = dict(collate_fn=subsample_collate, num_workers=4, pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)
val_loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, **train_loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              sampler=sampler, **train_loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **val_loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=128,
    n_layers=6,
    n_head=8,
    slice_num=32,
    mlp_ratio=4,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

def lr_lambda(epoch):
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / cfg.warmup_epochs
    progress = (epoch - cfg.warmup_epochs) / max(MAX_EPOCHS - cfg.warmup_epochs, 1)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
ch_weights = torch.tensor([1.0, 1.0, cfg.p_channel_weight], device=device).view(1, 1, 3)


def _re_factor(x_raw):
    """Per-sample Re² scale factor for pressure: x_raw is [B, N, 24]. Returns [B, 1, 1]."""
    log_re = x_raw[:, 0, 13]
    return torch.exp(2.0 * (log_re - cfg.log_re_ref)).view(-1, 1, 1)


def _re_factor_v(x_raw):
    """Per-sample Re scale factor for velocity (linear in Re). Returns [B, 1, 1]."""
    log_re = x_raw[:, 0, 13]
    return torch.exp(log_re - cfg.log_re_ref).view(-1, 1, 1)


def _scale_y(y, x_raw):
    """Apply Cp/velocity normalization: y[..., 0:2] /= re_factor_v, y[..., 2] /= re_factor_p."""
    out = y
    if cfg.velocity_norm:
        rfv = _re_factor_v(x_raw)
        out = torch.cat([out[..., :2] / rfv, out[..., 2:3]], dim=-1)
    if cfg.cp_normalize:
        rfp = _re_factor(x_raw)
        out = torch.cat([out[..., :2], out[..., 2:3] / rfp], dim=-1)
    return out


def _unscale_pred(pred, x_raw):
    """Reverse normalization on predictions back to physical units."""
    out = pred
    if cfg.cp_normalize:
        rfp = _re_factor(x_raw)
        out = torch.cat([out[..., :2], out[..., 2:3] * rfp], dim=-1)
    if cfg.velocity_norm:
        rfv = _re_factor_v(x_raw)
        out = torch.cat([out[..., :2] * rfv, out[..., 2:3]], dim=-1)
    return out


# Recompute target stats over rescaled (Cp / velocity-normalized) targets.
if cfg.cp_normalize or cfg.velocity_norm:
    print("Computing rescaled-target stats...")
    import random
    random.seed(42)
    n_subset = min(200, len(train_ds))
    subset_idx = random.sample(range(len(train_ds)), n_subset)
    y_scaled_chunks = []
    for i in subset_idx:
        xr, yr, _ = train_ds[i]
        log_re_i = xr[0, 13].item()
        ys = yr.clone()
        if cfg.velocity_norm:
            rfv = float(torch.exp(torch.tensor(log_re_i - cfg.log_re_ref)))
            ys[:, 0] = ys[:, 0] / rfv
            ys[:, 1] = ys[:, 1] / rfv
        if cfg.cp_normalize:
            rfp = float(torch.exp(torch.tensor(2.0 * (log_re_i - cfg.log_re_ref))))
            ys[:, 2] = ys[:, 2] / rfp
        y_scaled_chunks.append(ys)
    y_scaled_all = torch.cat(y_scaled_chunks, dim=0)
    new_mean = y_scaled_all.mean(dim=0)
    new_std = y_scaled_all.std(dim=0)
    print(f"Rescaled stats (Ux,Uy,p): mean={new_mean.tolist()}, std={new_std.tolist()}")
    if cfg.velocity_norm:
        stats["y_mean"][0] = new_mean[0].item()
        stats["y_mean"][1] = new_mean[1].item()
        stats["y_std"][0] = new_std[0].item()
        stats["y_std"][1] = new_std[1].item()
    if cfg.cp_normalize:
        stats["y_mean"][2] = new_mean[2].item()
        stats["y_std"][2] = new_std[2].item()

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

# Save runtime info (Cp/velocity normalization params + adjusted stats) for predict.py.
runtime_info = {
    "cp_normalize": cfg.cp_normalize,
    "velocity_norm": cfg.velocity_norm,
    "log_re_ref": cfg.log_re_ref,
    "y_mean": [stats["y_mean"][i].item() for i in range(3)],
    "y_std": [stats["y_std"][i].item() for i in range(3)],
}
with open(model_dir / "runtime.yaml", "w") as f:
    yaml.dump(runtime_info, f)

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

        y = _scale_y(y, x)
        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
            pred = model({"x": x})["preds"]
            err = pred.float() - y_norm
            # Huber: L2 below delta, L1 above — robust to outliers, matches MAE direction.
            abs_err = err.abs()
            quad = torch.minimum(abs_err, torch.tensor(cfg.huber_delta, device=device))
            lin = abs_err - quad
            huber = 0.5 * quad ** 2 + cfg.huber_delta * lin
            ch_loss = huber * ch_weights

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (ch_loss * vol_mask.unsqueeze(-1)).sum() / (vol_mask.sum().clamp(min=1) * 3)
        surf_loss = (ch_loss * surf_mask.unsqueeze(-1)).sum() / (surf_mask.sum().clamp(min=1) * 3)
        loss = (vol_loss + cfg.surf_weight * surf_loss) / cfg.grad_accum

        loss.backward()
        if (n_batches + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        global_step += 1
        wandb.log({"train/loss": loss.item() * cfg.grad_accum, "global_step": global_step})

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

                x_raw = x
                y_phys = y
                y = _scale_y(y, x_raw)
                x = (x_raw - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.use_amp):
                    pred = model({"x": x})["preds"].float()
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                pred_orig = _unscale_pred(pred_orig, x_raw)
                err = (pred_orig - y_phys).abs()
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
    avg_surf_p = sum(sm[f"{n}/mae_surf_p"] for n, sm in split_metrics.items()) / len(split_metrics)
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
    # Select checkpoint by avg surface pressure MAE — the actual ranking metric.
    if avg_surf_p < best_val:
        best_val = avg_surf_p
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_surf_p": avg_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name[:10]}={split_metrics[name][f'{name}/mae_surf_p']:.2f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[surf={epoch_surf:.4f}]  "
        f"avg_surf_p={avg_surf_p:.2f}  surf_p[{split_summary}]{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, avg_surf_p={best_metrics['avg_surf_p']:.4f}, val/loss={best_metrics['val_loss']:.4f}")
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

wandb.finish()
