"""Train Transolver-based CFD surrogate.

Recipe (proven on this benchmark):
  - hidden=256, layers=8, slice_num=96, heads=8, mlp_ratio=2 (full Transolver)
  - bf16 AMP autocast for speed
  - Warmup + cosine LR schedule
  - EMA shadow weights (decay 0.999)
  - Huber loss with surf_weight=10
  - Gradient clipping
  - Best checkpoint by mean_mae_surf_p (the leaderboard metric)
"""

import os
import time
import copy
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from model import Transolver, EMA
from viz import visualize


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30))  # minutes


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    huber_beta: float = 1.0
    p_weight: float = 1.0  # multiplier on the pressure-channel loss (leaderboard metric)
    v_weight: float = 1.0  # multiplier on the velocity-channel loss
    epochs: int = 50
    grad_clip: float = 1.0
    warmup_steps: int = 100
    ema_decay: float = 0.999
    train_max_nodes: int = 80000  # subsample mesh for training; 0 disables
    keep_surface_nodes: bool = True  # always keep surface nodes when subsampling
    val_batch_size: int = 2
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    resume: str | None = None
    use_checkpoint: bool = False  # gradient checkpointing


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

torch.set_float32_matmul_precision("high")

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
    name: DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}


def subsample_batch(x, y, is_surface, mask, max_nodes, keep_surface):
    """Subsample mesh nodes per-sample to max_nodes for faster training.

    Returns padded subsampled tensors. Surface nodes are kept when keep_surface=True.
    """
    if max_nodes <= 0:
        return x, y, is_surface, mask
    B, N_max = x.shape[0], x.shape[1]
    counts = mask.sum(dim=1).long()  # [B]
    if counts.max().item() <= max_nodes:
        return x, y, is_surface, mask

    new_max = min(max_nodes, int(counts.max().item()))
    new_x = torch.zeros(B, new_max, x.shape[-1], device=x.device, dtype=x.dtype)
    new_y = torch.zeros(B, new_max, y.shape[-1], device=y.device, dtype=y.dtype)
    new_surf = torch.zeros(B, new_max, dtype=torch.bool, device=is_surface.device)
    new_mask = torch.zeros(B, new_max, dtype=torch.bool, device=mask.device)
    for b in range(B):
        n = int(counts[b].item())
        if n <= new_max:
            new_x[b, :n] = x[b, :n]
            new_y[b, :n] = y[b, :n]
            new_surf[b, :n] = is_surface[b, :n]
            new_mask[b, :n] = True
            continue
        if keep_surface:
            surf_idx = torch.nonzero(is_surface[b, :n], as_tuple=False).flatten()
            n_surf = surf_idx.numel()
            need = new_max - n_surf
            if need <= 0:
                # rare: surface alone exceeds budget — truncate
                pick = surf_idx[torch.randperm(n_surf, device=x.device)[:new_max]]
                idx = pick
            else:
                vol_mask = ~is_surface[b, :n]
                vol_idx = torch.nonzero(vol_mask, as_tuple=False).flatten()
                vol_pick = vol_idx[torch.randperm(vol_idx.numel(), device=x.device)[:need]]
                idx = torch.cat([surf_idx, vol_pick], dim=0)
        else:
            idx = torch.randperm(n, device=x.device)[:new_max]
        idx = idx.sort().values
        new_x[b, :idx.numel()] = x[b, idx]
        new_y[b, :idx.numel()] = y[b, idx]
        new_surf[b, :idx.numel()] = is_surface[b, idx]
        new_mask[b, :idx.numel()] = True
    return new_x, new_y, new_surf, new_mask

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,
    n_layers=8,
    n_head=8,
    slice_num=96,
    mlp_ratio=2,
    dropout=0.0,
    use_checkpoint=cfg.use_checkpoint,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Params: {n_params/1e6:.2f}M")

if cfg.resume:
    state = torch.load(cfg.resume, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Resumed from {cfg.resume}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# Warmup + cosine schedule
total_steps = MAX_EPOCHS * len(train_loader)


def lr_lambda(step):
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
ema = EMA(model, decay=cfg.ema_decay)

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
ema_path = model_dir / "checkpoint_ema.pt"
with open(model_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_score = float("inf")  # mean(mae_surf_p) — lower is better
best_metrics: dict = {}
best_is_ema = False
global_step = 0
train_start = time.time()

amp_dtype = torch.bfloat16


def compute_loss(pred, y_norm, mask, is_surface, surf_weight, beta,
                 p_weight=1.0, v_weight=1.0):
    """Huber loss split into volume and surface components, with per-channel weights.

    Channel order is [Ux, Uy, p]; v_weight applies to Ux and Uy, p_weight to p.
    """
    abs_err = (pred - y_norm).abs()
    sq = 0.5 * (pred - y_norm) ** 2
    huber = torch.where(abs_err < beta, sq / beta, abs_err - 0.5 * beta)

    ch_w = torch.tensor([v_weight, v_weight, p_weight], device=pred.device, dtype=huber.dtype)
    huber_w = huber * ch_w

    vol_mask = (mask & ~is_surface).unsqueeze(-1).float()
    surf_mask = (mask & is_surface).unsqueeze(-1).float()
    vol_loss = (huber_w * vol_mask).sum() / (vol_mask.sum().clamp(min=1.0) * ch_w.mean())
    surf_loss = (huber_w * surf_mask).sum() / (surf_mask.sum().clamp(min=1.0) * ch_w.mean())
    return vol_loss + surf_weight * surf_loss, vol_loss, surf_loss


def run_validation(model_to_eval):
    """Returns split_metrics, mean_val_loss, mean_mae_surf_p."""
    model_to_eval.eval()
    val_loss_sum = 0.0
    surf_p_sum = 0.0
    split_metrics = {}
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

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pred = model_to_eval({"x": x})["preds"]
                pred = pred.float()

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
        surf_p_sum += mae_surf[2].item()
    return split_metrics, val_loss_sum / len(val_loaders), surf_p_sum / len(val_loaders)


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
        x, y, is_surface, mask = subsample_batch(
            x, y, is_surface, mask, cfg.train_max_nodes, cfg.keep_surface_nodes
        )
        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            pred = model({"x": x})["preds"]
        loss, vol_loss, surf_loss = compute_loss(
            pred.float(), y_norm, mask, is_surface, cfg.surf_weight, cfg.huber_beta,
            p_weight=cfg.p_weight, v_weight=cfg.v_weight,
        )
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()
        ema.update(model)

        global_step += 1
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

    epoch_vol /= n_batches
    epoch_surf /= n_batches

    # Validate live model + EMA model
    split_metrics_live, mean_val_live, mean_surf_p_live = run_validation(model)

    # Apply EMA to a copy and validate
    ema_model = copy.deepcopy(model)
    ema.apply_to(ema_model)
    split_metrics_ema, mean_val_ema, mean_surf_p_ema = run_validation(ema_model)

    # Pick best of (live, ema) by mean_mae_surf_p
    if mean_surf_p_ema <= mean_surf_p_live:
        chosen_label = "ema"
        chosen_score = mean_surf_p_ema
        chosen_split_metrics = split_metrics_ema
        chosen_val_loss = mean_val_ema
    else:
        chosen_label = "live"
        chosen_score = mean_surf_p_live
        chosen_split_metrics = split_metrics_live
        chosen_val_loss = mean_val_live

    dt = time.time() - t0
    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": chosen_val_loss,
        "val/mean_mae_surf_p": chosen_score,
        "val/mean_mae_surf_p_live": mean_surf_p_live,
        "val/mean_mae_surf_p_ema": mean_surf_p_ema,
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in chosen_split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    if chosen_score < best_score:
        best_score = chosen_score
        best_metrics = {
            "epoch": epoch + 1,
            "val/mean_mae_surf_p": chosen_score,
            "val_loss": chosen_val_loss,
            "selected": chosen_label,
        }
        for sm in chosen_split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        # Save the chosen variant as best.pt
        if chosen_label == "ema":
            torch.save(ema_model.state_dict(), model_path)
            best_is_ema = True
        else:
            torch.save(model.state_dict(), model_path)
            best_is_ema = False
        # Also always save EMA shadow for diagnostic resume
        torch.save(ema.state_dict(), ema_path)
        tag = f" * ({chosen_label})"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={chosen_split_metrics[name][f'{name}/mae_surf_p']:.2f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"surf_p[{split_summary}]  mean={chosen_score:.2f}{tag}"
    )

    del ema_model

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, mean_mae_surf_p={best_metrics['val/mean_mae_surf_p']:.4f}")
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    # Mirror best to canonical paths
    research_tag = os.environ.get("RESEARCH_TAG", "default")
    kaggler_name = os.environ.get("KAGGLER_NAME", cfg.agent or "unknown")
    Path("checkpoints").mkdir(exist_ok=True)
    canonical = Path("checkpoints/best.pt")
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{research_tag}/{kaggler_name}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    torch.save(state, canonical)
    torch.save(state, pvc_dir / "checkpoint.pt")
    with open(pvc_dir / "config.yaml", "w") as f:
        yaml.dump(model_config, f)
    print(f"Mirrored best to: {canonical} and {pvc_dir / 'checkpoint.pt'}")

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
        print(f"predict.py failed:\n{result.stderr[-1000:]}")

wandb.finish()
