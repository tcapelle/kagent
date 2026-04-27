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
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    val_batch_size: int = 2  # validation runs on full meshes — keep small to fit in VRAM
    # Per-region/per-channel L1 weights. The leaderboard ranks by surface pressure MAE → big weight there.
    surf_p_weight: float = 6.0
    surf_uv_weight: float = 1.0
    vol_p_weight: float = 0.5
    vol_uv_weight: float = 0.5
    epochs: int = 8  # cosine anneals over this many epochs; pick to roughly match wall-clock budget
    grad_clip: float = 1.0
    bf16: bool = True
    train_subsample: int = 40000  # subsample non-surface nodes per training sample; surfaces always kept
    warm_start: str | None = None  # path to a previous checkpoint.pt to resume from
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

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,
    n_layers=6,
    n_head=8,
    slice_num=128,
    mlp_ratio=2,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
if cfg.warm_start:
    state = torch.load(cfg.warm_start, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Warm-start from {cfg.warm_start}: missing={len(missing)} unexpected={len(unexpected)}")
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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
config_path = model_dir / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(model_config, f)

# Mirror to PVC for durability
pvc_dir = Path(f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG','default')}/{cfg.agent or 'unknown'}/checkpoints/model-{run.id}")
pvc_dir.mkdir(parents=True, exist_ok=True)
pvc_path = pvc_dir / "checkpoint.pt"
pvc_cfg = pvc_dir / "config.yaml"
with open(pvc_cfg, "w") as f:
    yaml.dump(model_config, f)

# Also mirror to a stable git path for committing
git_ckpt = Path("checkpoints/best.pt")
git_ckpt.parent.mkdir(parents=True, exist_ok=True)

best_val = float("inf")
best_surf_p = float("inf")  # primary leaderboard metric
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

    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if cfg.bf16 else torch.amp.autocast("cuda", enabled=False)
    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Per-sample point subsampling. Always keep all surface nodes; randomly drop volume nodes
        # until each sample has at most `train_subsample` real points. Padding stays masked out.
        if cfg.train_subsample and cfg.train_subsample > 0:
            B, N, _ = x.shape
            # Importance: surface and real (mask) nodes get high score; padding gets very low score.
            # Top-k preserves all surface nodes when train_subsample >= n_surface for any sample.
            score = torch.rand(B, N, device=device)
            score = score + is_surface.float() * 10.0  # surfaces always selected first
            score = score + mask.float() * 1.0         # then real volume nodes, then padding (lowest)
            k = min(cfg.train_subsample, N)
            keep_idx = score.topk(k, dim=1).indices  # [B, k]
            gather_xy = keep_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = x.gather(1, gather_xy)
            y = y.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, y.shape[-1]))
            is_surface = is_surface.gather(1, keep_idx)
            mask = mask.gather(1, keep_idx)

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with autocast_ctx:
            pred = model({"x": x})["preds"]
        # L1 in normalised space matches the per-channel MAE metric (MAE_phys = std * MAE_norm).
        abs_err = (pred.float() - y_norm).abs()  # [B, N, 3]

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        # Split errors into UV (channels 0-1) and pressure (channel 2), each averaged over its region.
        uv_err = abs_err[..., :2].mean(dim=-1)
        p_err = abs_err[..., 2]
        n_surf = surf_mask.sum().clamp(min=1)
        n_vol = vol_mask.sum().clamp(min=1)
        l_surf_uv = (uv_err * surf_mask).sum() / n_surf
        l_surf_p = (p_err * surf_mask).sum() / n_surf
        l_vol_uv = (uv_err * vol_mask).sum() / n_vol
        l_vol_p = (p_err * vol_mask).sum() / n_vol
        loss = (
            cfg.surf_p_weight * l_surf_p
            + cfg.surf_uv_weight * l_surf_uv
            + cfg.vol_p_weight * l_vol_p
            + cfg.vol_uv_weight * l_vol_uv
        )
        vol_loss = l_vol_uv + l_vol_p
        surf_loss = l_surf_uv + l_surf_p

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

                with autocast_ctx:
                    pred = model({"x": x})["preds"]
                pred = pred.float()
                abs_err = (pred - y_norm).abs()

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (abs_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (abs_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vb, 1)
        val_surf /= max(n_vb, 1)
        split_loss = (
            cfg.vol_uv_weight * val_vol  # approximation: per-channel split for val_vol/val_surf is overkill
            + cfg.surf_p_weight * val_surf
        )
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

    # Primary leaderboard metric: avg surface pressure MAE across the 4 val splits.
    avg_surf_p = sum(
        split_metrics[name][f"{name}/mae_surf_p"] for name in VAL_SPLIT_NAMES
    ) / len(VAL_SPLIT_NAMES)
    metrics["val/avg_surf_p"] = avg_surf_p
    wandb.log({"val/avg_surf_p": avg_surf_p, "global_step": global_step})

    tag = ""
    if avg_surf_p < best_surf_p:
        best_surf_p = avg_surf_p
        best_val = mean_val_loss
        best_metrics = {
            "epoch": epoch + 1,
            "val_loss": mean_val_loss,
            "avg_surf_p": avg_surf_p,
        }
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), pvc_path)
        torch.save(model.state_dict(), git_ckpt)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}]  surf_p={avg_surf_p:.2f}{tag}"
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
    import gc
    import subprocess

    # Free GPU memory so the predict subprocess can load the model.
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    print("\nGenerating test predictions...")
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-1500:]}")

wandb.finish()
