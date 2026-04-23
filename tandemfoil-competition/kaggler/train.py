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
    batch_size: int = 8
    surf_weight: float = 10.0
    surf_p_weight: float = 0.0  # extra weight on surface pressure (scored metric)
    epochs: int = 25
    warmup_epochs: int = 2
    n_vol_train: int = 20000  # subsample volume points during training (0=keep all)
    resume_from: str | None = None  # path to checkpoint to initialize weights from
    select_by_surf_p: bool = False  # use avg mae_surf_p for best-ckpt selection
    ema_decay: float = 0.0  # >0 enables EMA with this decay
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


class EMA:
    """Exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def swap(self, model):
        """Swap model params with EMA params; returns the original params for restore."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


def subsample_volume(x, y, is_surface, mask, n_vol):
    """Keep all surface nodes, randomly subsample volume nodes to n_vol per sample.

    Returns a padded batch with smaller N.
    """
    B, _, D = x.shape
    kept_x, kept_y, kept_surf = [], [], []
    for i in range(B):
        m = mask[i]
        s = is_surface[i] & m
        v = m & ~is_surface[i]
        surf_idx = s.nonzero(as_tuple=True)[0]
        vol_idx = v.nonzero(as_tuple=True)[0]
        if len(vol_idx) > n_vol:
            perm = torch.randperm(len(vol_idx), device=vol_idx.device)[:n_vol]
            vol_idx = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_idx])
        kept_x.append(x[i, keep])
        kept_y.append(y[i, keep])
        kept_surf.append(is_surface[i, keep])
    max_n = max(t.shape[0] for t in kept_x)
    x_out = torch.zeros(B, max_n, D, device=x.device, dtype=x.dtype)
    y_out = torch.zeros(B, max_n, y.shape[-1], device=y.device, dtype=y.dtype)
    s_out = torch.zeros(B, max_n, device=x.device, dtype=torch.bool)
    m_out = torch.zeros(B, max_n, device=x.device, dtype=torch.bool)
    for i, (xi, yi, si) in enumerate(zip(kept_x, kept_y, kept_surf)):
        n = xi.shape[0]
        x_out[i, :n] = xi
        y_out[i, :n] = yi
        s_out[i, :n] = si
        m_out[i, :n] = True
    return x_out, y_out, s_out, m_out


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
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,
    n_layers=8,
    n_head=8,
    slice_num=128,
    mlp_ratio=2,
    use_checkpoint=True,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
if cfg.resume_from:
    model.load_state_dict(torch.load(cfg.resume_from, map_location=device, weights_only=True))
    print(f"Resumed weights from {cfg.resume_from}")
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, cfg.warmup_epochs))
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, MAX_EPOCHS - cfg.warmup_epochs))
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])

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

        if cfg.n_vol_train > 0:
            x, y, is_surface, mask = subsample_volume(x, y, is_surface, mask, cfg.n_vol_train)

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model({"x": x})["preds"].float()
            sq_err = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
            loss = vol_loss + cfg.surf_weight * surf_loss
            if cfg.surf_p_weight > 0:
                surf_p_loss = (sq_err[..., 2] * surf_mask).sum() / surf_mask.sum().clamp(min=1)
                loss = loss + cfg.surf_p_weight * surf_p_loss

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

    # --- Validate (with EMA weights if enabled) ---
    model.eval()
    ema_backup = ema.swap(model) if ema is not None else None
    val_loss_sum = 0.0
    split_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = n_vb = 0

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

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

    mean_val_loss = val_loss_sum / len(val_loaders)
    avg_mae_surf_p = sum(sm[f"{n}/mae_surf_p"] for n, sm in split_metrics.items()) / len(split_metrics)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "val/avg_mae_surf_p": avg_mae_surf_p,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    selection_metric = avg_mae_surf_p if cfg.select_by_surf_p else mean_val_loss
    tag = ""
    if selection_metric < best_val:
        best_val = selection_metric
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_mae_surf_p": avg_mae_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        # Save EMA weights if EMA enabled (model currently has EMA weights swapped in)
        torch.save(model.state_dict(), model_path)
        # Mirror to local checkpoints/ and PVC durable storage
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "best.pt")
        with open(ckpt_dir / "config.yaml", "w") as f:
            yaml.dump(model_config, f)
        pvc_dir = Path(f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG','default')}/{cfg.agent or 'unknown'}/checkpoints/model-{run.id}")
        pvc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pvc_dir / "checkpoint.pt")
        with open(pvc_dir / "config.yaml", "w") as f:
            yaml.dump(model_config, f)
        tag = " *"

    # Restore non-EMA weights for next training epoch
    if ema is not None and ema_backup is not None:
        ema.restore(model, ema_backup)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}]{tag}"
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
    print("\nGenerating test predictions...")
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-500:]}")

wandb.finish()
