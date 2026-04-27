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

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    p_weight: float = 3.0           # extra weight on pressure channel (the leaderboard metric)
    epochs: int = 200
    n_hidden: int = 256
    n_layers: int = 8
    n_head: int = 8
    slice_num: int = 96
    mlp_ratio: int = 2
    train_subsample: int = 40000    # subsample N nodes per training sample (keep all surface)
    bf16: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 3
    huber_beta: float = 0.1         # SmoothL1 transition; lower → more L1-like
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

def make_train_collate(n_max: int):
    """Random subsample of n_max nodes per training sample, but keep all surface nodes
    (they're rare and dominate the metric). Falls back to pad_collate if n_max is None.
    """
    if not n_max:
        return pad_collate

    def collate(batch):
        out = []
        for x, y, sf in batch:
            n = x.shape[0]
            if n <= n_max:
                out.append((x, y, sf))
                continue
            # Keep all surface nodes; subsample volume nodes uniformly.
            surf_idx = sf.nonzero(as_tuple=False).view(-1)
            vol_idx = (~sf).nonzero(as_tuple=False).view(-1)
            n_keep_vol = max(0, n_max - surf_idx.numel())
            if n_keep_vol < vol_idx.numel():
                perm = torch.randperm(vol_idx.numel())[:n_keep_vol]
                vol_idx = vol_idx[perm]
            keep = torch.cat([surf_idx, vol_idx])
            out.append((x[keep], y[keep], sf[keep]))
        return pad_collate(out)
    return collate


loader_kwargs = dict(num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=make_train_collate(cfg.train_subsample),
                              **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              collate_fn=make_train_collate(cfg.train_subsample),
                              **loader_kwargs)

# Validation runs on full meshes — no subsample.
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
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def lr_lambda(epoch: int) -> float:
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, MAX_EPOCHS - cfg.warmup_epochs)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Per-channel weights — extra emphasis on pressure (the leaderboard metric).
ch_w = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)


def smooth_l1(err: torch.Tensor, beta: float) -> torch.Tensor:
    abs_err = err.abs()
    return torch.where(abs_err < beta, 0.5 * err * err / beta, abs_err - 0.5 * beta)

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

# Mirror to git-tracked checkpoints/ for fast resume + predict.py
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(exist_ok=True)
with open(ckpt_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)

best_val = float("inf")          # primary selection: leaderboard metric
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

    autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if cfg.bf16 and torch.cuda.is_available()
                    else torch.amp.autocast(device_type="cuda", enabled=False))

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with autocast_ctx:
            pred = model({"x": x})["preds"]
            err = pred - y_norm
            sl1 = smooth_l1(err, cfg.huber_beta) * ch_w  # per-channel weighted

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (sl1 * vol_mask.unsqueeze(-1)).sum() / (vol_mask.sum().clamp(min=1) * ch_w.sum())
            surf_loss = (sl1 * surf_mask.unsqueeze(-1)).sum() / (surf_mask.sum().clamp(min=1) * ch_w.sum())
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

        with torch.no_grad(), autocast_ctx:
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                pred = model({"x": x})["preds"]
                err = (pred - y_norm).float()
                sl1 = smooth_l1(err, cfg.huber_beta) * ch_w

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sl1 * vol_mask.unsqueeze(-1)).sum().item() / (vol_mask.sum().clamp(min=1).item() * ch_w.sum().item())
                val_surf += (sl1 * surf_mask.unsqueeze(-1)).sum().item() / (surf_mask.sum().clamp(min=1).item() * ch_w.sum().item())
                n_vb += 1

                pred_orig = pred.float() * stats["y_std"] + stats["y_mean"]
                err_phys = (pred_orig - y).abs()
                mae_surf += (err_phys * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err_phys * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
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
    avg_surf_p = sum(
        split_metrics[s][f"{s}/mae_surf_p"] for s in VAL_SPLIT_NAMES
    ) / len(VAL_SPLIT_NAMES)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "val/avg_mae_surf_p": avg_surf_p,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    # Select on the actual leaderboard metric — avg surface-pressure MAE.
    if avg_surf_p < best_val:
        best_val = avg_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_loss": mean_val_loss,
            "avg_mae_surf_p": avg_surf_p,
        }
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), ckpt_dir / "best.pt")
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name.replace('val_', '')}={split_metrics[name][f'{name}/mae_surf_p']:.2f}"
        for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"surf_p[{split_summary}] avg={avg_surf_p:.2f}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(
        f"Best: epoch {best_metrics['epoch']}, "
        f"avg_mae_surf_p={best_metrics['avg_mae_surf_p']:.4f}, "
        f"val/loss={best_metrics['val_loss']:.4f}"
    )
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
