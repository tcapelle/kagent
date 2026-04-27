"""Train Transolver on TandemFoilSet — surface-pressure focused.

Leaderboard ranks by avg surface-pressure MAE only, so the loss skews toward
surface p while keeping a smaller volume term so the field stays consistent.
Optionally warm-starts from a prior checkpoint.

Run:
  uv run train.py --agent <name> --wandb_name "<name>/<desc>" \
    [--warm_start <path>] [--lr 5e-5] [--epochs 30]
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


MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    epochs: int = 50
    grad_clip: float = 1.0

    # L1 loss in normalized space. Surface gets a global weight, plus an extra
    # per-channel multiplier on pressure (the leaderboard metric).
    surf_weight: float = 10.0
    p_weight: float = 5.0

    # Architecture.
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    slice_num: int = 128
    mlp_ratio: int = 2
    dropout: float = 0.0

    # Memory: subsample volume nodes per sample during training (keep all surface).
    train_subsample: int = 40000
    bf16: bool = True

    warm_start: str | None = None  # path to checkpoint to warm-start from
    skip_warmup: bool = False  # skip warmup when warm-starting
    save_last_k: int = 0  # save checkpoint after each of the last K epochs for SWA
    # Comma-separated multipliers for [racecar_single, racecar_tandem, cruise].
    # Default "1,1,1" = uniform domain sampling (same as data.py default).
    domain_weights: str = "1,1,1"
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
    """Keep all surface nodes, randomly subsample interior to ≤train_subsample."""
    if cfg.train_subsample <= 0:
        return pad_collate(batch)
    keep_n = cfg.train_subsample
    out = []
    for x, y, is_surf in batch:
        surf_idx = torch.nonzero(is_surf, as_tuple=False).squeeze(-1)
        vol_idx = torch.nonzero(~is_surf, as_tuple=False).squeeze(-1)
        if vol_idx.numel() > keep_n:
            perm = torch.randperm(vol_idx.numel())[:keep_n]
            vol_idx = vol_idx[perm]
        idx = torch.cat([surf_idx, vol_idx])
        out.append((x[idx], y[idx], is_surf[idx]))
    return pad_collate(out)


loader_kwargs = dict(num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=subsample_collate, **loader_kwargs)
else:
    domain_w = [float(x) for x in cfg.domain_weights.split(",")]
    domain_names = ["racecar_single", "racecar_tandem", "cruise"]
    assert len(domain_w) == len(domain_names), f"need {len(domain_names)} domain weights"
    import json as _json
    with open(f"{cfg.splits_dir}/meta.json") as _f:
        _meta = _json.load(_f)
    weights = sample_weights.clone()
    for name, w in zip(domain_names, domain_w):
        for i in _meta["domain_groups"][name]:
            weights[i] *= w
    print(f"Domain weights: {dict(zip(domain_names, domain_w))}")
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              collate_fn=subsample_collate, **loader_kwargs)

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
if cfg.warm_start:
    sd = torch.load(cfg.warm_start, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    print(f"Loaded warm-start from {cfg.warm_start}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
if cfg.skip_warmup:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
else:
    warmup_epochs = min(3, max(MAX_EPOCHS // 8, 1))
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(MAX_EPOCHS - warmup_epochs, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

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

ch_w_surf = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)

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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
            pred = model({"x": x})["preds"]
            err = (pred.float() - y_norm).abs()

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_err = err * ch_w_surf
        surf_loss = (surf_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
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

                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    pred = model({"x": x})["preds"]
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

    mean_val_loss = val_loss_sum / len(val_loaders)
    avg_surf_p = sum(split_metrics[s][f"{s}/mae_surf_p"] for s in VAL_SPLIT_NAMES) / len(VAL_SPLIT_NAMES)
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
    # Track best by avg_surf_p (the leaderboard metric).
    if avg_surf_p < best_val:
        best_val = avg_surf_p
        best_metrics = {"epoch": epoch + 1, "avg_surf_p": avg_surf_p, "val_loss": mean_val_loss}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        tag = " *"

    # Snapshot the last K epochs for offline SWA.
    if cfg.save_last_k > 0 and (MAX_EPOCHS - (epoch + 1)) < cfg.save_last_k:
        snap_path = model_dir / f"snap_e{epoch+1:03d}.pt"
        torch.save(model.state_dict(), snap_path)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name.replace('val_','')}={split_metrics[name][f'{name}/mae_surf_p']:.2f}"
        for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"surf_p[{split_summary}]  avg={avg_surf_p:.2f}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, avg_surf_p={best_metrics['avg_surf_p']:.2f}, val/loss={best_metrics['val_loss']:.4f}")
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

# --- Mirror best checkpoint to PVC for durability ---
if best_metrics and not cfg.debug:
    pvc_root = Path(os.environ.get("PVC_ROOT", "/mnt/new-pvc/kagent"))
    tag = os.environ.get("RESEARCH_TAG", "default")
    name = cfg.agent or os.environ.get("KAGGLER_NAME", "unknown")
    pvc_dir = pvc_root / tag / name / "checkpoints" / f"model-{run.id}"
    pvc_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(model_path, pvc_dir / "checkpoint.pt")
    shutil.copy2(model_dir / "config.yaml", pvc_dir / "config.yaml")
    # Mirror per-epoch snapshots (for offline SWA) too.
    for snap in sorted(model_dir.glob("snap_e*.pt")):
        shutil.copy2(snap, pvc_dir / snap.name)
    print(f"Mirrored checkpoint to {pvc_dir}")

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
