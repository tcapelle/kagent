"""Train Transolver on TandemFoilSet.

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
  python train.py --warm_start <ckpt> --lr 5e-5 --epochs 30 ...
"""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from model import Transolver


MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    p_weight: float = 3.0  # extra multiplier on surface pressure (primary metric)
    epochs: int = 30
    grad_clip: float = 1.0
    loss_type: str = "l1"  # mse | l1 | smoothl1
    slice_num: int = 64
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    mlp_ratio: int = 2
    train_subsample: int = 40000  # 0 = no subsampling
    bf16: bool = True
    warmup_epochs: int = 3
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    warm_start: str | None = None  # path to checkpoint for fine-tuning
    cp_normalize: bool = False  # divide pressure by exp(2*(log_re - LOG_RE_REF)) before training


cfg = sp.parse(Config)
MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
stats = {k: v.to(device) for k, v in stats.items()}


# --- Cp normalization ---
# Pressure scales as ρU^2 ∝ Re^2 in kinematic units. Across Re-regimes the std
# of pressure varies wildly (17 to 304). Dividing by exp(2*(log_re - LOG_RE_REF))
# flattens to a roughly-O(1) Cp coefficient, much easier to learn.
LOG_RE_REF = 0.0
P_MEAN_CP = stats["y_mean"][2].clone()
P_STD_CP = stats["y_std"][2].clone()
if cfg.cp_normalize:
    print("Computing Cp-normalization stats from 200 train samples...")
    log_res = []
    for i in range(min(200, len(train_ds))):
        xi, _, _ = train_ds[i]
        log_res.append(xi[0, 13].item())
    LOG_RE_REF = float(torch.tensor(log_res).median())
    p_cp_vals = []
    for i in range(min(200, len(train_ds))):
        xi, yi, _ = train_ds[i]
        log_re = xi[0, 13].item()
        re_factor = torch.exp(torch.tensor(2.0 * (log_re - LOG_RE_REF)))
        p_cp_vals.append(yi[:, 2] / re_factor)
    all_p_cp = torch.cat(p_cp_vals)
    P_MEAN_CP = all_p_cp.mean().to(device)
    P_STD_CP = all_p_cp.std().to(device)
    # Override pressure stats so the normalization in training uses Cp values
    stats["y_mean"] = stats["y_mean"].clone()
    stats["y_std"] = stats["y_std"].clone()
    stats["y_mean"][2] = P_MEAN_CP
    stats["y_std"][2] = P_STD_CP
    print(f"  LOG_RE_REF={LOG_RE_REF:.4f}, p_mean_cp={P_MEAN_CP.item():.4f}, p_std_cp={P_STD_CP.item():.4f}")


def cp_transform_y(x_unnorm, y):
    """Divide pressure channel y[..., 2] by re_factor in-place, return new y."""
    log_re = x_unnorm[..., 13]  # [B, N]
    re_factor = torch.exp(2.0 * (log_re - LOG_RE_REF))  # [B, N]
    y = y.clone()
    y[..., 2] = y[..., 2] / re_factor
    return y


def cp_undo_pred(x_unnorm, pred_phys):
    """Multiply predicted pressure channel back by re_factor."""
    log_re = x_unnorm[..., 13]
    re_factor = torch.exp(2.0 * (log_re - LOG_RE_REF))
    pred_phys = pred_phys.clone()
    pred_phys[..., 2] = pred_phys[..., 2] * re_factor
    return pred_phys


def subsample_collate(batch):
    """Per-sample random subsampling: keep all surface nodes, randomly sample
    up to cfg.train_subsample non-surface nodes, then pad."""
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
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
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
    print(f"Loaded warm-start weights from {cfg.warm_start}")
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params/1e6:.2f}M params")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# Warmup + cosine
warmup_eps = min(cfg.warmup_epochs, MAX_EPOCHS // 4)
warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, total_iters=max(warmup_eps, 1))
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(MAX_EPOCHS - warmup_eps, 1))
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup, cosine], milestones=[warmup_eps]
)

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

# Save Cp normalization stats so predict.py can apply the same transform
cp_runtime = {
    "cp_normalize": bool(cfg.cp_normalize),
    "log_re_ref": float(LOG_RE_REF),
    "p_mean_cp": float(P_MEAN_CP.item()),
    "p_std_cp": float(P_STD_CP.item()),
}
with open(model_dir / "runtime.yaml", "w") as f:
    yaml.dump(cp_runtime, f)

best_val = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()


def loss_fn(pred, target):
    diff = pred - target
    if cfg.loss_type == "l1":
        return diff.abs()
    if cfg.loss_type == "smoothl1":
        return torch.where(diff.abs() < 1.0, 0.5 * diff**2, diff.abs() - 0.5)
    return diff ** 2


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

        if cfg.cp_normalize:
            y = cp_transform_y(x, y)
        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
            pred = model({"x": x})["preds"]
            err = loss_fn(pred.float(), y_norm)

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
            ch_w_surf = torch.tensor([1.0, 1.0, cfg.p_weight], device=err.device)
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

                x_unnorm = x  # keep for Cp transform
                x = (x - stats["x_mean"]) / stats["x_std"]
                y_for_norm = cp_transform_y(x_unnorm, y) if cfg.cp_normalize else y
                y_norm = (y_for_norm - stats["y_mean"]) / stats["y_std"]

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
                if cfg.cp_normalize:
                    pred_orig = cp_undo_pred(x_unnorm, pred_orig)
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

    tag = ""
    if mean_val_loss < best_val:
        best_val = mean_val_loss
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        tag = " *"

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

# --- Auto-submit predictions FIRST (before slow viz) so kills are safe ---
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

# --- Mirror checkpoint to PVC for durability ---
if best_metrics:
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG','default')}/{cfg.agent or 'unknown'}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(model_path, pvc_dir / "checkpoint.pt")
    shutil.copy(model_dir / "config.yaml", pvc_dir / "config.yaml")
    print(f"Mirrored checkpoint to {pvc_dir}")

wandb.finish()
