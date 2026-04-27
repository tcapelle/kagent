"""Train Transolver on TandemFoilSet.

Validation tracks (4):
  val_single_in_dist        — single-foil random holdout
  val_geom_camber_rc        — unseen front foil camber M=6-8 (raceCar)
  val_geom_camber_cruise    — unseen front foil camber M=2-4 (cruise)
  val_re_rand               — stratified Re holdout across tandem domains

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
"""

import contextlib
import gc
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


MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))  # minutes


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    surf_weight: float = 10.0
    epochs: int = 30
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    slice_num: int = 64
    mlp_ratio: int = 2
    use_bf16: bool = True
    subsample_n: int = 40000  # nodes/sample during training; 0 disables


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

# Validation uses full mesh; smaller batch to avoid OOM on big meshes.
val_loaders = {
    name: DataLoader(ds, batch_size=4, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}


def subsample(x, y, is_surface, mask, n_target: int):
    """Random per-sample subsample to ~n_target nodes; surface nodes always kept.

    Returns padded tensors of shape [B, n_target, ...] with mask updated.
    """
    B, N, D = x.shape
    if N <= n_target:
        return x, y, is_surface, mask
    new_N = n_target
    out_x = torch.zeros(B, new_N, D, device=x.device, dtype=x.dtype)
    out_y = torch.zeros(B, new_N, y.shape[-1], device=y.device, dtype=y.dtype)
    out_surf = torch.zeros(B, new_N, device=is_surface.device, dtype=is_surface.dtype)
    out_mask = torch.zeros(B, new_N, device=mask.device, dtype=mask.dtype)
    for i in range(B):
        valid = mask[i].nonzero(as_tuple=True)[0]
        n_valid = valid.numel()
        if n_valid <= new_N:
            sel = valid
        else:
            surf_local = valid[is_surface[i, valid]]
            n_surf = surf_local.numel()
            if n_surf >= new_N:
                perm = surf_local[torch.randperm(n_surf, device=x.device)[:new_N]]
                sel = perm
            else:
                vol_local = valid[~is_surface[i, valid]]
                n_keep = new_N - n_surf
                vol_perm = vol_local[torch.randperm(vol_local.numel(), device=x.device)[:n_keep]]
                sel = torch.cat([surf_local, vol_perm])
        n_sel = sel.numel()
        out_x[i, :n_sel] = x[i, sel]
        out_y[i, :n_sel] = y[i, sel]
        out_surf[i, :n_sel] = is_surface[i, sel]
        out_mask[i, :n_sel] = True
    return out_x, out_y, out_surf, out_mask


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
print(f"Model params: {n_params/1e6:.1f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)


def autocast_ctx():
    if cfg.use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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

        if cfg.subsample_n > 0:
            x, y, is_surface, mask = subsample(x, y, is_surface, mask, cfg.subsample_n)

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        with autocast_ctx():
            pred = model({"x": x})["preds"]
        diff = (pred.float() - y_norm).abs()  # L1 loss in normalized space

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        vol_loss = (diff * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
        surf_loss = (diff * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
        loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_vol += vol_loss.item()
        epoch_surf += surf_loss.item()
        n_batches += 1

    scheduler.step()
    epoch_vol /= max(n_batches, 1)
    epoch_surf /= max(n_batches, 1)

    # --- Validate (full mesh, no subsampling) ---
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

                with autocast_ctx():
                    pred = model({"x": x})["preds"]
                pred = pred.float()
                diff = (pred - y_norm).abs()

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (diff * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (diff * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
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
    avg_mae_surf_p = sum(sm[f"{n}/mae_surf_p"] for n, sm in zip(VAL_SPLIT_NAMES, split_metrics.values())) / len(split_metrics)
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

    tag = ""
    # Select on avg surface pressure MAE — the leaderboard ranks by this.
    if avg_mae_surf_p < best_val:
        best_val = avg_mae_surf_p
        best_metrics = {
            "epoch": epoch + 1,
            "val_loss": mean_val_loss,
            "avg_mae_surf_p": avg_mae_surf_p,
        }
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)

        local_best = Path("checkpoints/best.pt")
        local_best.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), local_best)

        research_tag = os.environ.get("RESEARCH_TAG", "default")
        agent_name = cfg.agent or os.environ.get("KAGGLER_NAME", "unknown")
        pvc_dir = Path(f"/mnt/new-pvc/kagent/{research_tag}/{agent_name}/checkpoints/model-{run.id}")
        pvc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pvc_dir / "checkpoint.pt")
        with open(pvc_dir / "config.yaml", "w") as f:
            yaml.dump(model_config, f)

        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[avg_surf_p={avg_mae_surf_p:.2f} {split_summary}]{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(
        f"Best: epoch {best_metrics['epoch']}, "
        f"avg_mae_surf_p={best_metrics['avg_mae_surf_p']:.3f}, "
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

wandb.finish()

# --- Auto-submit predictions (separate process; free GPU memory first) ---
if best_metrics and not cfg.debug:
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    import subprocess
    print("\nGenerating test predictions...")
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-1000:]}")
