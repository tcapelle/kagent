"""Train Transolver on TandemFoilSet — surface-pressure-focused fine-tune.

Leaderboard ranks by avg surface-pressure MAE only, so the loss skews
toward surface p while still keeping a small volume term so the field
stays consistent. Optional warm-start from a prior checkpoint.

Run:
  uv run train.py --agent thorfinn --wandb_name "thorfinn/<desc>" \
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

    # Loss: L1 on denormalized values. Surface pressure is the leaderboard
    # metric, so we weight it most. Other components keep field consistent.
    surf_p_weight: float = 6.0
    surf_uv_weight: float = 1.0
    vol_p_weight: float = 0.5
    vol_uv_weight: float = 0.5
    # If "frieren": L1 in normalized space, surf_weight=10, p_weight=3.
    loss_kind: str = "default"
    surf_weight: float = 10.0
    p_weight: float = 3.0

    # Memory: subsample N points per sample during training.
    train_subsample: int = 40000
    grad_clip: float = 1.0
    warmup_frac: float = 0.05  # set 0 for warm-start
    n_hidden: int = 192
    n_layers: int = 6
    n_head: int = 6
    slice_num: int = 128
    mlp_ratio: int = 2
    dropout: float = 0.0

    warm_start: str | None = None  # path to checkpoint
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
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=0,
    fun_dim=X_DIM,
    out_dim=3,
    n_hidden=cfg.n_hidden,
    n_layers=cfg.n_layers,
    n_head=cfg.n_head,
    slice_num=cfg.slice_num,
    mlp_ratio=cfg.mlp_ratio,
    dropout=cfg.dropout,
    output_fields=["Ux", "Uy", "p"],
    output_dims=[1, 1, 1],
)
model = Transolver(**model_config).to(device)
if cfg.warm_start:
    sd = torch.load(cfg.warm_start, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Warm-start: {cfg.warm_start}  missing={len(missing)} unexpected={len(unexpected)}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
total_steps = max(MAX_EPOCHS * max(len(train_loader), 1), 1)
warmup_steps = int(cfg.warmup_frac * total_steps)


def lr_at(step: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    p = min(max(p, 0.0), 1.0)
    return 0.5 * (1.0 + torch.cos(torch.tensor(p * 3.14159265)).item())


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

# Mirror best checkpoint to durable PVC location.
pvc_ckpt_dir = Path(f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG', 'default')}"
                    f"/{cfg.agent or 'unknown'}/checkpoints/model-{run.id}")
pvc_ckpt_dir.mkdir(parents=True, exist_ok=True)
pvc_ckpt = pvc_ckpt_dir / "checkpoint.pt"
with open(pvc_ckpt_dir / "config.yaml", "w") as f:
    yaml.dump(model_config, f)


# Per-channel loss: L1 in physical units, split surface vs volume.
def compute_loss(pred_norm, y, surf_mask, vol_mask):
    pred = pred_norm * stats["y_std"] + stats["y_mean"]
    err = (pred - y).abs()  # [B, N, 3]

    surf_n = surf_mask.sum().clamp(min=1)
    vol_n = vol_mask.sum().clamp(min=1)
    e_surf = (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1)) / surf_n
    e_vol = (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1)) / vol_n

    if cfg.loss_kind == "frieren":
        # Frieren-style: L1 in normalized space, [Ux, Uy, p*p_weight] summed.
        # Surface side weighted by surf_weight.
        y_norm = (y - stats["y_mean"]) / stats["y_std"]
        err_norm = (pred_norm - y_norm).abs()  # [B, N, 3]
        ch_w = torch.tensor([1.0, 1.0, cfg.p_weight], device=err_norm.device)
        per_pt = (err_norm * ch_w).sum(-1)  # [B, N]
        surf_loss = (per_pt * surf_mask).sum() / surf_n
        vol_loss = (per_pt * vol_mask).sum() / vol_n
        loss = vol_loss + cfg.surf_weight * surf_loss
    else:
        # Normalize per channel by y_std so the scales are commensurate.
        s = stats["y_std"]
        surf_uv = (e_surf[0] + e_surf[1]) / (s[0] + s[1])
        surf_p = e_surf[2] / s[2]
        vol_uv = (e_vol[0] + e_vol[1]) / (s[0] + s[1])
        vol_p = e_vol[2] / s[2]
        loss = (
            cfg.surf_p_weight * surf_p
            + cfg.surf_uv_weight * surf_uv
            + cfg.vol_p_weight * vol_p
            + cfg.vol_uv_weight * vol_uv
        )

    return loss, dict(
        surf_p=e_surf[2].item(), surf_ux=e_surf[0].item(), surf_uy=e_surf[1].item(),
        vol_p=e_vol[2].item(), vol_ux=e_vol[0].item(), vol_uy=e_vol[1].item(),
    )


best_val = float("inf")
best_metrics: dict = {}
global_step = 0
train_start = time.time()

for epoch in range(MAX_EPOCHS):
    if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT - 1.5:
        print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
        break

    t0 = time.time()
    model.train()
    epoch_loss = 0.0
    n_batches = 0

    for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        # Subsample volume nodes per-sample to control memory while keeping
        # all surface nodes (they matter most for the metric).
        if cfg.train_subsample > 0:
            B, N = mask.shape
            keep = torch.zeros_like(mask)
            for b in range(B):
                surf_b = is_surface[b] & mask[b]
                vol_b = (~is_surface[b]) & mask[b]
                vol_idx = vol_b.nonzero(as_tuple=False).squeeze(-1)
                k = min(cfg.train_subsample, vol_idx.numel())
                if k > 0:
                    pick = vol_idx[torch.randperm(vol_idx.numel(), device=device)[:k]]
                    keep[b, pick] = True
                keep[b] |= surf_b
            # gather the active subset to a compact, padded layout
            new_max = int(keep.sum(1).max().item())
            x_s = torch.zeros(B, new_max, x.shape[-1], device=device, dtype=x.dtype)
            y_s = torch.zeros(B, new_max, y.shape[-1], device=device, dtype=y.dtype)
            surf_s = torch.zeros(B, new_max, device=device, dtype=torch.bool)
            mask_s = torch.zeros(B, new_max, device=device, dtype=torch.bool)
            for b in range(B):
                idx = keep[b].nonzero(as_tuple=False).squeeze(-1)
                k = idx.numel()
                x_s[b, :k] = x[b, idx]
                y_s[b, :k] = y[b, idx]
                surf_s[b, :k] = is_surface[b, idx]
                mask_s[b, :k] = True
            x, y, is_surface, mask = x_s, y_s, surf_s, mask_s

        x_norm = (x - stats["x_mean"]) / stats["x_std"]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model({"x": x_norm})["preds"]
        pred = pred.float()

        vol_mask = mask & ~is_surface
        surf_mask = mask & is_surface
        loss, comp = compute_loss(pred, y, surf_mask, vol_mask)

        # warmup + cosine LR
        for g in optimizer.param_groups:
            g["lr"] = cfg.lr * lr_at(global_step)

        optimizer.zero_grad()
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        global_step += 1

        epoch_loss += loss.item()
        n_batches += 1
        wandb.log({
            "train/loss": loss.item(),
            "train/surf_p": comp["surf_p"],
            "train/surf_uv": (comp["surf_ux"] + comp["surf_uy"]) / 2,
            "train/vol_p": comp["vol_p"],
            "lr": optimizer.param_groups[0]["lr"],
            "global_step": global_step,
        })

    epoch_loss /= max(n_batches, 1)

    # --- Validate ---
    model.eval()
    split_metrics: dict[str, dict] = {}
    avg_surf_p = 0.0

    for split_name, vloader in val_loaders.items():
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = 0
        sq_norm_vol = sq_norm_surf = 0.0
        n_vb = 0

        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_norm = model({"x": x_norm})["preds"]
                pred_norm = pred_norm.float()

                pred = pred_norm * stats["y_std"] + stats["y_mean"]
                err = (pred - y).abs()
                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface

                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

                y_norm = (y - stats["y_mean"]) / stats["y_std"]
                sq = (pred_norm - y_norm) ** 2
                sq_norm_vol += (sq * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                sq_norm_surf += (sq * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

        mae_surf = mae_surf / max(n_surf, 1)
        mae_vol = mae_vol / max(n_vol, 1)
        vol_loss_n = sq_norm_vol / max(n_vb, 1)
        surf_loss_n = sq_norm_surf / max(n_vb, 1)

        split_metrics[split_name] = {
            f"{split_name}/loss": vol_loss_n + 10.0 * surf_loss_n,
            f"{split_name}/vol_loss": vol_loss_n,
            f"{split_name}/surf_loss": surf_loss_n,
            f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
            f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
            f"{split_name}/mae_surf_p": mae_surf[2].item(),
            f"{split_name}/mae_vol_Ux": mae_vol[0].item(),
            f"{split_name}/mae_vol_Uy": mae_vol[1].item(),
            f"{split_name}/mae_vol_p": mae_vol[2].item(),
        }
        avg_surf_p += mae_surf[2].item()

    avg_surf_p /= len(val_loaders)
    dt = time.time() - t0

    metrics = {
        "train/epoch_loss": epoch_loss,
        "val/avg_surf_p": avg_surf_p,
        "val/loss": sum(s[f"{n}/loss"] for n, s in split_metrics.items()) / len(split_metrics),
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    if avg_surf_p < best_val:
        best_val = avg_surf_p
        best_metrics = {"epoch": epoch + 1, "avg_surf_p": avg_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        torch.save(model.state_dict(), pvc_ckpt)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/mae_surf_p']:.2f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"loss={epoch_loss:.4f}  avg_surf_p={avg_surf_p:.2f}  "
        f"[{split_summary}]{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, avg_surf_p={best_metrics['avg_surf_p']:.4f}")
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

# Auto-submit predictions
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
