"""Train Transolver on TandemFoilSet.

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
"""

import copy
import math
import os
import shutil
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


class EMA:
    """Exponential moving average of model weights for smoother evaluation."""
    def __init__(self, model, decay=0.999):
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)
        for sb, b in zip(self.shadow.buffers(), model.buffers()):
            sb.data.copy_(b.data)


MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", 30.0))


@dataclass
class Config:
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 15.0
    p_weight: float = 3.0          # extra weight on pressure channel in normalized loss
    epochs: int = 80
    warmup_epochs: int = 3
    grad_clip: float = 1.0
    train_max_nodes: int = 20000   # per-sample subsample target during training
    ema_decay: float = 0.999       # EMA decay for evaluation; set 0.0 to disable
    input_noise: float = 0.03      # gaussian noise stddev on normalized inputs at train time
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

# Per-channel loss weights in normalized space: weight pressure higher because the
# leaderboard ranks on surface pressure MAE.
chan_w = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)

def make_train_collate(max_nodes: int):
    """Subsample each training sample to <= max_nodes (keep all surface + random vol)."""
    def _collate(batch):
        new_batch = []
        for x, y, sf in batch:
            n = x.shape[0]
            if n <= max_nodes:
                new_batch.append((x, y, sf))
                continue
            surf_idx = torch.where(sf)[0]
            vol_idx = torch.where(~sf)[0]
            n_keep = max(0, max_nodes - surf_idx.shape[0])
            perm = torch.randperm(vol_idx.shape[0])[:n_keep]
            keep_vol = vol_idx[perm]
            keep = torch.cat([surf_idx, keep_vol])
            new_batch.append((x[keep], y[keep], sf[keep]))
        return pad_collate(new_batch)
    return _collate


train_collate = make_train_collate(cfg.train_max_nodes)

loader_kwargs = dict(num_workers=4, pin_memory=True,
                     persistent_workers=True, prefetch_factor=2)

if cfg.debug:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              collate_fn=train_collate, shuffle=True, **loader_kwargs)
else:
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              collate_fn=train_collate, sampler=sampler, **loader_kwargs)

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, collate_fn=pad_collate,
                     shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model_config = dict(
    space_dim=2,
    fun_dim=X_DIM - 2,
    out_dim=3,
    n_hidden=256,
    n_layers=4,
    n_head=8,
    slice_num=128,
    mlp_ratio=2,
    dropout=0.05,
    pos_freqs=8,
    pos_max_freq=32.0,
)

model = Transolver(**model_config).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.2f}M")

ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                               betas=(0.9, 0.95))

# Warmup + cosine LR.
def lr_lambda(epoch):
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / max(MAX_EPOCHS - cfg.warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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

# Best checkpoint mirrored for the kaggler harness.
LOCAL_BEST = Path("checkpoints/best.pt")
LOCAL_BEST.parent.mkdir(parents=True, exist_ok=True)
PVC_BEST_DIR = Path(
    f"/mnt/new-pvc/kagent/{os.environ.get('RESEARCH_TAG', 'default')}/"
    f"{os.environ.get('KAGGLER_NAME', cfg.agent or 'unknown')}/checkpoints/model-{run.id}"
)


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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        is_surface = is_surface.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        x = (x - stats["x_mean"]) / stats["x_std"]
        y_norm = (y - stats["y_mean"]) / stats["y_std"]

        if cfg.input_noise > 0:
            # Noise only on real (non-padded) nodes so padding stays exact zero.
            noise = torch.randn_like(x) * cfg.input_noise * mask.unsqueeze(-1).float()
            x = x + noise

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model({"x": x, "mask": mask})["preds"]
            sq_err = (pred - y_norm) ** 2

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            chan_w_b = chan_w.view(1, 1, 3)
            vol_sq = (sq_err * chan_w_b * vol_mask.unsqueeze(-1)).sum()
            surf_sq = (sq_err * chan_w_b * surf_mask.unsqueeze(-1)).sum()
            denom_chan = chan_w.sum()
            vol_loss = vol_sq / vol_mask.sum().clamp(min=1) / denom_chan * 3.0
            surf_loss = surf_sq / surf_mask.sum().clamp(min=1) / denom_chan * 3.0
            loss = vol_loss + cfg.surf_weight * surf_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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

    # --- Validate ---
    eval_model = ema.shadow if ema is not None else model
    eval_model.eval()
    val_loss_sum = 0.0
    split_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = n_vb = 0

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for x, y, is_surface, mask in vloader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                pred = eval_model({"x": x, "mask": mask})["preds"]
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item() / 3.0
                val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item() / 3.0
                n_vb += 1

                pred_orig = pred.float() * stats["y_std"] + stats["y_mean"]
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
    mean_surf_p = sum(m[f"{s}/mae_surf_p"] for s, m in split_metrics.items()) / len(split_metrics)
    dt = time.time() - t0

    metrics = {
        "train/vol_loss": epoch_vol,
        "train/surf_loss": epoch_surf,
        "val/loss": mean_val_loss,
        "val/avg_surf_p": mean_surf_p,
        "lr": scheduler.get_last_lr()[0],
        "epoch_time_s": dt,
        "global_step": global_step,
    }
    for sm in split_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    tag = ""
    # Track best by avg surface pressure MAE — that's the leaderboard metric.
    if mean_surf_p < best_val:
        best_val = mean_surf_p
        best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "avg_surf_p": mean_surf_p}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        # Save EMA weights (used for evaluation) so predict.py loads the same model.
        save_state = eval_model.state_dict()
        torch.save(save_state, model_path)
        shutil.copy(model_path, LOCAL_BEST)
        PVC_BEST_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, PVC_BEST_DIR / "checkpoint.pt")
        with open(PVC_BEST_DIR / "config.yaml", "w") as f:
            yaml.dump(model_config, f)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    split_summary = "  ".join(
        f"{name}={split_metrics[name][f'{name}/loss']:.4f}" for name in VAL_SPLIT_NAMES
    )
    print(
        f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
        f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
        f"val[{split_summary}]  surf_p={mean_surf_p:.2f}{tag}"
    )

# --- Final ---
total_time = (time.time() - train_start) / 60.0
print(f"\nDone ({total_time:.1f} min)")

if best_metrics:
    print(f"Best: epoch {best_metrics['epoch']}, avg_surf_p={best_metrics['avg_surf_p']:.2f}, val/loss={best_metrics['val_loss']:.4f}")
    wandb.summary.update({"best_" + k: v for k, v in best_metrics.items() if not k.startswith("best_")})
    wandb.summary.update({k: v for k, v in best_metrics.items() if k.startswith("best_")})

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

# --- Auto-submit predictions ---
if best_metrics and not cfg.debug:
    import gc
    import subprocess
    del model, optimizer, train_loader, val_loaders
    if ema is not None:
        del ema
    gc.collect()
    torch.cuda.empty_cache()
    print("\nGenerating test predictions...")
    pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
    if cfg.agent:
        pred_cmd += ["--agent", cfg.agent]
    result = subprocess.run(pred_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"predict.py failed:\n{result.stderr[-500:]}")
