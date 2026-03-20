"""Train a CFD surrogate model.

Run:
  uv run train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import copy
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from data import X_DIM, VAL_SPLIT_NAMES


# ---------------------------------------------------------------------------
# Model: ResBlock MLP (4x FF expansion) with separate per-channel heads + EMA
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, ff_mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ChannelHead(nn.Module):
    """Per-channel output head with its own hidden layer."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class CFDModel(nn.Module):
    def __init__(self, in_dim=24, out_dim=3, hidden=256, n_blocks=8,
                 n_fourier=48, **kwargs):
        super().__init__()
        # Learnable Fourier features for spatial coordinates (dims 0-1)
        self.n_fourier = n_fourier
        self.fourier_B = nn.Parameter(
            torch.randn(2, n_fourier) * 2 * math.pi)
        # Input: original features + Fourier features (sin + cos)
        self.proj_in = nn.Linear(in_dim + 2 * n_fourier, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(hidden)
        self.heads = nn.ModuleList([ChannelHead(hidden) for _ in range(out_dim)])

    def forward(self, data, **kwargs):
        x = data["x"]
        # Compute Fourier features from spatial coords (dims 0-1)
        coords = x[..., :2]  # [B, N, 2]
        proj = coords @ self.fourier_B  # [B, N, n_fourier]
        fourier_feats = torch.cat([proj.sin(), proj.cos()], dim=-1)  # [B, N, 2*n_fourier]
        x_aug = torch.cat([x, fourier_feats], dim=-1)

        h = self.proj_in(x_aug)
        h = self.blocks(h)
        h = self.final_norm(h)
        return {"preds": torch.cat([head(h) for head in self.heads], dim=-1)}


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.lerp_(m_param.data, 1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import simple_parsing as sp
    import wandb
    import yaml
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm
    from data import pad_collate, load_data
    from viz import visualize

    MAX_TIMEOUT = 30.0  # minutes

    @dataclass
    class Config:
        lr: float = 3e-4
        weight_decay: float = 1e-4
        batch_size: int = 4
        accum_steps: int = 1
        surf_weight: float = 10.0
        epochs: int = 32
        grad_clip: float = 1.0
        seed: int = 42
        hidden: int = 256
        n_blocks: int = 8
        ema_decay: float = 0.998
        val_every: int = 3
        splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits"
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
        name: DataLoader(ds, batch_size=1, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks).to(device)
    model = torch.compile(model)
    ema = EMA(model, decay=cfg.ema_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=MAX_EPOCHS,
        steps_per_epoch=len(train_loader), pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda")

    # --- W&B ---
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v1"),
        group=cfg.wandb_group,
        name=cfg.wandb_name,
        tags=[cfg.agent] if cfg.agent else [],
        config={**asdict(cfg), "n_params": n_params,
                "train_samples": len(train_ds),
                "val_samples": {k: len(v) for k, v in val_splits.items()}},
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
        yaml.dump({"n_params": n_params, "hidden": cfg.hidden, "n_blocks": cfg.n_blocks}, f)

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
        optimizer.zero_grad()

        for step, (x, y, is_surface, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            finite = y.isfinite().all(dim=-1)
            mask = mask & finite
            y = y.clone()
            y[~finite] = 0.0
            x = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            # Mask out nodes with extreme normalized targets (outliers)
            reasonable = (y_norm.abs() <= 6).all(dim=-1)
            mask = mask & reasonable

            with torch.amp.autocast("cuda"):
                pred = model({"x": x})["preds"]
                # Smooth L1 (Huber) loss with beta=1.0
                err = torch.nn.functional.smooth_l1_loss(
                    pred, y_norm, reduction='none', beta=1.0)

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                vol_loss = (err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
                surf_loss = (err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
                loss = (vol_loss + cfg.surf_weight * surf_loss) / cfg.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % cfg.accum_steps == 0 or step == len(train_loader) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                ema.update(model)

            scheduler.step()
            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        epoch_vol /= n_batches
        epoch_surf /= n_batches

        # --- Validate (use EMA model, skip some epochs for speed) ---
        do_val = (epoch + 1) % cfg.val_every == 0 or epoch == MAX_EPOCHS - 1 or epoch == 0

        if not do_val:
            dt = time.time() - t0
            wandb.log({
                "train/vol_loss": epoch_vol,
                "train/surf_loss": epoch_surf,
                "lr": scheduler.get_last_lr()[0],
                "epoch_time_s": dt,
                "global_step": global_step,
            })
            print(f"Epoch {epoch+1:3d} ({dt:.0f}s)  train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  (skip val)")
            continue

        # Use EMA model for validation
        eval_model = ema.shadow
        eval_model.eval()
        torch.cuda.empty_cache()
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

                    finite = y.isfinite().all(dim=-1)
                    mask = mask & finite
                    y = y.clone()
                    y[~finite] = 0.0
                    x = (x - stats["x_mean"]) / stats["x_std"]
                    y_norm = (y - stats["y_mean"]) / stats["y_std"]
                    reasonable = (y_norm.abs() <= 6).all(dim=-1)
                    mask = mask & reasonable

                    pred = eval_model({"x": x})["preds"]
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
        finite_losses = [split_metrics[n][f"{n}/loss"] for n in VAL_SPLIT_NAMES if math.isfinite(split_metrics[n][f"{n}/loss"])]
        safe_val_loss = sum(finite_losses) / len(finite_losses) if finite_losses else float("inf")
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
        if safe_val_loss < best_val:
            best_val = safe_val_loss
            best_metrics = {"epoch": epoch + 1, "val_loss": safe_val_loss}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            # Save EMA weights as checkpoint
            torch.save(ema.state_dict(), model_path)
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

        # Load EMA checkpoint for visualization (strip _orig_mod. prefix from compile)
        eval_model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks).to(device)
        sd = torch.load(model_path, map_location=device, weights_only=True)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        eval_model.load_state_dict(sd)
        eval_model.eval()
        plot_dir = Path("plots") / run.id
        n = 1 if cfg.debug else 4
        for split_name, split_ds in val_splits.items():
            images = visualize(eval_model, split_ds, stats, device, n_samples=n,
                               out_dir=plot_dir / split_name)
            if images:
                wandb.log({
                    f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                    "global_step": global_step,
                })

    wandb.finish()
