"""Train a CFD surrogate model.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from data import X_DIM, VAL_SPLIT_NAMES

# Global conditioning feature indices (same for all nodes in a sample)
COND_DIMS = list(range(13, 24))  # Re, AoA1, NACA1(3), AoA2, NACA2(3), gap, stagger
# Per-node geometric dims for Fourier encoding (pos + saf + dsdf)
FOURIER_DIMS = list(range(12))  # x, z, saf(2), dsdf(8)


# ---------------------------------------------------------------------------
# Fourier positional encoding
# ---------------------------------------------------------------------------

class FourierEncoding(nn.Module):
    """Fourier features for spatial positions. Adds sin/cos at multiple frequencies."""
    def __init__(self, n_freq=8):
        super().__init__()
        self.n_freq = n_freq
        # Logarithmically spaced frequencies
        freqs = 2.0 ** torch.arange(n_freq).float()  # [1, 2, 4, ..., 128]
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self):
        return len(FOURIER_DIMS) * self.n_freq * 2  # sin + cos for each freq and dim

    def forward(self, x):
        # x: [B, N, D] — extract spatial dims
        pos = x[..., FOURIER_DIMS]  # [B, N, 2]
        # pos * freqs: [B, N, 2, n_freq]
        scaled = pos.unsqueeze(-1) * self.freqs  # broadcast
        # [B, N, 2*n_freq*2]
        fourier = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # [B, N, 2, 2*n_freq]
        return fourier.reshape(*pos.shape[:-1], -1)  # [B, N, 2*2*n_freq]


# ---------------------------------------------------------------------------
# FiLM-conditioned ResidualMLP with separate heads + Fourier encoding
# ---------------------------------------------------------------------------

class FiLMResBlock(nn.Module):
    """ResBlock with Feature-wise Linear Modulation from global conditioning."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()
        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        film_out = self.film(cond)
        gamma, beta = film_out.chunk(2, dim=-1)
        h = self.norm(x)
        h = h * (1 + gamma) + beta
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


class FiLMResidualMLP(nn.Module):
    def __init__(self, in_dim=24, out_dim=3, hidden=256, n_blocks=12,
                 cond_dim=len(COND_DIMS), n_fourier_freq=8):
        super().__init__()
        self.hidden = hidden
        self.fourier = FourierEncoding(n_freq=n_fourier_freq)
        effective_in = in_dim + self.fourier.out_dim

        n_pre = n_blocks // 2
        n_post = n_blocks - n_pre

        self.proj_in = nn.Linear(effective_in, hidden)
        self.pre_blocks = nn.ModuleList([FiLMResBlock(hidden, cond_dim) for _ in range(n_pre)])
        self.proj_mid = nn.Linear(hidden * 2, hidden)
        self.post_blocks = nn.ModuleList([FiLMResBlock(hidden, cond_dim) for _ in range(n_post)])

        self.head_vel = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.head_p = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        # Re-dependent pressure scaling for OOD Re generalization
        # Learns a scale factor as function of conditioning (esp. log(Re))
        self.p_scale = nn.Sequential(
            nn.Linear(cond_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, data, **kwargs):
        x_in = data["x"]  # [B, N, in_dim]
        cond = x_in[:, 0:1, COND_DIMS]  # [B, 1, cond_dim]

        # Append Fourier features to input
        fourier_feats = self.fourier(x_in)
        x_aug = torch.cat([x_in, fourier_feats], dim=-1)

        x = self.proj_in(x_aug)
        for block in self.pre_blocks:
            x = block(x, cond)
        g = x.max(dim=1, keepdim=True).values.expand_as(x)
        x = self.proj_mid(torch.cat([x, g], dim=-1))
        for block in self.post_blocks:
            x = block(x, cond)

        vel = self.head_vel(x)
        p_base = self.head_p(x)  # [B, N, 1]
        p_scale = self.p_scale(cond)  # [B, 1, 1]
        p = p_base * (1 + p_scale)  # Re-dependent scaling
        return {"preds": torch.cat([vel, p], dim=-1)}


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].lerp_(v, 1 - self.decay)

    def apply(self, model):
        """Swap model weights with EMA shadow. Call again to restore."""
        for k, v in model.state_dict().items():
            tmp = v.data.clone()
            v.data.copy_(self.shadow[k])
            self.shadow[k] = tmp


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
        lr: float = 5e-4
        weight_decay: float = 5e-4
        batch_size: int = 4
        val_batch_size: int = 2
        accum_steps: int = 1
        surf_weight: float = 10.0
        epochs: int = 27
        val_every: int = 24  # validate every N epochs
        ema_decay: float = 0.998
        n_fourier_freq: int = 4
        hidden: int = 256
        n_blocks: int = 8
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

    train_loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
    val_loader_kwargs = dict(collate_fn=pad_collate, num_workers=0, pin_memory=True)

    if cfg.debug:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, **train_loader_kwargs)
    else:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  sampler=sampler, **train_loader_kwargs)

    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False, **val_loader_kwargs)
        for name, ds in val_splits.items()
    }

    # --- Build model ---
    model = FiLMResidualMLP(
        in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks,
        n_fourier_freq=cfg.n_fourier_freq,
    ).to(device)
    model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader) // cfg.accum_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=MAX_EPOCHS,
        steps_per_epoch=steps_per_epoch,
    )

    scaler = torch.amp.GradScaler("cuda")
    ema = EMA(model, decay=cfg.ema_decay)

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
        yaml.dump({"n_params": n_params, "hidden": cfg.hidden, "n_blocks": cfg.n_blocks,
                    "n_fourier_freq": cfg.n_fourier_freq}, f)

    best_val = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        elapsed_min = (time.time() - train_start) / 60.0
        if elapsed_min >= MAX_TIMEOUT:
            print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
            break

        t0 = time.time()
        model.train()
        epoch_vol = epoch_surf = 0.0
        n_batches = 0

        for step_i, (x, y, is_surface, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x = (x - stats["x_mean"]) / stats["x_std"]
            finite = y.isfinite().all(dim=-1)
            y_safe = torch.where(finite.unsqueeze(-1), y, torch.zeros_like(y))
            y_norm = (y_safe - stats["y_mean"]) / stats["y_std"]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model({"x": x})["preds"]
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface & finite
                surf_mask = mask & is_surface & finite
                vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
                surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
                loss = (vol_loss + cfg.surf_weight * surf_loss) / cfg.accum_steps

            scaler.scale(loss).backward()

            if (step_i + 1) % cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                ema.update(model)

            epoch_vol += vol_loss.item() * cfg.accum_steps
            epoch_surf += surf_loss.item() * cfg.accum_steps
            n_batches += 1

        epoch_vol /= n_batches
        epoch_surf /= n_batches
        wandb.log({"train/vol_loss": epoch_vol, "train/surf_loss": epoch_surf,
                    "train/loss": epoch_vol + cfg.surf_weight * epoch_surf,
                    "global_step": global_step})

        # --- Skip validation on non-val epochs (except last) ---
        do_val = (epoch + 1) % cfg.val_every == 0 or epoch == MAX_EPOCHS - 1
        remaining_min = MAX_TIMEOUT - (time.time() - train_start) / 60.0
        if remaining_min < 3.0:
            do_val = True

        if not do_val:
            dt = time.time() - t0
            print(
                f"Epoch {epoch+1:3d} ({dt:.0f}s)  "
                f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  (skip val)"
            )
            continue

        # --- Validate with EMA weights ---
        ema.apply(model)  # swap to EMA weights
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
                    finite = y.isfinite().all(dim=-1)
                    y_safe = torch.where(finite.unsqueeze(-1), y, torch.zeros_like(y))
                    y_norm = (y_safe - stats["y_mean"]) / stats["y_std"]

                    pred = model({"x": x})["preds"].float()
                    sq_err = (pred - y_norm) ** 2

                    vol_mask = mask & ~is_surface & finite
                    surf_mask = mask & is_surface & finite
                    val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                    val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                    n_vb += 1

                    pred_orig = pred * stats["y_std"] + stats["y_mean"]
                    err = (pred_orig - y_safe).abs()
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

        ema.apply(model)  # swap back to training weights

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

    wandb.finish()
