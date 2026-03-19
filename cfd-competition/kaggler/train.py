"""Train a CFD surrogate model.

Run:
  uv run train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import torch
import torch.nn as nn


class FiLMResBlock(nn.Module):
    """Residual block with FiLM conditioning from global flow parameters."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        # FiLM: condition generates scale and shift
        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        # Apply FiLM conditioning after first linear
        h = self.fc1(h)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + gamma) + beta  # FiLM modulation
        h = torch.nn.functional.gelu(h)
        h = self.fc2(h)
        return x + h


class CFDModel(nn.Module):
    def __init__(self, in_dim=24, out_dim=3, hidden=512, n_blocks=4, cond_dim=13):
        super().__init__()
        # cond_dim=13: features 13-23 (11) + log(Re^2) + log(Re^0.5) (2)
        self.cond_start = 13
        self.cond_end = 24
        self.proj_in = nn.Linear(in_dim, hidden)
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden // 2),
        )
        self.global_proj = nn.Linear(hidden, hidden // 4)
        self.cond_merge = nn.Linear(hidden // 2 + hidden // 4, hidden // 2)
        self.blocks = nn.ModuleList([FiLMResBlock(hidden, hidden // 2) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(hidden)
        self.vel_head = nn.Linear(hidden, 2)
        self.p_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data, **kwargs):
        x_in = data["x"]  # [B, N, 24]
        # Extract global conditioning (same for all nodes in a sample)
        cond_raw = x_in[:, 0, self.cond_start:self.cond_end]  # [B, 11] - take from first node
        # Add physics-motivated features: Re^2 and Re^0.5 for pressure scaling
        log_re = cond_raw[:, 0:1]  # log(Re)
        re_sq = (2 * log_re)  # log(Re^2) = 2*log(Re)
        re_sqrt = (0.5 * log_re)  # log(Re^0.5) = 0.5*log(Re)
        cond_aug = torch.cat([cond_raw, re_sq, re_sqrt], dim=-1)  # [B, 13]
        cond = self.cond_proj(cond_aug)  # [B, hidden//2]

        # Add global context from mean of projected features
        x = self.proj_in(x_in)
        global_ctx = x.mean(dim=1)  # [B, hidden]
        global_feat = self.global_proj(global_ctx)  # [B, hidden//4]
        cond = torch.cat([cond, global_feat], dim=-1)  # [B, hidden//2 + hidden//4]
        cond = self.cond_merge(cond)  # [B, hidden//2]
        cond = cond.unsqueeze(1)  # [B, 1, hidden//2] for broadcasting
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm(x)
        vel = self.vel_head(x)
        p = self.p_head(x)
        return {"preds": torch.cat([vel, p], dim=-1)}


if __name__ == "__main__":
    import os
    import time
    from dataclasses import dataclass, asdict
    from pathlib import Path

    import simple_parsing as sp
    import wandb
    import yaml
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm

    from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
    from viz import visualize

    MAX_TIMEOUT = 30.0  # minutes

    @dataclass
    class Config:
        lr: float = 1e-3
        weight_decay: float = 5e-4
        batch_size: int = 4
        val_batch_size: int = 2
        surf_weight: float = 10.0
        hidden: int = 512
        n_blocks: int = 4
        val_every: int = 5
        epochs: int = 45
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
        name: DataLoader(ds, batch_size=cfg.val_batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    raw_model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks).to(device)
    model = torch.compile(raw_model)

    # EMA for better generalization
    ema_model = torch.optim.swa_utils.AveragedModel(raw_model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # AMP for faster training + lower memory
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

        for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            # Exclude nodes with inf/nan targets
            valid_y = y.isfinite().all(dim=-1)
            mask = mask & valid_y

            # Normalize inputs and targets
            x = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]
            y_norm = torch.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.amp.autocast("cuda"):
                pred = model({"x": x})["preds"]
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
                surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
                loss = vol_loss + cfg.surf_weight * surf_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema_model.update_parameters(raw_model)

            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        scheduler.step()
        epoch_vol /= n_batches
        epoch_surf /= n_batches

        do_val = (epoch + 1) % cfg.val_every == 0 or epoch == 0 or epoch == MAX_EPOCHS - 1
        if not do_val:
            dt = time.time() - t0
            wandb.log({"train/vol_loss": epoch_vol, "train/surf_loss": epoch_surf,
                        "train/loss": epoch_vol + cfg.surf_weight * epoch_surf,
                        "lr": scheduler.get_last_lr()[0], "global_step": global_step})
            print(f"Epoch {epoch+1:3d} ({dt:.0f}s) train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}] (skip val)")
            continue

        # --- Validate using EMA model ---
        model.eval()
        ema_model.eval()
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

                    # Exclude nodes with inf/nan targets
                    valid_y = y.isfinite().all(dim=-1)
                    mask = mask & valid_y

                    x = (x - stats["x_mean"]) / stats["x_std"]
                    y_norm = (y - stats["y_mean"]) / stats["y_std"]
                    y_norm = torch.nan_to_num(y_norm, nan=0.0, posinf=0.0, neginf=0.0)

                    pred = ema_model({"x": x})["preds"]
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
            torch.save(ema_model.module.state_dict(), model_path)
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

        ema_model.module.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        plot_dir = Path("plots") / run.id
        n = 1 if cfg.debug else 4
        for split_name, split_ds in val_splits.items():
            images = visualize(ema_model, split_ds, stats, device, n_samples=n,
                               out_dir=plot_dir / split_name)
            if images:
                wandb.log({
                    f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                    "global_step": global_step,
                })

    wandb.finish()
