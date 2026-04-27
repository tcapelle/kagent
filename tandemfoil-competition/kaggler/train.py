"""ResMLP surrogate for TandemFoilSet.

Per-node MLP with residual blocks and Fourier features on position. Inputs
already encode geometry context (saf + dsdf), so a point-wise ResMLP can
capture much of the field structure without explicit neighbour aggregation.

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
"""

import os
import time
import random
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from viz import visualize


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Random Fourier features applied to 2D position channels."""
    def __init__(self, n_freqs: int = 16, sigma: float = 5.0):
        super().__init__()
        B = torch.randn(2, n_freqs) * sigma
        self.register_buffer("B", B)

    def forward(self, pos):
        proj = pos @ self.B
        return torch.cat([torch.sin(proj * 2 * torch.pi),
                          torch.cos(proj * 2 * torch.pi)], dim=-1)


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class ResMLP(nn.Module):
    def __init__(self, in_dim: int = 24, hidden: int = 512, n_blocks: int = 8,
                 out_dim: int = 3, expansion: int = 4, dropout: float = 0.0,
                 n_freqs: int = 32, fourier_sigma: float = 4.0):
        super().__init__()
        self.fourier = FourierFeatures(n_freqs=n_freqs, sigma=fourier_sigma)
        embed_in = in_dim + 2 * n_freqs
        self.embed = nn.Sequential(nn.Linear(embed_in, hidden), nn.GELU())
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden, expansion, dropout) for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        x = data["x"]
        ff = self.fourier(x[..., :2])
        h = torch.cat([x, ff], dim=-1)
        h = self.embed(h)
        for block in self.blocks:
            h = block(h)
        return {"preds": self.head(h)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 20.0
    epochs: int = 60
    warmup_steps: int = 200
    train_subsample: int = 50000
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    hidden: int = 384
    n_blocks: int = 6
    expansion: int = 4
    dropout: float = 0.0
    n_freqs: int = 32
    fourier_sigma: float = 4.0
    # bf16 autocast
    amp: bool = True
    # validate every N epochs (after epoch 1)
    val_every: int = 2
    # cosine LR target ~= this many epochs
    cosine_epochs: int = 30


def subsample_batch(x, y, is_surface, mask, k_vol):
    """Keep all surface nodes; randomly sample k_vol volume nodes per sample."""
    B, _, D = x.shape
    new_x_list, new_y_list, new_s_list = [], [], []
    max_len = 0
    for b in range(B):
        surf_idx = (mask[b] & is_surface[b]).nonzero(as_tuple=False).squeeze(-1)
        vol_idx = (mask[b] & ~is_surface[b]).nonzero(as_tuple=False).squeeze(-1)
        if vol_idx.numel() > k_vol:
            perm = torch.randperm(vol_idx.numel(), device=vol_idx.device)[:k_vol]
            vol_idx = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_idx])
        new_x_list.append(x[b, keep])
        new_y_list.append(y[b, keep])
        new_s_list.append(is_surface[b, keep])
        max_len = max(max_len, keep.numel())

    out_x = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
    out_y = torch.zeros(B, max_len, y.shape[-1], device=y.device, dtype=y.dtype)
    out_s = torch.zeros(B, max_len, dtype=torch.bool, device=is_surface.device)
    out_m = torch.zeros(B, max_len, dtype=torch.bool, device=mask.device)
    for b in range(B):
        n = new_x_list[b].shape[0]
        out_x[b, :n] = new_x_list[b]
        out_y[b, :n] = new_y_list[b]
        out_s[b, :n] = new_s_list[b]
        out_m[b, :n] = True
    return out_x, out_y, out_s, out_m


def _validate(model, val_loaders, stats, device, surf_weight, amp=False):
    model.eval()
    val_loss_sum = 0.0
    surf_p_sum = 0.0
    split_metrics: dict[str, dict] = {}
    amp_dtype = torch.bfloat16 if amp else torch.float32

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

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
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
        split_loss = val_vol + surf_weight * val_surf
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
        surf_p_sum += mae_surf[2].item()

    return val_loss_sum / len(val_loaders), surf_p_sum / len(val_loaders), split_metrics


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

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
        in_dim=X_DIM,
        hidden=cfg.hidden,
        n_blocks=cfg.n_blocks,
        out_dim=3,
        expansion=cfg.expansion,
        dropout=cfg.dropout,
        n_freqs=cfg.n_freqs,
        fourier_sigma=cfg.fourier_sigma,
    )

    model = ResMLP(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * (cfg.cosine_epochs if not cfg.debug else MAX_EPOCHS)
    warmup = cfg.warmup_steps if not cfg.debug else 5

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v2"),
        group=cfg.wandb_group,
        name=cfg.wandb_name,
        tags=[cfg.agent] if cfg.agent else [],
        config={
            **asdict(cfg),
            "model": "ResMLP+Fourier",
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

    best_surf_p = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT - 2.0:
            print(f"Approaching timeout. Stopping at epoch {epoch}.")
            break

        t0 = time.time()
        model.train()
        epoch_vol = epoch_surf = 0.0
        n_batches = 0

        amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
        for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if cfg.train_subsample > 0:
                x, y, is_surface, mask = subsample_batch(x, y, is_surface, mask, cfg.train_subsample)

            x = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.amp):
                pred = model({"x": x})["preds"]
                sq_err = (pred.float() - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                vol_loss = (sq_err * vol_mask.unsqueeze(-1)).sum() / vol_mask.sum().clamp(min=1)
                surf_loss = (sq_err * surf_mask.unsqueeze(-1)).sum() / surf_mask.sum().clamp(min=1)
                loss = vol_loss + cfg.surf_weight * surf_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            wandb.log({
                "train/loss": loss.item(),
                "train/vol_loss_step": vol_loss.item(),
                "train/surf_loss_step": surf_loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "global_step": global_step,
            })

            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        epoch_vol /= n_batches
        epoch_surf /= n_batches

        # Validate on epoch 1 always, then every val_every epochs, and last epoch.
        do_val = (epoch == 0) or ((epoch + 1) % cfg.val_every == 0)
        if do_val:
            mean_val_loss, mean_surf_p, split_metrics = _validate(
                model, val_loaders, stats, device, cfg.surf_weight, amp=cfg.amp,
            )
        else:
            mean_val_loss, mean_surf_p, split_metrics = float("nan"), float("nan"), {}

        dt = time.time() - t0

        metrics = {
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "epoch_time_s": dt,
            "global_step": global_step,
        }
        if do_val:
            metrics.update({"val/loss": mean_val_loss, "val/mean_surf_p": mean_surf_p})
            for sm in split_metrics.values():
                metrics.update(sm)
        wandb.log(metrics)

        tag = ""
        if do_val and mean_surf_p < best_surf_p:
            best_surf_p = mean_surf_p
            best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "mean_surf_p": mean_surf_p}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(model.state_dict(), model_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        if do_val:
            split_summary = "  ".join(
                f"{name.replace('val_', '')}_p={split_metrics[name][f'{name}/mae_surf_p']:.2f}"
                for name in VAL_SPLIT_NAMES
            )
            print(
                f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
                f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
                f"surf_p[{split_summary}] mean={mean_surf_p:.2f}{tag}"
            )
        else:
            print(
                f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
                f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  (no val)"
            )

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}, "
              f"mean_surf_p={best_metrics['mean_surf_p']:.2f}")
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

        repo_ckpt_dir = Path("checkpoints")
        repo_ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), repo_ckpt_dir / "best.pt")
        research_tag = os.environ.get("RESEARCH_TAG", "default")
        kaggler_name = os.environ.get("KAGGLER_NAME", cfg.agent or "unknown")
        pvc_dir = Path(f"/mnt/new-pvc/kagent/{research_tag}/{kaggler_name}/checkpoints/model-{run.id}")
        pvc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pvc_dir / "checkpoint.pt")
        with open(pvc_dir / "config.yaml", "w") as f:
            yaml.dump(model_config, f)
        print(f"Mirrored checkpoint to {pvc_dir}")

    if best_metrics and not cfg.debug:
        # Free GPU memory before spawning predict.py subprocess (otherwise both
        # processes try to fit on one GPU and the child OOMs).
        import gc
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        print("\nGenerating test predictions...")
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-1500:]}")

    wandb.finish()


if __name__ == "__main__":
    main()
