"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from data import T_IN, T_OUT


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Random Fourier features for positional encoding."""
    def __init__(self, in_dim=3, n_freqs=64):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * 10.0)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class EnhancedMLP(nn.Module):
    """Enhanced MLP with residual prediction, Fourier features, and no-slip enforcement."""

    def __init__(self, hidden=512, n_blocks=8, n_fourier_freqs=64, vel_mean=None, vel_std=None):
        super().__init__()
        self.n_fourier_freqs = n_fourier_freqs
        self.fourier = FourierFeatures(in_dim=3, n_freqs=n_fourier_freqs)

        # Input: fourier_pos(2*n_freqs) + pos(3) + velocity_in(5*3) + velocity_stats(3+3)
        in_dim = 2 * n_fourier_freqs + 3 + T_IN * 3 + 6
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean)
            self.register_buffer("vel_std", vel_std)
        else:
            self.register_buffer("vel_mean", torch.zeros(3))
            self.register_buffer("vel_std", torch.ones(3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        pos_feat = self.fourier(pos)  # [B, N, 2*n_freqs]

        vel_mean = velocity_in.mean(dim=1)  # [B, N, 3]
        vel_std = velocity_in.std(dim=1)    # [B, N, 3]
        vel_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]

        x = torch.cat([pos_feat, pos, vel_flat, vel_mean, vel_std], dim=-1)

        x = self.proj_in(x)
        x = self.blocks(x)
        delta = self.proj_out(x).reshape(B, T_OUT, N, 3)

        # Residual: add last input timestep as base
        pred = velocity_in[:, -1:, :, :] + delta

        # No-slip boundary condition
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                pred[i, :, idcs_airfoil[i], :] = 0.0

        return pred


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
    import wandb
    model.eval()
    val_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        total_l2 = 0.0
        total_mae = torch.zeros(3, device=device, dtype=torch.float64)
        n_samples = 0

        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    pred = model(v_in, pos, t, idcs)

                pred = pred.float()
                l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
                total_l2 += l2_err.sum().item()

                mae = (pred - v_out).abs().mean(dim=(1, 2))
                total_mae += mae.double().sum(dim=0)
                n_samples += v_in.shape[0]

        mean_l2 = total_l2 / max(n_samples, 1)
        mean_mae = total_mae / max(n_samples, 1)

        val_metrics[split_name] = {
            f"{split_name}/l2_error": mean_l2,
            f"{split_name}/mae_Ux": mean_mae[0].item(),
            f"{split_name}/mae_Uy": mean_mae[1].item(),
            f"{split_name}/mae_Uz": mean_mae[2].item(),
        }

    mean_val = sum(m[f"{k}/l2_error"] for k, m in val_metrics.items()) / len(val_metrics)

    metrics = {"val/l2_error": mean_val, "global_step": global_step}
    for sm in val_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    return mean_val, val_metrics


# ---------------------------------------------------------------------------
# Training (only runs as main script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import simple_parsing as sp
    import wandb
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from data import N_POINTS, VAL_SPLIT_NAMES, collate_fn, load_data

    MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))

    @dataclass
    class Config:
        lr: float = 3e-4
        weight_decay: float = 1e-4
        batch_size: int = 1
        epochs: int = 50
        subsample_train: int = 25000
        splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
        wandb_group: str | None = None
        wandb_name: str | None = None
        agent: str | None = None
        debug: bool = False

    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    vel_mean = stats["vel_mean"].to(device)
    vel_std = stats["vel_std"].to(device)

    model = EnhancedMLP(
        hidden=512, n_blocks=8, n_fourier_freqs=64,
        vel_mean=vel_mean, vel_std=vel_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")

    RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-gram"),
        group=cfg.wandb_group or RESEARCH_TAG,
        name=cfg.wandb_name,
        tags=[t for t in [cfg.agent, RESEARCH_TAG] if t],
        config={**asdict(cfg), "n_params": n_params,
                "train_samples": len(train_ds),
                "val_samples": {k: len(v) for k, v in val_splits.items()}},
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    model_dir = Path(f"models/model-{run.id}")
    model_dir.mkdir(parents=True)
    model_path = model_dir / "checkpoint.pt"

    # --- Training loop ---
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
        epoch_loss = 0.0
        n_batches = 0

        for v_in, v_out, pos, t, idcs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            B, T_i, N, C = v_in.shape

            # Subsample points during training for speed
            if cfg.subsample_train and cfg.subsample_train < N:
                idx = torch.randperm(N, device=device)[:cfg.subsample_train]
                idx = idx.sort().values
                v_in_s = v_in[:, :, idx, :]
                v_out_s = v_out[:, :, idx, :]
                pos_s = pos[:, idx, :]
                mask = torch.zeros(N, dtype=torch.bool, device=device)
                mask[idx] = True
                remap = torch.cumsum(mask, dim=0) - 1
                idcs_s = []
                for i in range(B):
                    if idcs[i] is not None and len(idcs[i]) > 0:
                        airfoil_mask = mask[idcs[i].to(device)]
                        kept = idcs[i].to(device)[airfoil_mask]
                        idcs_s.append(remap[kept])
                    else:
                        idcs_s.append(torch.tensor([], dtype=torch.long, device=device))
            else:
                v_in_s, v_out_s, pos_s, idcs_s = v_in, v_out, pos, idcs

            with torch.amp.autocast("cuda"):
                pred = model(v_in_s, pos_s, t, idcs_s)
                loss = (pred - v_out_s).pow(2).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": scheduler.get_last_lr()[0],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(model.state_dict(), model_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
            f"train={epoch_loss:.4f}  val/l2={mean_val:.4f}{tag}"
        )

    # --- Final ---
    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/l2_error={best_metrics['val_l2_error']:.4f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

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
