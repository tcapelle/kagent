"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class FrierenModel(nn.Module):
    """Residual prediction with normalized velocity, time conditioning, and no-slip BC.

    Key ideas:
    - Normalize inputs/outputs using dataset stats for balanced gradients
    - Predict residual (delta from last input timestep) in normalized space
    - Enforce no-slip BC as hard constraint
    - Use time difference as a scalar conditioning signal
    """

    def __init__(self, vel_mean, vel_std, hidden=512, n_blocks=10):
        super().__init__()
        self.register_buffer('vel_mean', vel_mean.reshape(1, 1, 1, 3))  # [1,1,1,3]
        self.register_buffer('vel_std', vel_std.reshape(1, 1, 1, 3))    # [1,1,1,3]

        # Input: pos(3) + normalized velocity_in(5*3=15) + trend(3) + local_std(3) + time_diff(1) = 25
        in_dim = 3 + T_IN * 3 + 3 + 3 + 1
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # Initialize output layer to near-zero so initial prediction ≈ last input timestep
        nn.init.zeros_(self.proj_out[-1].weight)
        nn.init.zeros_(self.proj_out[-1].bias)

    def normalize(self, vel):
        return (vel - self.vel_mean) / self.vel_std

    def denormalize(self, vel):
        return vel * self.vel_std + self.vel_mean

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocity
        vel_norm = self.normalize(velocity_in)  # [B, 5, N, 3]

        # Time difference feature (scalar per sample, broadcast to all points)
        dt = (t[:, 5:].mean(dim=1) - t[:, :5].mean(dim=1))  # [B]
        dt_feat = dt.reshape(B, 1, 1).expand(B, N, 1)  # [B, N, 1]

        # Flatten velocity input
        vel_flat = vel_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)  # [B, N, 15]

        # Velocity trend and turbulence intensity
        vel_trend = vel_norm[:, -1] - vel_norm[:, 0]  # [B, N, 3]
        vel_local_std = vel_norm.std(dim=1)  # [B, N, 3]

        # Concatenate all features
        x = torch.cat([pos, vel_flat, vel_trend, vel_local_std, dt_feat], dim=-1)  # [B, N, 25]
        x = self.proj_in(x)  # [B, N, hidden]
        x = self.blocks(x)
        delta_norm = self.proj_out(x)  # [B, N, T_OUT*3]
        delta_norm = delta_norm.reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)  # [B, T_OUT, N, 3]

        # Residual: add last normalized input timestep
        last_norm = vel_norm[:, -1:, :, :]  # [B, 1, N, 3]
        out_norm = last_norm + delta_norm

        # Denormalize
        out = self.denormalize(out_norm)  # [B, T_OUT, N, 3]

        # No-slip BC
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                out[i, :, idcs_airfoil[i], :] = 0.0

        return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
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

                with torch.amp.autocast('cuda'):
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
# Config + main
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 80
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    subsample_train: int = 50000  # subsample during training for speed
    grad_accum: int = 1


if __name__ == "__main__":
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=1, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    vel_mean = stats['vel_mean'].to(device)
    vel_std = stats['vel_std'].to(device)
    model = FrierenModel(vel_mean, vel_std, hidden=512, n_blocks=10).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # T_max matches approximate training budget (30 min / ~37s per epoch)
    actual_epochs = min(MAX_EPOCHS, 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=actual_epochs)
    scaler = torch.amp.GradScaler("cuda")

    RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

    # --- W&B setup ---
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
        optimizer.zero_grad()

        for batch_idx, (v_in, v_out, pos, t, idcs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False)):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            # Optional subsampling
            if cfg.subsample_train > 0 and cfg.subsample_train < N_POINTS:
                idx = torch.randperm(N_POINTS, device=device)[:cfg.subsample_train]
                v_in = v_in[:, :, idx, :]
                v_out = v_out[:, :, idx, :]
                pos = pos[:, idx, :]
                B = v_in.shape[0]
                idx_cpu = idx.cpu()
                idx_set = set(idx_cpu.tolist())
                idx_to_new = {old: new for new, old in enumerate(idx_cpu.tolist())}
                new_idcs = []
                for i in range(B):
                    if idcs[i] is not None:
                        mapped = [idx_to_new[o] for o in idcs[i].tolist() if o in idx_set]
                        new_idcs.append(torch.tensor(mapped, dtype=torch.long))
                    else:
                        new_idcs.append(torch.tensor([], dtype=torch.long))
                idcs = new_idcs

            with torch.amp.autocast('cuda'):
                pred = model(v_in, pos, t, idcs)
                # Mixed loss: MSE + L2-norm (competition metric proxy)
                mse_loss = (pred - v_out).pow(2).mean()
                l2_norm_loss = (pred - v_out).norm(dim=3).mean()  # L2 norm per point
                loss = (0.5 * mse_loss + 0.5 * l2_norm_loss) / cfg.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            wandb.log({"train/loss": loss.item() * cfg.grad_accum, "global_step": global_step})
            epoch_loss += loss.item() * cfg.grad_accum
            n_batches += 1

        # Flush remaining gradients if last batch didn't complete accumulation
        if n_batches % cfg.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        # --- Validate ---
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
