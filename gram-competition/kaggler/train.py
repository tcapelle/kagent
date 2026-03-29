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
    def __init__(self, in_dim=3, n_freqs=64):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * 10.0)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class AirflowMLP(nn.Module):
    """MLP with input normalization, residual prediction, temporal features, no-slip."""

    def __init__(self, hidden=512, n_blocks=10, n_fourier_freqs=64,
                 vel_mean=None, vel_std=None, dropout=0.0):
        super().__init__()
        self.fourier = FourierFeatures(in_dim=3, n_freqs=n_fourier_freqs)

        # Register normalization stats
        self.register_buffer("vel_mean", vel_mean if vel_mean is not None else torch.zeros(3))
        self.register_buffer("vel_std", vel_std if vel_std is not None else torch.ones(3))

        # Input features per point:
        # - Fourier pos: 2*n_fourier_freqs
        # - Normalized pos: 3
        # - Normalized velocity_in (5 timesteps): 5*3 = 15
        # - Velocity temporal diffs (4 diffs between consecutive timesteps): 4*3 = 12
        # - Per-point velocity mean and std: 3+3 = 6
        # - Airfoil indicator: 1
        in_dim = 2 * n_fourier_freqs + 3 + T_IN * 3 + 4 * 3 + 6 + 1
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocities
        vel_norm = (velocity_in - self.vel_mean) / self.vel_std  # [B, T, N, 3]

        # Fourier position features
        pos_feat = self.fourier(pos)  # [B, N, 2*n_freqs]

        # Temporal differences (acceleration-like features)
        vel_diffs = vel_norm[:, 1:] - vel_norm[:, :-1]  # [B, T-1, N, 3]
        vel_diffs_flat = vel_diffs.permute(0, 2, 1, 3).reshape(B, N, (T - 1) * C)  # [B, N, 12]

        # Per-point statistics
        vel_pt_mean = vel_norm.mean(dim=1)  # [B, N, 3]
        vel_pt_std = vel_norm.std(dim=1)    # [B, N, 3]

        # Flatten normalized velocities
        vel_flat = vel_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)  # [B, N, 15]

        # Airfoil indicator feature
        airfoil_feat = torch.zeros(B, N, 1, device=velocity_in.device)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                airfoil_feat[i, idcs_airfoil[i].to(velocity_in.device), 0] = 1.0

        # Concatenate all features
        x = torch.cat([pos_feat, pos, vel_flat, vel_diffs_flat, vel_pt_mean, vel_pt_std, airfoil_feat], dim=-1)

        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x).reshape(B, T_OUT, N, 3)

        # Predict delta in normalized space, then denormalize
        # Output = last_input + delta (in raw space)
        # delta_raw = delta_norm * vel_std
        delta_raw = delta_norm * self.vel_std

        pred = velocity_in[:, -1:, :, :] + delta_raw

        # No-slip boundary condition
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                pred[i, :, idcs_airfoil[i].to(pred.device), :] = 0.0

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
# Training
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
        lr: float = 5e-4
        weight_decay: float = 1e-4
        batch_size: int = 1
        epochs: int = 80
        subsample_train: int = 40000
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

    model = AirflowMLP(
        hidden=512, n_blocks=10, n_fourier_freqs=64,
        vel_mean=vel_mean, vel_std=vel_std, dropout=0.05,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Warmup + cosine decay
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(MAX_EPOCHS - warmup_epochs, 1)
        return 0.5 * (1 + __import__('math').cos(__import__('math').pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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

            # Subsample points during training
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
                # Smooth L1 (Huber) loss — more robust than MSE, better for L2 metric
                loss = nn.functional.smooth_l1_loss(pred, v_out_s)

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

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/l2_error={best_metrics['val_l2_error']:.4f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

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
