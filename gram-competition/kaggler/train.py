"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import math
import time
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn

from data import T_IN, T_OUT


# ---------------------------------------------------------------------------
# Model: Autoregressive single-step predictor
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=3, n_freqs=128):
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
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class AutoregressivePredictor(nn.Module):
    """Predicts one timestep at a time, autoregressively."""

    def __init__(self, hidden=512, n_blocks=10, n_fourier_freqs=96,
                 vel_mean=None, vel_std=None, window=5):
        super().__init__()
        self.window = window
        self.fourier = FourierFeatures(in_dim=3, n_freqs=n_fourier_freqs)

        self.register_buffer("vel_mean", vel_mean if vel_mean is not None else torch.zeros(3))
        self.register_buffer("vel_std", vel_std if vel_std is not None else torch.ones(3))

        in_dim = 2 * n_fourier_freqs + 3 + window * 3 + (window - 1) * 3 + 1
        out_dim = 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def predict_single_step(self, vel_window_norm, pos, pos_feat, airfoil_feat):
        B, W, N, C = vel_window_norm.shape
        diffs = vel_window_norm[:, 1:] - vel_window_norm[:, :-1]
        diffs_flat = diffs.permute(0, 2, 1, 3).reshape(B, N, (W - 1) * C)
        vel_flat = vel_window_norm.permute(0, 2, 1, 3).reshape(B, N, W * C)
        x = torch.cat([pos_feat, pos, vel_flat, diffs_flat, airfoil_feat], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x)
        return vel_window_norm[:, -1] + delta_norm

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        vel_norm = (velocity_in - self.vel_mean) / self.vel_std
        pos_feat = self.fourier(pos)

        airfoil_feat = torch.zeros(B, N, 1, device=velocity_in.device)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                airfoil_feat[i, idcs_airfoil[i].to(velocity_in.device), 0] = 1.0

        preds_norm = []
        window = vel_norm

        for step in range(T_OUT):
            next_norm = self.predict_single_step(window, pos, pos_feat, airfoil_feat)
            preds_norm.append(next_norm)
            window = torch.cat([window[:, 1:], next_norm.unsqueeze(1)], dim=1)

        preds_norm = torch.stack(preds_norm, dim=1)
        pred = preds_norm * self.vel_std + self.vel_mean

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
# Training with scheduled sampling
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
        epochs: int = 200
        subsample_train: int = 10000
        grad_accum: int = 4
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

    model = AutoregressivePredictor(
        hidden=512, n_blocks=10, n_fourier_freqs=96,
        vel_mean=vel_mean, vel_std=vel_std, window=T_IN,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_steps = 300
    total_steps = MAX_EPOCHS * len(train_ds) // cfg.grad_accum
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

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

    # --- Training loop with scheduled sampling ---
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

        # Scheduled sampling: increase use of model predictions over time
        # Start with pure teacher forcing, linearly increase to 50% autoregressive
        elapsed_frac = min((time.time() - train_start) / (MAX_TIMEOUT * 60), 1.0)
        ss_prob = min(0.5, elapsed_frac * 0.5)  # prob of using model prediction

        optimizer.zero_grad()

        for batch_idx, (v_in, v_out, pos, t, idcs) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            B, T_i, N, C = v_in.shape

            # Subsample points
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
                vel_norm = (v_in_s - model.vel_mean) / model.vel_std
                out_norm = (v_out_s - model.vel_mean) / model.vel_std
                pos_feat = model.fourier(pos_s)

                airfoil_feat = torch.zeros(B, v_in_s.shape[2], 1, device=device)
                for i in range(B):
                    if idcs_s[i] is not None and len(idcs_s[i]) > 0:
                        airfoil_feat[i, idcs_s[i].to(device), 0] = 1.0

                total_loss = torch.tensor(0.0, device=device)
                all_vel_gt = torch.cat([vel_norm, out_norm], dim=1)  # [B, 10, N, 3]

                # Scheduled sampling: build window mixing GT and predictions
                window = vel_norm.clone()  # [B, 5, N, 3] - start with GT input

                for step in range(T_OUT):
                    target = all_vel_gt[:, step + T_IN]  # [B, N, 3]
                    pred_norm = model.predict_single_step(window, pos_s, pos_feat, airfoil_feat)
                    total_loss = total_loss + (pred_norm - target).pow(2).mean()

                    # Decide: use GT or prediction for next window
                    if random.random() < ss_prob:
                        # Use model prediction (autoregressive)
                        next_step = pred_norm.detach()
                    else:
                        # Use ground truth (teacher forcing)
                        next_step = all_vel_gt[:, step + T_IN]

                    window = torch.cat([window[:, 1:], next_step.unsqueeze(1)], dim=1)

                loss = total_loss / (T_OUT * cfg.grad_accum)

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * cfg.grad_accum
            n_batches += 1

        # Handle remaining gradients
        if n_batches % cfg.grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        epoch_loss /= max(n_batches, 1)

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": scheduler.get_last_lr()[0],
                   "epoch_time_s": dt, "ss_prob": ss_prob, "global_step": global_step})

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
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB] ss={ss_prob:.2f}  "
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
