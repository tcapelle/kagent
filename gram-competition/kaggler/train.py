"""Train a 3D airflow velocity predictor.

Residual MLP with no-slip BC, Fourier position features, and per-sample
normalization. Predicts delta from the last input timestep.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import math
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def fourier_features(x: torch.Tensor, num_freqs: int, scale: float = 1.0) -> torch.Tensor:
    """Apply log-spaced sin/cos Fourier features to each channel of x.

    x: [..., C] -> [..., C * (1 + 2*num_freqs)]  (original + sin + cos per freq)
    """
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi * scale
    xf = x.unsqueeze(-1) * freqs  # [..., C, F]
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1)  # [..., C, 2F]
    enc = enc.reshape(*x.shape[:-1], -1)  # [..., C*2F]
    return torch.cat([x, enc], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Point-wise residual MLP with Fourier pos features.

    Predicts a delta for each of the T_OUT future timesteps, added to the
    last input timestep velocity. Output at airfoil indices is hard-zeroed.
    """

    def __init__(self, hidden=512, n_blocks=8, num_pos_freqs=10, num_vel_freqs=4,
                 dropout=0.0, vel_mean=None, vel_std=None):
        super().__init__()
        # Features per point:
        #   pos Fourier (3 * (1 + 2*F_p))
        #   normalized v_in flattened (T_IN * 3)
        #   normalized v_in Fourier on last timestep (3 * 2*F_v)
        #   airfoil indicator (1)
        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vin_dim = T_IN * 3
        vin_fourier_dim = 3 * 2 * num_vel_freqs
        in_dim = pos_dim + vin_dim + vin_fourier_dim + 1

        out_dim = T_OUT * 3

        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(
            *[ResBlock(hidden, mult=2, dropout=dropout) for _ in range(n_blocks)]
        )
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # Zero-init last layer so the model starts as identity (residual = 0).
        nn.init.zeros_(self.proj_out[1].weight)
        nn.init.zeros_(self.proj_out[1].bias)

        # Register velocity normalization stats.
        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        """
        velocity_in: [B, 5, N, 3]
        pos:         [B, N, 3]
        idcs_airfoil: list[tensor] length B — surface indices
        Returns:     [B, 5, N, 3]  (future velocity field)
        """
        B, T, N, C = velocity_in.shape
        last_v = velocity_in[:, -1]  # [B, N, 3]

        # Normalize velocity.
        v_norm = (velocity_in - self.vel_mean) / self.vel_std  # [B, 5, N, 3]

        # Build per-point features.
        pos_feat = fourier_features(pos, self.num_pos_freqs, scale=1.0)  # [B, N, 3*(1+2F)]
        vin_flat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T_IN * 3)  # [B, N, 15]
        # Fourier on last-norm velocity (without identity term to avoid double count).
        last_norm = v_norm[:, -1]  # [B, N, 3]
        freqs = 2.0 ** torch.arange(
            self.num_vel_freqs, device=pos.device, dtype=pos.dtype
        ) * math.pi
        vf = last_norm.unsqueeze(-1) * freqs  # [B, N, 3, F]
        vin_fourier = torch.cat([vf.sin(), vf.cos()], dim=-1).reshape(B, N, -1)

        airfoil_ind = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            airfoil_ind[b, idx.to(pos.device).long(), 0] = 1.0

        x = torch.cat([pos_feat, vin_flat, vin_fourier, airfoil_ind], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)  # [B, T_OUT, N, 3]

        # Denormalize delta using only std (mean-free delta).
        delta = delta_norm * self.vel_std

        # Residual: add last input timestep as prior.
        pred = last_v.unsqueeze(1) + delta  # [B, T_OUT, N, 3]

        # No-slip BC: zero velocity on airfoil surface.
        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx.to(pred.device).long(), :] = 0.0

        return pred


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

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
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
# Config + data loading
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1
    epochs: int = 80
    # Training-time point subsampling for speed; full 100k at eval.
    subsample_points: int = 20000
    hidden: int = 512
    n_blocks: int = 8
    num_pos_freqs: int = 10
    num_vel_freqs: int = 4
    dropout: float = 0.0
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


def subsample_batch(v_in, v_out, pos, idcs_airfoil, k):
    """Randomly sample k points per sample. Rebuild airfoil indices in-subsample."""
    B, T, N, C = v_in.shape
    if k is None or k >= N:
        return v_in, v_out, pos, idcs_airfoil
    new_idcs = []
    perm = torch.stack([torch.randperm(N, device=v_in.device)[:k] for _ in range(B)])
    v_in_s = torch.gather(v_in, 2, perm[:, None, :, None].expand(-1, T, -1, C))
    v_out_s = torch.gather(v_out, 2, perm[:, None, :, None].expand(-1, T_OUT, -1, C))
    pos_s = torch.gather(pos, 1, perm[:, :, None].expand(-1, -1, C))
    for b in range(B):
        af = idcs_airfoil[b].to(v_in.device).long()
        mask_full = torch.zeros(N, dtype=torch.bool, device=v_in.device)
        mask_full[af] = True
        mask_sub = mask_full[perm[b]]
        new_idcs.append(mask_sub.nonzero(as_tuple=False).squeeze(-1))
    return v_in_s, v_out_s, pos_s, new_idcs


def main():
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

    model = ResidualMLP(
        hidden=cfg.hidden,
        n_blocks=cfg.n_blocks,
        num_pos_freqs=cfg.num_pos_freqs,
        num_vel_freqs=cfg.num_vel_freqs,
        dropout=cfg.dropout,
        vel_mean=stats["vel_mean"],
        vel_std=stats["vel_std"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

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

    KAGGLER_NAME = os.environ.get("KAGGLER_NAME", cfg.agent or "local")
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}/{KAGGLER_NAME}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    model_path = pvc_dir / "checkpoint.pt"

    git_ckpt_path = Path("checkpoints/best.pt")
    git_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

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

            v_in_s, v_out_s, pos_s, idcs_s = subsample_batch(
                v_in, v_out, pos, idcs, cfg.subsample_points
            )

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(v_in_s, pos_s, t, idcs_s)
                loss = (pred.float() - v_out_s).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
            shutil.copyfile(model_path, git_ckpt_path)
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


if __name__ == "__main__":
    main()
