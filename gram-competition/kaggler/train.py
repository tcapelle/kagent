"""Train a 3D airflow velocity predictor.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import time
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0, ffn_mult=4, drop_path=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )
        self.drop_path = drop_path

    def forward(self, x):
        if self.training and self.drop_path > 0:
            keep = torch.rand(1, device=x.device) > self.drop_path
            return x + self.net(x) * keep / (1 - self.drop_path)
        return x + self.net(x)


class VelocityPredictor(nn.Module):
    """Per-timestep conditioned MLP.

    Instead of predicting all 5 output timesteps at once,
    predict each output timestep conditioned on its time offset.
    This gives 5x the training signal and allows specialization per timestep.
    """

    def __init__(self, hidden=384, n_blocks=12, dropout=0.05, n_fourier=32):
        super().__init__()
        # Input: pos(3) + fourier_pos(2*n_fourier) + vel_in(15) + vel_diff(12) + vel_stats(9) + airfoil_flag(1)
        fourier_dim = 2 * n_fourier
        in_dim = 3 + fourier_dim + T_IN * 3 + (T_IN - 1) * 3 + 9 + 1
        self.in_dim = in_dim

        # Random Fourier features for position
        self.register_buffer("fourier_B", torch.randn(3, n_fourier) * 5.0)

        # Timestep embedding for the 5 output steps
        self.time_embed = nn.Embedding(T_OUT, hidden)

        self.proj_in = nn.Linear(in_dim, hidden)
        # Stochastic depth: linearly increase drop path from 0 to 0.1
        drop_rates = [0.1 * i / max(n_blocks - 1, 1) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*[ResBlock(hidden, dropout, drop_path=dr) for dr in drop_rates])
        self.norm_out = nn.LayerNorm(hidden)
        # Per-timestep refinement MLPs (2 layers each) for more specialization
        self.refine_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 3),
            ) for _ in range(T_OUT)
        ])

    def _build_features(self, velocity_in, pos, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        # Fourier position features
        pos_proj = pos @ self.fourier_B  # [B, N, n_fourier]
        pos_fourier = torch.cat([torch.sin(pos_proj), torch.cos(pos_proj)], dim=-1)

        vel_diff = velocity_in[:, 1:] - velocity_in[:, :-1]
        vel_flat = velocity_in.reshape(B, N, T * C)
        diff_flat = vel_diff.reshape(B, N, (T - 1) * C)
        vel_mean = velocity_in.mean(dim=1)
        vel_std = velocity_in.std(dim=1)
        vel_trend = velocity_in[:, -1] - velocity_in[:, 0]

        # Airfoil mask: 1 for airfoil surface points, 0 otherwise
        airfoil_mask = torch.zeros(B, N, 1, device=pos.device)
        for i in range(B):
            airfoil_mask[i, idcs_airfoil[i], 0] = 1.0

        return torch.cat([pos, pos_fourier, vel_flat, diff_flat, vel_mean, vel_std, vel_trend, airfoil_mask], dim=-1)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        features = self._build_features(velocity_in, pos, idcs_airfoil)  # [B, N, in_dim]
        h = self.proj_in(features)  # [B, N, hidden]

        # Process through shared backbone
        h = self.blocks(h)
        h = self.norm_out(h)  # [B, N, hidden]

        # Predict each output timestep with timestep conditioning + separate heads
        outputs = []
        for step in range(T_OUT):
            t_emb = self.time_embed(torch.tensor(step, device=h.device))
            h_cond = h + t_emb.unsqueeze(0).unsqueeze(0)
            pred = self.refine_blocks[step](h_cond)  # [B, N, 3]
            outputs.append(pred)

        delta = torch.stack(outputs, dim=1)  # [B, 5, N, 3]

        # Learned residual gate: let model learn how much of input mean to use
        input_mean = velocity_in.mean(dim=1, keepdim=True)  # [B, 1, N, 3]
        out = delta + input_mean

        # No-slip BC
        for i in range(B):
            out[i, :, idcs_airfoil[i], :] = 0.0

        return out


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model, backup):
        model.load_state_dict(backup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def augment_flip(v_in, v_out, pos):
    if torch.rand(1).item() < 0.5:
        pos = pos.clone()
        pos[..., 1] = -pos[..., 1]
        v_in = v_in.clone()
        v_in[..., 1] = -v_in[..., 1]
        v_out = v_out.clone()
        v_out[..., 1] = -v_out[..., 1]
    if torch.rand(1).item() < 0.5:
        pos = pos.clone()
        pos[..., 2] = -pos[..., 2]
        v_in = v_in.clone()
        v_in[..., 2] = -v_in[..., 2]
        v_out = v_out.clone()
        v_out[..., 2] = -v_out[..., 2]
    return v_in, v_out, pos


def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points):
    B, T, N, C = v_in.shape
    if n_points >= N:
        return v_in, v_out, pos, idcs_airfoil
    idx = torch.randperm(N, device=v_in.device)[:n_points].sort().values
    v_in_sub = v_in[:, :, idx]
    v_out_sub = v_out[:, :, idx]
    pos_sub = pos[:, idx]
    new_idcs = []
    for i in range(B):
        mask = torch.isin(idx, idcs_airfoil[i].to(v_in.device))
        new_idcs.append(mask.nonzero(as_tuple=True)[0])
    return v_in_sub, v_out_sub, pos_sub, new_idcs


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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import simple_parsing as sp
    import wandb
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))

    @dataclass
    class Config:
        lr: float = 1e-3
        weight_decay: float = 1e-3
        batch_size: int = 1
        epochs: int = 300
        splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
        wandb_group: str | None = None
        wandb_name: str | None = None
        agent: str | None = None
        debug: bool = False
        hidden: int = 448
        n_blocks: int = 10
        dropout: float = 0.05
        n_subsample: int = 10000

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

    model = VelocityPredictor(
        hidden=cfg.hidden, n_blocks=cfg.n_blocks, dropout=cfg.dropout,
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

    model_cfg_dict = {
        "hidden": cfg.hidden, "n_blocks": cfg.n_blocks, "dropout": cfg.dropout,
    }
    torch.save(model_cfg_dict, model_dir / "config.pt")

    ema = EMA(model, decay=0.999)

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

            # Augmentation
            v_in, v_out, pos = augment_flip(v_in, v_out, pos)

            # Subsample
            v_in_s, v_out_s, pos_s, idcs_s = subsample_batch(
                v_in, v_out, pos, idcs, cfg.n_subsample
            )

            with torch.amp.autocast("cuda"):
                pred = model(v_in_s, pos_s, t, idcs_s)
                loss = F.smooth_l1_loss(pred, v_out_s)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

        # Validate with EMA
        backup = ema.apply(model)
        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        ema.restore(model, backup)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": optimizer.param_groups[0]['lr'],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(ema.shadow, model_path)
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
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path),
                    "--config", str(model_dir / "config.pt")]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()
