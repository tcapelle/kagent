"""Train a 3D airflow velocity predictor.

Architecture: Residual MLP with EdgeConv (k-NN message passing) for spatial
interaction, Fourier position encoding, temporal derivatives, surface indicator,
no-slip enforcement, velocity normalization, AMP.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import math
import os
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
# Utilities
# ---------------------------------------------------------------------------

def chunked_knn(pos, k, chunk_size=5000):
    """Compute k-NN indices for point cloud. pos: [B, N, 3] -> [B, N, k]"""
    B, N, _ = pos.shape
    device = pos.device
    knn_idx = torch.empty(B, N, k, dtype=torch.long, device=device)
    pos_f = pos.float()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(pos_f[:, start:end], pos_f)  # [B, chunk, N]
        # Set self-distances to inf
        local_idx = torch.arange(end - start, device=device)
        dists[:, local_idx, local_idx + start] = float('inf')
        knn_idx[:, start:end] = dists.topk(k, dim=-1, largest=False).indices
    return knn_idx


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


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


class EdgeConvBlock(nn.Module):
    """DGCNN-style edge convolution for spatial interaction."""

    def __init__(self, dim, k=20):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos, knn_idx):
        """x: [B,N,D], pos: [B,N,3], knn_idx: [B,N,k]"""
        B, N, D = x.shape
        k = knn_idx.shape[-1]

        # Gather neighbor features and positions
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        x_j = x[batch_idx, knn_idx]          # [B, N, k, D]
        pos_j = pos[batch_idx, knn_idx]       # [B, N, k, 3]

        # Edge features: feature difference + relative position
        diff = x_j - x.unsqueeze(2)           # [B, N, k, D]
        rel_pos = pos_j - pos.unsqueeze(2)    # [B, N, k, 3]
        edge_feat = torch.cat([diff, rel_pos], dim=-1)  # [B, N, k, D+3]

        # Process edges and aggregate
        edge_out = self.edge_mlp(edge_feat)   # [B, N, k, D]
        agg = edge_out.mean(dim=2)            # [B, N, D]

        return self.norm(x + agg)


class AirflowPredictor(nn.Module):
    """Residual MLP + EdgeConv with physics-informed design and Fourier features."""

    def __init__(self, hidden=512, n_blocks=8, n_fourier=64, dropout=0.05,
                 k_neighbors=8, n_edge_conv=4, vel_mean=None, vel_std=None):
        super().__init__()
        self.n_fourier = n_fourier
        self.k_neighbors = k_neighbors

        # Temporal 1D conv for velocity feature extraction
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=T_IN, padding=0),  # pool T_IN -> 1
            nn.GELU(),
        )
        temporal_feat_dim = 32

        # Input features:
        # fourier_pos(384) + temporal_conv(32) + vel_in_flat(15) + vel_mag(5)
        # + vel_derivatives(12) + time(5) + surface_flag(1) = 454
        pos_feat_dim = n_fourier * 2 * 3
        vel_feat_dim = T_IN * 3
        vel_mag_dim = T_IN  # velocity magnitude per timestep
        deriv_feat_dim = (T_IN - 1) * 3
        time_feat_dim = T_IN
        surface_dim = 1
        in_dim = pos_feat_dim + temporal_feat_dim + vel_feat_dim + vel_mag_dim + deriv_feat_dim + time_feat_dim + surface_dim

        out_dim = T_OUT * 3

        # Learnable Fourier feature frequencies (multi-scale initialization)
        freqs = torch.cat([
            torch.randn(3, n_fourier // 4) * 0.5,   # low frequency
            torch.randn(3, n_fourier // 2) * 2.0,    # medium frequency
            torch.randn(3, n_fourier - n_fourier // 4 - n_fourier // 2) * 8.0,  # high frequency
        ], dim=1)
        self.fourier_freqs = nn.Parameter(freqs)

        self.proj_in = nn.Linear(in_dim, hidden)

        # Interleave ResBlocks and EdgeConv layers
        # n_edge_conv EdgeConv layers split ResBlocks into (n_edge_conv+1) groups
        n_groups = n_edge_conv + 1
        blocks_per_group = n_blocks // n_groups
        remainder = n_blocks % n_groups

        self.res_groups = nn.ModuleList()
        self.edge_convs = nn.ModuleList()
        for i in range(n_groups):
            n = blocks_per_group + (1 if i < remainder else 0)
            self.res_groups.append(nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n)]))
            if i < n_edge_conv:
                self.edge_convs.append(EdgeConvBlock(hidden, k=k_neighbors))

        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

    def _fourier_encode(self, pos):
        """Random Fourier features for position. pos: [B, N, 3]"""
        B, N, _ = pos.shape
        feats = []
        for i in range(3):
            p = pos[:, :, i:i+1] * self.fourier_freqs[i:i+1, :]
            feats.append(torch.sin(p))
            feats.append(torch.cos(p))
        return torch.cat(feats, dim=-1)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocities
        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std

        # Fourier position encoding
        pos_feat = self._fourier_encode(pos)

        # Temporal conv features: [B, T, N, 3] -> [B*N, 3, T] -> [B*N, 32, 1] -> [B, N, 32]
        v_temp = v_in_norm.permute(0, 2, 3, 1).reshape(B * N, C, T)  # [B*N, 3, 5]
        v_temp = self.temporal_conv(v_temp).squeeze(-1)  # [B*N, 32]
        v_temp = v_temp.reshape(B, N, -1)  # [B, N, 32]

        # Flattened velocity features (keep for explicit access)
        v_flat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)

        # Velocity magnitude per timestep
        v_mag = v_in_norm.norm(dim=3)  # [B, T, N]
        v_mag = v_mag.permute(0, 2, 1)  # [B, N, T]

        # Temporal derivatives
        v_deriv = v_in_norm[:, 1:] - v_in_norm[:, :-1]
        v_deriv_flat = v_deriv.permute(0, 2, 1, 3).reshape(B, N, (T-1) * C)

        # Time features
        t_in = t[:, :T_IN]
        t_range = t_in[:, -1:] - t_in[:, :1] + 1e-6
        t_features = (t_in - t_in[:, :1]) / t_range
        t_features = t_features.unsqueeze(1).expand(B, N, T_IN)

        # Surface indicator
        surface_flag = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                surface_flag[i, idcs_airfoil[i], 0] = 1.0

        # Concatenate all features
        x = torch.cat([pos_feat, v_temp, v_flat, v_mag, v_deriv_flat, t_features, surface_flag], dim=-1)

        # Compute k-NN graph (outside autocast for float32 precision)
        with torch.amp.autocast("cuda", enabled=False):
            knn_idx = chunked_knn(pos.float(), self.k_neighbors)

        x = self.proj_in(x)
        for i, res_group in enumerate(self.res_groups):
            x = res_group(x)
            if i < len(self.edge_convs):
                x = self.edge_convs[i](x, pos, knn_idx)
        delta_norm = self.proj_out(x)
        delta_norm = delta_norm.reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)

        # Residual prediction from last input timestep
        last_in_norm = v_in_norm[:, -1:, :, :]
        pred_norm = last_in_norm + delta_norm

        # Denormalize
        pred = pred_norm * self.vel_std + self.vel_mean

        # No-slip enforcement
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                pred[i, :, idcs_airfoil[i], :] = 0.0

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
# Config + data loading
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 8e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 80
    subsample_points: int = 15000
    hidden: int = 512
    n_blocks: int = 8
    n_fourier: int = 64
    k_neighbors: int = 6
    n_edge_conv: int = 6
    dropout: float = 0.05
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
    name: DataLoader(ds, batch_size=1, shuffle=False, **loader_kwargs)
    for name, ds in val_splits.items()
}

model = AirflowPredictor(
    hidden=cfg.hidden, n_blocks=cfg.n_blocks, n_fourier=cfg.n_fourier,
    dropout=cfg.dropout, k_neighbors=cfg.k_neighbors, n_edge_conv=cfg.n_edge_conv,
    vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

# OneCycleLR for faster convergence
# Estimate actual epochs we'll complete based on timeout
# Each epoch takes ~30-40s train + occasional validation
estimated_epochs = min(MAX_EPOCHS, int(MAX_TIMEOUT * 60 / 35))  # conservative
steps_per_epoch = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=cfg.lr, epochs=estimated_epochs, steps_per_epoch=steps_per_epoch,
    pct_start=0.1, anneal_strategy='cos',
)
scaler = torch.amp.GradScaler("cuda")

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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

        B = v_in.shape[0]
        N = v_in.shape[2]

        # Subsample points during training
        if cfg.subsample_points and cfg.subsample_points < N:
            sub_idx = torch.randperm(N, device=device)[:cfg.subsample_points]
            sub_idx = sub_idx.sort().values
            v_in_sub = v_in[:, :, sub_idx, :]
            v_out_sub = v_out[:, :, sub_idx, :]
            pos_sub = pos[:, sub_idx, :]

            sub_set = set(sub_idx.cpu().tolist())
            idx_map = {old: new for new, old in enumerate(sub_idx.cpu().tolist())}
            idcs_sub = []
            for i in range(B):
                if idcs[i] is not None:
                    airfoil_set = set(idcs[i].tolist())
                    kept = airfoil_set & sub_set
                    idcs_sub.append(torch.tensor([idx_map[k] for k in kept], dtype=torch.long))
                else:
                    idcs_sub.append(torch.tensor([], dtype=torch.long))
        else:
            v_in_sub, v_out_sub, pos_sub, idcs_sub = v_in, v_out, pos, idcs

        with torch.amp.autocast("cuda"):
            pred = model(v_in_sub, pos_sub, t, idcs_sub)
            # Two-phase loss: MSE early for fast convergence, L2 later for metric
            elapsed_frac = (time.time() - train_start) / (MAX_TIMEOUT * 60)
            if elapsed_frac < 0.4:
                loss = (pred - v_out_sub).pow(2).mean()
            else:
                loss = (pred - v_out_sub).norm(dim=3).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1
        wandb.log({"train/loss": loss.item(), "global_step": global_step})

        epoch_loss += loss.item()
        n_batches += 1
    epoch_loss /= n_batches
    dt = time.time() - t0

    # Validate less frequently early on to save time
    elapsed_frac = (time.time() - train_start) / (MAX_TIMEOUT * 60)
    do_val = (elapsed_frac > 0.6) or (epoch % 5 == 0) or (epoch < 2)

    if do_val:
        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt_total = time.time() - t0
    else:
        mean_val = best_val + 1  # skip
        dt_total = dt

    wandb.log({"train/epoch_loss": epoch_loss, "lr": optimizer.param_groups[0]['lr'],
               "epoch_time_s": dt_total, "global_step": global_step})

    tag = ""
    if do_val and mean_val < best_val:
        best_val = mean_val
        best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
        for sm in split_metrics.values():
            best_metrics.update({f"best_{k}": v for k, v in sm.items()})
        torch.save(model.state_dict(), model_path)
        tag = " *"

    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    val_str = f"val/l2={mean_val:.4f}" if do_val else "val/l2=skip"
    print(
        f"Epoch {epoch+1:3d} ({dt_total:.0f}s) [{peak_gb:.1f}GB]  "
        f"train={epoch_loss:.4f}  {val_str}{tag}"
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
