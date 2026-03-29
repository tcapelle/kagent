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
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model: Anchor-based Spatial MLP
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


class AnchorSpatialMLP(nn.Module):
    """Per-point MLP with spatial context from anchor points.

    Architecture:
    1. Sample K anchor points from each sample
    2. Compute per-point features at all N points and K anchors
    3. RBF-weighted aggregation: each point gets spatial context from anchors
    4. Concatenate per-point features + spatial context
    5. MLP → predict velocity delta
    6. Add to last input (residual)
    7. No-slip BC

    The anchor approach gives O(N*K) spatial interaction instead of O(N²).
    """

    def __init__(self, hidden=256, n_blocks=6, n_anchors=256, sigma=0.1):
        super().__init__()
        self.n_anchors = n_anchors
        self.sigma = sigma

        # Per-point features: pos(3) + velocity_in(5*3=15) + airfoil(1) = 19
        point_feat_dim = 3 + T_IN * 3 + 1

        # Anchor feature encoder: point features → anchor features
        self.anchor_encoder = nn.Sequential(
            nn.Linear(point_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Main MLP: per-point features(19) + spatial context from anchors(hidden) = 19+hidden
        in_dim = point_feat_dim + hidden
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        device = pos.device

        v_flat = velocity_in.reshape(B, N, T * C)  # [B, N, 15]

        # Airfoil indicator
        airfoil_mask = torch.zeros(B, N, 1, device=device)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                idx = idcs_airfoil[i].to(device)
                airfoil_mask[i, idx, 0] = 1.0

        # Per-point features
        point_feats = torch.cat([pos, v_flat, airfoil_mask], dim=-1)  # [B, N, 19]

        # Sample anchor points (use first K points — deterministic for reproducibility)
        K = min(self.n_anchors, N)
        # Use farthest point sampling approximation: random subset
        anchor_idx = torch.randint(N, (B, K), device=device)
        anchor_pos = torch.gather(pos, 1, anchor_idx.unsqueeze(-1).expand(-1, -1, 3))  # [B, K, 3]
        anchor_feats = torch.gather(point_feats, 1, anchor_idx.unsqueeze(-1).expand(-1, -1, point_feats.shape[-1]))  # [B, K, 19]

        # Encode anchor features
        anchor_encoded = self.anchor_encoder(anchor_feats)  # [B, K, hidden]

        # RBF attention: each point attends to anchors based on distance
        # Compute squared distances: [B, N, K]
        # Chunked to avoid OOM with N=100k, K=256
        spatial_context = torch.zeros(B, N, anchor_encoded.shape[-1], device=device)

        chunk_size = 20000
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            pos_chunk = pos[:, start:end, :]  # [B, chunk, 3]

            sq_dist = torch.cdist(pos_chunk, anchor_pos).pow(2)  # [B, chunk, K]
            weights = torch.exp(-sq_dist / (2 * self.sigma ** 2))  # [B, chunk, K]
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)  # normalize

            # Weighted aggregation of anchor features
            spatial_context[:, start:end, :] = torch.bmm(weights, anchor_encoded)  # [B, chunk, hidden]

        # Concatenate per-point features + spatial context
        x = torch.cat([point_feats, spatial_context], dim=-1)  # [B, N, 19+hidden]

        x = self.proj_in(x)
        x = self.blocks(x)
        delta = self.proj_out(x)  # [B, N, 15]
        delta = delta.reshape(B, T_OUT, N, 3)

        # Residual: add last input timestep
        pred = velocity_in[:, -1:, :, :] + delta

        # No-slip BC
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                idx = idcs_airfoil[i].to(pred.device)
                pred[i, :, idx, :] = 0.0

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
# Point subsampling
# ---------------------------------------------------------------------------

def subsample_batch(v_in, v_out, pos, idcs_airfoil, n_points=20000, surface_weight=3.0):
    B, T, N, C = v_in.shape
    device = v_in.device

    new_v_in, new_v_out, new_pos, new_idcs = [], [], [], []

    for i in range(B):
        weights = torch.ones(N, device=device)
        if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
            surface_idx = idcs_airfoil[i].to(device)
            weights[surface_idx] = surface_weight

        idx = torch.multinomial(weights, n_points, replacement=False)
        idx_sorted = idx.sort().values

        new_v_in.append(v_in[i, :, idx_sorted, :])
        new_v_out.append(v_out[i, :, idx_sorted, :])
        new_pos.append(pos[i, idx_sorted, :])

        if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[surface_idx] = True
            sub_mask = mask[idx_sorted]
            new_idcs.append(sub_mask.nonzero(as_tuple=True)[0].cpu())
        else:
            new_idcs.append(torch.tensor([], dtype=torch.long))

    return torch.stack(new_v_in), torch.stack(new_v_out), torch.stack(new_pos), new_idcs


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))  # minutes


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 2
    epochs: int = 200
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    hidden: int = 256
    n_blocks: int = 6
    n_anchors: int = 256
    sigma: float = 0.1
    n_subsample: int = 10000


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

    model = AnchorSpatialMLP(
        hidden=cfg.hidden, n_blocks=cfg.n_blocks,
        n_anchors=cfg.n_anchors, sigma=cfg.sigma,
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

            # Subsample for speed
            v_in_s, v_out_s, pos_s, idcs_s = subsample_batch(
                v_in, v_out, pos, idcs, n_points=cfg.n_subsample
            )

            with torch.amp.autocast("cuda"):
                pred = model(v_in_s, pos_s, t, idcs_s)
                loss = (pred - v_out_s).pow(2).mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

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
