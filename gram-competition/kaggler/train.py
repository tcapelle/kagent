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
# Multi-resolution GNN: subsample → GNN → interpolate back
# ---------------------------------------------------------------------------

class MessagePassLayer(nn.Module):
    """Simple message passing on k-NN graph using pure PyTorch."""
    def __init__(self, dim, k=16):
        super().__init__()
        self.k = k
        self.msg_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, pos, knn_idx):
        """x: [N, D], pos: [N, 3], knn_idx: [N, k]"""
        N, D = x.shape
        k = knn_idx.shape[1]

        # Gather neighbor features
        x_j = x[knn_idx.reshape(-1)].reshape(N, k, D)  # [N, k, D]
        pos_j = pos[knn_idx.reshape(-1)].reshape(N, k, 3)
        x_i = x.unsqueeze(1).expand(-1, k, -1)  # [N, k, D]
        rel_pos = pos_j - pos.unsqueeze(1)  # [N, k, 3]

        # Message
        msg_input = torch.cat([x_i, x_j, rel_pos], dim=-1)  # [N, k, 2D+3]
        msgs = self.msg_mlp(msg_input)  # [N, k, D]
        agg = msgs.mean(dim=1)  # [N, D]

        # Update
        out = self.update_mlp(torch.cat([x, agg], dim=-1))
        return x + out


class MultiResGNN(nn.Module):
    """Multi-resolution GNN that operates on a subsampled point set."""
    def __init__(self, dim, n_layers=3, k=16):
        super().__init__()
        self.k = k
        self.layers = nn.ModuleList([MessagePassLayer(dim, k) for _ in range(n_layers)])

    def build_knn(self, pos):
        """Build k-NN graph using cdist (feasible for <10k points)."""
        dist = torch.cdist(pos, pos)  # [N, N]
        _, knn_idx = dist.topk(self.k, dim=1, largest=False)  # [N, k]
        return knn_idx

    def forward(self, x, pos):
        """x: [N, D], pos: [N, 3]"""
        knn_idx = self.build_knn(pos)
        for layer in self.layers:
            x = layer(x, pos, knn_idx)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class VelocityPredictor(nn.Module):
    """Multi-resolution model: pointwise encoder → subsample → GNN → upsample → refine.

    1. Encode per-point features with MLP
    2. Random subsample to n_sub points
    3. Apply GNN message passing on subsampled points
    4. Interpolate GNN features back to all points (inverse distance weighting)
    5. Combine with pointwise features and predict
    """

    def __init__(self, hidden=256, n_point_blocks=4, n_gnn_layers=4,
                 n_refine_blocks=3, n_sub=5000, k=16, dropout=0.05):
        super().__init__()
        self.n_sub = n_sub
        in_dim = 3 + T_IN * 3 + (T_IN - 1) * 3 + 9  # 39
        out_dim = T_OUT * 3  # 15

        # Pointwise encoder
        self.proj_in = nn.Linear(in_dim, hidden)
        self.point_blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_point_blocks)])

        # GNN on subsampled points
        self.gnn = MultiResGNN(hidden, n_layers=n_gnn_layers, k=k)

        # Refinement after upsampling
        self.refine = nn.Sequential(
            nn.Linear(hidden * 2, hidden),  # concat pointwise + GNN features
            *[ResBlock(hidden, dropout) for _ in range(n_refine_blocks)],
        )

        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)

    def _subsample_and_upsample(self, point_feat, pos, n_sub):
        """Subsample, apply GNN, interpolate back.
        point_feat: [B, N, D], pos: [B, N, 3]
        Returns: gnn_feat [B, N, D]
        """
        B, N, D = point_feat.shape
        n_sub = min(n_sub, N)

        all_gnn_feat = []
        for b in range(B):
            # Random subsample
            idx = torch.randperm(N, device=pos.device)[:n_sub]
            sub_feat = point_feat[b, idx]  # [n_sub, D]
            sub_pos = pos[b, idx]  # [n_sub, 3]

            # GNN on subsampled points
            sub_feat = self.gnn(sub_feat, sub_pos)  # [n_sub, D]

            # Interpolate back to all points using k=3 nearest neighbors + IDW
            dist = torch.cdist(pos[b], sub_pos)  # [N, n_sub]
            _, nn_idx = dist.topk(3, dim=1, largest=False)  # [N, 3]
            nn_dist = torch.gather(dist, 1, nn_idx)  # [N, 3]

            # Inverse distance weights
            w = 1.0 / (nn_dist + 1e-6)  # [N, 3]
            w = w / w.sum(dim=1, keepdim=True)  # [N, 3]

            # Weighted sum of neighbor features
            nn_feat = sub_feat[nn_idx.reshape(-1)].reshape(N, 3, D)  # [N, 3, D]
            gnn_feat = (nn_feat * w.unsqueeze(-1)).sum(dim=1)  # [N, D]
            all_gnn_feat.append(gnn_feat)

        return torch.stack(all_gnn_feat)  # [B, N, D]

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        last_vel = velocity_in[:, -1]

        vel_diff = velocity_in[:, 1:] - velocity_in[:, :-1]
        vel_flat = velocity_in.reshape(B, N, T * C)
        diff_flat = vel_diff.reshape(B, N, (T - 1) * C)
        vel_mean = velocity_in.mean(dim=1)
        vel_std = velocity_in.std(dim=1)
        vel_trend = velocity_in[:, -1] - velocity_in[:, 0]

        x = torch.cat([pos, vel_flat, diff_flat, vel_mean, vel_std, vel_trend], dim=-1)

        # Pointwise encoding
        point_feat = self.proj_in(x)
        point_feat = self.point_blocks(point_feat)  # [B, N, hidden]

        # Multi-resolution GNN
        gnn_feat = self._subsample_and_upsample(point_feat, pos, self.n_sub)

        # Combine and refine
        combined = torch.cat([point_feat, gnn_feat], dim=-1)  # [B, N, 2*hidden]
        x = self.refine(combined)

        delta = self.proj_out(self.norm_out(x)).reshape(B, T_OUT, N, 3)
        out = delta + last_vel.unsqueeze(1)

        # No-slip BC
        for i in range(B):
            out[i, :, idcs_airfoil[i], :] = 0.0

        return out


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
        lr: float = 5e-4
        weight_decay: float = 1e-4
        batch_size: int = 1
        epochs: int = 100
        splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
        wandb_group: str | None = None
        wandb_name: str | None = None
        agent: str | None = None
        debug: bool = False
        hidden: int = 256
        n_point_blocks: int = 4
        n_gnn_layers: int = 4
        n_refine_blocks: int = 3
        n_sub: int = 5000
        k: int = 16
        dropout: float = 0.05

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
        hidden=cfg.hidden, n_point_blocks=cfg.n_point_blocks,
        n_gnn_layers=cfg.n_gnn_layers, n_refine_blocks=cfg.n_refine_blocks,
        n_sub=cfg.n_sub, k=cfg.k, dropout=cfg.dropout,
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
        "hidden": cfg.hidden, "n_point_blocks": cfg.n_point_blocks,
        "n_gnn_layers": cfg.n_gnn_layers, "n_refine_blocks": cfg.n_refine_blocks,
        "n_sub": cfg.n_sub, "k": cfg.k, "dropout": cfg.dropout,
    }
    torch.save(model_cfg_dict, model_dir / "config.pt")

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

            with torch.amp.autocast("cuda"):
                pred = model(v_in, pos, t, idcs)
                loss = (pred - v_out).pow(2).mean()

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
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path),
                    "--config", str(model_dir / "config.pt")]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()
