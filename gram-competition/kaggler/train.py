"""Train a 3D airflow velocity predictor.

Template — fill in your model architecture.
The training loop, loss, validation, and W&B logging are provided.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import shutil
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
# Baseline MLP — replace with your own architecture
#
# Model contract:
#   Input:  velocity_in [B, 5, N, 3], pos [B, N, 3], t [B, 10], idcs_airfoil list[tensor]
#   Output: velocity_out [B, 5, N, 3]  (predicted future velocity field)
#
# Note: the real competition uses model(t, pos, idcs_airfoil, velocity_in) —
#       different arg order. If you submit to the real comp, wrap accordingly.
# ---------------------------------------------------------------------------


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


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        xn = self.norm1(x)
        a, _ = self.attn(xn, xn, xn, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


def knn_graph(pos, k, chunk=4096):
    """KNN indices from positions. pos: [B, N, 3]. Returns [B, N, k] (excludes self)."""
    B, N, _ = pos.shape
    out = torch.empty(B, N, k, dtype=torch.long, device=pos.device)
    for b in range(B):
        pb = pos[b]
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(pb[s:e].unsqueeze(0), pb.unsqueeze(0)).squeeze(0)  # [chunk, N]
            _, idx = d.topk(k + 1, dim=-1, largest=False)  # +1 because self is included
            out[b, s:e] = idx[:, 1:k+1]  # drop self
    return out


class EdgeConvBlock(nn.Module):
    """DGCNN-style EdgeConv: for each edge (i,j), MLP([x_i, x_j - x_i]) then max-pool over j."""
    def __init__(self, dim, mlp_ratio=1.0):
        super().__init__()
        h = int(dim * mlp_ratio * 2)
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, h), nn.GELU(), nn.Linear(h, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, knn_idx):
        # x: [B, N, D], knn_idx: [B, N, k]
        B, N, D = x.shape
        k = knn_idx.shape[-1]
        flat_idx = knn_idx.reshape(B, N * k, 1).expand(-1, -1, D)
        neighbors = x.gather(dim=1, index=flat_idx).reshape(B, N, k, D)  # [B, N, k, D]
        xi = x.unsqueeze(2).expand(-1, -1, k, -1)                        # [B, N, k, D]
        edge_feat = torch.cat([xi, neighbors - xi], dim=-1)              # [B, N, k, 2D]
        msg = self.mlp(edge_feat)                                        # [B, N, k, D]
        agg = msg.max(dim=2).values                                      # [B, N, D]
        return self.norm(x + agg)


def knn_interpolate(pos_full, pos_anchor, feat_anchor, k=3, chunk=8192):
    """For each point in pos_full, gather k nearest anchors' features (inverse-dist weighted).

    pos_full:    [B, N, 3]
    pos_anchor:  [B, K, 3]
    feat_anchor: [B, K, D]
    Returns:     [B, N, D]
    """
    B, N, _ = pos_full.shape
    _, K, D = feat_anchor.shape
    out = torch.empty(B, N, D, device=feat_anchor.device, dtype=feat_anchor.dtype)
    for b in range(B):
        pf = pos_full[b]      # [N, 3]
        pa = pos_anchor[b]    # [K, 3]
        fa = feat_anchor[b]   # [K, D]
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            dist = torch.cdist(pf[s:e].unsqueeze(0), pa.unsqueeze(0)).squeeze(0)  # [chunk, K]
            topk_d, topk_i = dist.topk(k, dim=-1, largest=False)  # [chunk, k]
            w = 1.0 / (topk_d + 1e-6)
            w = w / w.sum(dim=-1, keepdim=True)
            gathered = fa[topk_i]  # [chunk, k, D]
            out[b, s:e] = (gathered * w.unsqueeze(-1)).sum(dim=1)
    return out


class FourierEmbed(nn.Module):
    """Multi-resolution sinusoidal encoding for positions. pos ∈ [-1, 1]^3 → [B, N, 6*n_freqs]."""
    def __init__(self, n_freqs=8, base=2.0):
        super().__init__()
        freqs = (base ** torch.arange(n_freqs).float()) * torch.pi
        self.register_buffer("freqs", freqs.view(1, 1, 1, -1))  # [1, 1, 1, n_freqs]
        self.n_freqs = n_freqs
        self.out_dim = 3 * 2 * n_freqs

    def forward(self, pos):  # [B, N, 3] in [-1, 1]
        proj = pos.unsqueeze(-1) * self.freqs  # [B, N, 3, n_freqs]
        return torch.cat([proj.sin(), proj.cos()], dim=-1).flatten(-2)  # [B, N, 6*n_freqs]


class BaselineMLP(nn.Module):
    """Baseline MLP + EdgeConv GNN refinement branch (zero-init).

    Point branch: per-point MLP (baseline behavior).
    Spatial branch: subsample K anchors, run EdgeConv GNN with KNN graph, zero-init
    output head and KNN-interpolate back to full 100k. At init, total = point_pred.
    """

    def __init__(self, hidden=256, n_blocks=6, edge_blocks=4, edge_k=16,
                 n_anchors=8192, coarse_blocks=3, coarse_k=32, n_anchors_coarse=2048,
                 fourier_freqs=8, vel_mean=None, vel_std=None):
        super().__init__()
        self.n_anchors = n_anchors
        self.n_anchors_coarse = n_anchors_coarse
        self.edge_k = edge_k
        self.coarse_k = coarse_k
        self.fourier = FourierEmbed(n_freqs=fourier_freqs)
        vel_diff_dim = (T_IN - 1) * 3
        vel_mag_dim = T_IN
        in_dim = 3 + self.fourier.out_dim + T_IN * 3 + vel_diff_dim + vel_mag_dim
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.point_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.point_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # Fine-scale EdgeConv (local detail)
        self.anchor_proj = nn.Linear(hidden, hidden)
        self.edge_blocks = nn.ModuleList([EdgeConvBlock(hidden) for _ in range(edge_blocks)])
        self.anchor_norm = nn.LayerNorm(hidden)
        self.spatial_head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)

        # Coarse-scale EdgeConv (global reach)
        self.coarse_proj = nn.Linear(hidden, hidden)
        self.coarse_blocks_list = nn.ModuleList([EdgeConvBlock(hidden) for _ in range(coarse_blocks)])
        self.coarse_norm = nn.LayerNorm(hidden)
        self.coarse_head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_flat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)

        # Temporal velocity diffs and per-frame magnitudes (normalized space)
        v_pt = v_norm.permute(0, 2, 1, 3)                     # [B, N, T, 3]
        v_diff = (v_pt[:, :, 1:] - v_pt[:, :, :-1]).reshape(B, N, (T - 1) * 3)
        v_mag = v_pt.norm(dim=-1)                             # [B, N, T]

        pos_min = pos.amin(dim=1, keepdim=True)
        pos_max = pos.amax(dim=1, keepdim=True)
        pos_n = (pos - pos_min) / (pos_max - pos_min + 1e-6) * 2 - 1

        pos_feat = self.fourier(pos_n)             # [B, N, 6*n_freqs]
        x_in = torch.cat([pos_n, pos_feat, v_flat, v_diff, v_mag], dim=-1)
        x = self.proj_in(x_in)                     # [B, N, D]
        x = self.point_blocks(x)                   # [B, N, D]
        point_pred = self.point_head(x)            # [B, N, out_dim]

        # Fine-scale EdgeConv: subsample anchors, build KNN graph, EdgeConv, interpolate back
        K = min(self.n_anchors, N)
        if self.training:
            idx = torch.randperm(N, device=x.device)[:K]
        else:
            stride = max(N // K, 1)
            idx = torch.arange(0, stride * K, stride, device=x.device)[:K]
        anchor_x = self.anchor_proj(x[:, idx])
        anchor_pos = pos_n[:, idx]
        knn_idx = knn_graph(anchor_pos, self.edge_k)
        for block in self.edge_blocks:
            anchor_x = block(anchor_x, knn_idx)
        anchor_x = self.anchor_norm(anchor_x)
        interp_fine = knn_interpolate(pos_n, anchor_pos, anchor_x, k=3)
        spatial_pred = self.spatial_head(interp_fine)

        # Coarse-scale EdgeConv: fewer anchors, larger k → global receptive field
        Kc = min(self.n_anchors_coarse, N)
        if self.training:
            idx_c = torch.randperm(N, device=x.device)[:Kc]
        else:
            stride_c = max(N // Kc, 1)
            idx_c = torch.arange(0, stride_c * Kc, stride_c, device=x.device)[:Kc]
        coarse_x = self.coarse_proj(x[:, idx_c])
        coarse_pos = pos_n[:, idx_c]
        coarse_knn = knn_graph(coarse_pos, self.coarse_k)
        for block in self.coarse_blocks_list:
            coarse_x = block(coarse_x, coarse_knn)
        coarse_x = self.coarse_norm(coarse_x)
        interp_coarse = knn_interpolate(pos_n, coarse_pos, coarse_x, k=3)
        coarse_pred = self.coarse_head(interp_coarse)

        out_combined = (point_pred + spatial_pred + coarse_pred).reshape(B, N, T_OUT, 3)
        out_norm = out_combined.permute(0, 2, 1, 3)  # [B, T_OUT, N, 3]
        out = out_norm * self.vel_std + self.vel_mean

        mask = torch.ones(B, N, device=out.device, dtype=out.dtype)
        for b in range(B):
            if idcs_airfoil[b] is not None and len(idcs_airfoil[b]) > 0:
                mask[b, idcs_airfoil[b].to(out.device)] = 0.0
        return out * mask.unsqueeze(1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
    """Run validation, log to W&B. Returns mean val metric (L2 velocity error)."""
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

                pred = model(v_in, pos, t, idcs)  # [B, 5, N, 3]

                # L2 velocity error (competition hint metric)
                l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))  # [B]
                total_l2 += l2_err.sum().item()

                # Per-component MAE
                mae = (pred - v_out).abs().mean(dim=(1, 2))  # [B, 3]
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

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))  # minutes


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


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

    model = BaselineMLP(
        hidden=256, n_blocks=6, edge_blocks=5, edge_k=16, n_anchors=10000,
        coarse_blocks=3, coarse_k=32, n_anchors_coarse=2048,
        vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
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

            # Augment: random Y-flip (F1 wake symmetry). Applies per-batch.
            if torch.rand(1, device=device).item() < 0.5:
                v_in = v_in * torch.tensor([1., -1., 1.], device=device).view(1, 1, 1, 3)
                v_out = v_out * torch.tensor([1., -1., 1.], device=device).view(1, 1, 1, 3)
                pos = pos * torch.tensor([1., -1., 1.], device=device).view(1, 1, 3)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(v_in, pos, t, idcs)
                loss = ((pred - v_out) / model.vel_std).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
