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
# Spatial mixing via chunked local attention
# ---------------------------------------------------------------------------

class LocalAttention(nn.Module):
    """Self-attention within spatial chunks.
    Points are sorted by a space-filling curve (z-order) and grouped into chunks.
    Attention within each chunk provides local spatial interaction.
    """
    def __init__(self, dim, n_heads=4, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _z_order_sort(self, pos):
        """Sort points by z-order (Morton code) for spatial locality."""
        # Normalize positions to [0, 1023] for 10-bit Morton code
        pos_min = pos.min(dim=1, keepdim=True).values
        pos_max = pos.max(dim=1, keepdim=True).values
        pos_norm = ((pos - pos_min) / (pos_max - pos_min + 1e-6) * 1023).long().clamp(0, 1023)

        # Simple z-order: interleave bits (approximate with weighted sum for speed)
        morton = pos_norm[..., 0] * 1048576 + pos_norm[..., 1] * 1024 + pos_norm[..., 2]
        return morton.argsort(dim=1)

    def forward(self, x, pos):
        """x: [B, N, D], pos: [B, N, 3]"""
        B, N, D = x.shape

        # Sort by spatial locality
        sort_idx = self._z_order_sort(pos)  # [B, N]
        # Gather to sorted order
        x_sorted = torch.gather(x, 1, sort_idx.unsqueeze(-1).expand(-1, -1, D))

        # Pad to multiple of chunk_size
        cs = self.chunk_size
        pad_n = (cs - N % cs) % cs
        if pad_n > 0:
            x_sorted = F.pad(x_sorted, (0, 0, 0, pad_n))

        N_padded = x_sorted.shape[1]
        n_chunks = N_padded // cs

        # Reshape into chunks
        x_chunks = x_sorted.reshape(B * n_chunks, cs, D)

        # Multi-head self-attention
        qkv = self.qkv(x_chunks).reshape(B * n_chunks, cs, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nc, nh, cs, hd]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B * n_chunks, cs, D)
        out = self.proj(out)

        # Reshape back
        out = out.reshape(B, N_padded, D)
        if pad_n > 0:
            out = out[:, :N, :]

        # Unsort: scatter back to original order
        unsort_idx = sort_idx.argsort(dim=1)
        out = torch.gather(out, 1, unsort_idx.unsqueeze(-1).expand(-1, -1, D))

        return self.norm(out)


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


class SpatialMixBlock(nn.Module):
    """MLP block + local attention for spatial mixing."""
    def __init__(self, dim, n_heads=4, chunk_size=512, dropout=0.0):
        super().__init__()
        self.attn = LocalAttention(dim, n_heads, chunk_size)
        self.mlp = ResBlock(dim, dropout)

    def forward(self, x, pos):
        x = x + self.attn(x, pos)
        x = self.mlp(x)
        return x


class VelocityPredictor(nn.Module):
    """MLP with local spatial attention for velocity prediction.

    1. Encode per-point features
    2. Apply alternating MLP blocks and spatial attention blocks
    3. Residual prediction from last timestep
    4. No-slip BC
    """

    def __init__(self, hidden=256, n_mlp_blocks=4, n_spatial_blocks=3,
                 n_heads=4, chunk_size=512, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.n_spatial_blocks = n_spatial_blocks
        in_dim = 3 + T_IN * 3 + (T_IN - 1) * 3  # 30
        out_dim = T_OUT * 3  # 15

        self.proj_in = nn.Linear(in_dim, hidden)

        # Alternating MLP and spatial attention blocks
        self.blocks = nn.ModuleList()
        for i in range(n_mlp_blocks + n_spatial_blocks):
            if i % 2 == 0 and i // 2 < n_spatial_blocks:
                self.blocks.append(SpatialMixBlock(hidden, n_heads, chunk_size, dropout))
            else:
                self.blocks.append(ResBlock(hidden, dropout))

        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        last_vel = velocity_in[:, -1]

        vel_diff = velocity_in[:, 1:] - velocity_in[:, :-1]
        vel_flat = velocity_in.reshape(B, N, T * C)
        diff_flat = vel_diff.reshape(B, N, (T - 1) * C)
        x = torch.cat([pos, vel_flat, diff_flat], dim=-1)

        x = self.proj_in(x)

        for block in self.blocks:
            if isinstance(block, SpatialMixBlock):
                x = block(x, pos)
            else:
                x = block(x)

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


def augment_batch(v_in, v_out, pos):
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
        weight_decay: float = 0.01
        batch_size: int = 1
        epochs: int = 60
        splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
        wandb_group: str | None = None
        wandb_name: str | None = None
        agent: str | None = None
        debug: bool = False
        hidden: int = 256
        n_mlp_blocks: int = 4
        n_spatial_blocks: int = 3
        n_heads: int = 8
        chunk_size: int = 1024
        dropout: float = 0.1
        augment: bool = True

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
        hidden=cfg.hidden, n_mlp_blocks=cfg.n_mlp_blocks,
        n_spatial_blocks=cfg.n_spatial_blocks, n_heads=cfg.n_heads,
        chunk_size=cfg.chunk_size, dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_epochs = 3
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, MAX_EPOCHS - warmup_epochs)
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

    model_cfg_dict = {
        "hidden": cfg.hidden, "n_mlp_blocks": cfg.n_mlp_blocks,
        "n_spatial_blocks": cfg.n_spatial_blocks, "n_heads": cfg.n_heads,
        "chunk_size": cfg.chunk_size, "dropout": cfg.dropout,
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

            if cfg.augment:
                v_in, v_out, pos = augment_batch(v_in, v_out, pos)

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

        wandb.log({"train/epoch_loss": epoch_loss, "lr": optimizer.param_groups[0]['lr'],
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
