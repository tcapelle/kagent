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


def sinusoidal_embed(x, dim):
    """Sinusoidal embedding for tensor x of shape [..., K]. Returns [..., K, dim]."""
    device = x.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = x.unsqueeze(-1) * freqs
    return torch.cat([args.sin(), args.cos()], dim=-1)


def fourier_pos_embed(pos, num_bands=16):
    """Fourier positional features for 3D positions. Returns [..., 6*num_bands]."""
    # pos: [..., 3]
    scales = 2.0 ** torch.arange(num_bands, device=pos.device, dtype=torch.float32)  # [nb]
    # [..., 3, nb]
    x = pos.unsqueeze(-1) * scales * 3.14159265
    return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)  # [..., 3*2*nb]


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class Attention(nn.Module):
    """Generic attention: query set attends to key/value set. q,kv separately normed."""

    def __init__(self, dim, heads=8, dim_head=64, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, q_in, kv_in):
        q = self.to_q(self.norm_q(q_in))
        k, v = self.to_kv(self.norm_kv(kv_in)).chunk(2, dim=-1)
        B, Nq, _ = q.shape
        Nk = k.shape[1]
        q = q.view(B, Nq, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(B, Nk, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, Nk, self.heads, self.dim_head).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, Nq, self.heads * self.dim_head)
        return q_in + self.to_out(out)


class SelfAttn(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head)

    def forward(self, x):
        return self.attn(x, x)


class Perceiver(nn.Module):
    """Perceiver-IO for 3D airflow prediction.

    - Point features: Fourier(pos) + time-avg velocity + per-timestep velocity + airfoil mask.
    - Encoder: L learned latent queries cross-attend to N points (shared once).
    - Processor: self-attention + MLP on the L latents, several blocks.
    - Decoder: N point queries cross-attend to L latents for final features.
    - Output head: pointwise MLP predicting residual delta per output timestep.
    - Physics: velocity normalization, residual to v_in[-1], no-slip BC on airfoil.
    """

    def __init__(
        self,
        point_dim=256,
        latent_dim=384,
        n_latents=128,
        n_process_blocks=6,
        heads=8,
        dim_head=48,
        fourier_bands=16,
        vel_mean=None,
        vel_std=None,
    ):
        super().__init__()
        in_feat = 3 * 2 * fourier_bands + T_IN * 3 + 3 + 1  # fourier_pos + v_in(15) + v_mean(3) + mask(1)
        self.fourier_bands = fourier_bands

        self.proj_in = nn.Sequential(nn.Linear(in_feat, point_dim), nn.GELU(), nn.Linear(point_dim, point_dim))

        # Learned latent queries, modulated by sample-level features (time).
        self.latents = nn.Parameter(torch.randn(n_latents, latent_dim) * 0.02)
        t_dim = 64
        self.t_proj = nn.Sequential(
            nn.Linear(10 * t_dim, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim)
        )
        self.t_embed_dim = t_dim

        self.enc_attn = Attention(latent_dim, heads=heads, dim_head=dim_head, kv_dim=point_dim)
        self.enc_ff = FeedForward(latent_dim)

        self.proc_attn = nn.ModuleList([SelfAttn(latent_dim, heads=heads, dim_head=dim_head) for _ in range(n_process_blocks)])
        self.proc_ff = nn.ModuleList([FeedForward(latent_dim) for _ in range(n_process_blocks)])

        self.dec_attn = Attention(point_dim, heads=heads, dim_head=dim_head, kv_dim=latent_dim)
        self.dec_ff = FeedForward(point_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(point_dim), nn.Linear(point_dim, point_dim), nn.GELU(), nn.Linear(point_dim, T_OUT * 3)
        )

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        device = pos.device

        # Velocity normalization + residual anchor.
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_last_norm = v_norm[:, -1:, :, :]  # [B,1,N,3]
        v_time_mean = v_norm.mean(dim=1)     # [B,N,3]

        # Per-sample position normalization (center + scale) for Fourier features.
        pos_mean = pos.mean(dim=1, keepdim=True)
        pos_scale = pos.std(dim=(1, 2), keepdim=True).clamp_min(1e-3)
        pos_norm = (pos - pos_mean) / pos_scale
        pos_feat = fourier_pos_embed(pos_norm, num_bands=self.fourier_bands)  # [B,N,6*nb]

        # Airfoil mask feature.
        airfoil_mask = torch.zeros(B, N, 1, device=device, dtype=pos.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            airfoil_mask[b, idcs.to(device), 0] = 1.0

        point_in = torch.cat([
            pos_feat,
            v_norm.permute(0, 2, 1, 3).reshape(B, N, T * C),
            v_time_mean,
            airfoil_mask,
        ], dim=-1)
        x = self.proj_in(point_in)  # [B, N, point_dim]

        # Sample-level conditioning from time values.
        t_emb = sinusoidal_embed(t, self.t_embed_dim).reshape(B, -1)
        cond = self.t_proj(t_emb).unsqueeze(1)  # [B, 1, latent_dim]

        # Init latents and add conditioning.
        lat = self.latents.unsqueeze(0).expand(B, -1, -1) + cond  # [B, L, latent_dim]

        # Encoder cross-attn.
        lat = self.enc_attn(lat, x)
        lat = self.enc_ff(lat)

        # Processor self-attn blocks.
        for attn, ff in zip(self.proc_attn, self.proc_ff):
            lat = attn(lat)
            lat = ff(lat)

        # Decoder cross-attn: points attend to latents.
        x = self.dec_attn(x, lat)
        x = self.dec_ff(x)

        # Pointwise head for residual delta.
        delta_norm = self.head(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)  # [B,T_OUT,N,3]
        pred_norm = v_last_norm + delta_norm
        pred = pred_norm * self.vel_std + self.vel_mean

        # No-slip BC.
        for b, idcs in enumerate(idcs_airfoil):
            pred[b, :, idcs.to(device), :] = 0.0
        return pred


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

# Single source of truth for the model architecture; both train.py and
# predict.py import it so the model can always be reconstructed from a checkpoint.
MODEL_CFG = dict(
    point_dim=256,
    latent_dim=384,
    n_latents=128,
    n_process_blocks=6,
    heads=6,
    dim_head=64,
    fourier_bands=16,
)


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


if __name__ == "__main__":
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

    model = Perceiver(**MODEL_CFG, vel_mean=stats["vel_mean"], vel_std=stats["vel_std"]).to(device)

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

            pred = model(v_in, pos, t, idcs)
            loss = (pred - v_out).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
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
