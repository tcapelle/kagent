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


class ResConv3d(nn.Module):
    """Residual 3D conv block with GroupNorm + GELU."""

    def __init__(self, dim, groups=8):
        super().__init__()
        self.n1 = nn.GroupNorm(min(groups, dim), dim)
        self.c1 = nn.Conv3d(dim, dim, 3, padding=1)
        self.n2 = nn.GroupNorm(min(groups, dim), dim)
        self.c2 = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        h = self.c1(torch.nn.functional.gelu(self.n1(x)))
        h = self.c2(torch.nn.functional.gelu(self.n2(h)))
        return x + h


def voxelize(pos_norm, feat, grid_size):
    """Scatter point features onto a 3D voxel grid via mean-pool.

    pos_norm in [-1, 1]; feat [B, N, C]; returns voxel [B, C, G, G, G] and
    occupancy mask [B, 1, G, G, G].
    Axis convention: spatial axes 2/3/4 correspond to x/y/z.
    """
    B, N, C = feat.shape
    G = grid_size
    idx = ((pos_norm * 0.5 + 0.5) * G).floor().clamp(0, G - 1).long()  # [B, N, 3]
    flat = idx[..., 0] * (G * G) + idx[..., 1] * G + idx[..., 2]        # [B, N]

    out = torch.zeros(B, C, G * G * G, device=feat.device, dtype=feat.dtype)
    cnt = torch.zeros(B, 1, G * G * G, device=feat.device, dtype=feat.dtype)
    out.scatter_add_(2, flat.unsqueeze(1).expand(B, C, N), feat.transpose(1, 2))
    cnt.scatter_add_(2, flat.unsqueeze(1), torch.ones(B, 1, N, device=feat.device, dtype=feat.dtype))
    out = out / cnt.clamp_min(1.0)
    mask = (cnt > 0).to(feat.dtype)
    return out.view(B, C, G, G, G), mask.view(B, 1, G, G, G)


def sample_voxel(vox, pos_norm):
    """Trilinear-sample voxel grid at point positions. Returns [B, N, C]."""
    B, C = vox.shape[:2]
    N = pos_norm.shape[1]
    # grid_sample 3D: grid last dim is (W, H, D). Our axes 2/3/4 = x/y/z → (D, H, W)=(x, y, z)
    # so grid[...,0]=z, grid[...,1]=y, grid[...,2]=x.
    grid = pos_norm[:, :, [2, 1, 0]].view(B, 1, 1, N, 3)
    sampled = torch.nn.functional.grid_sample(
        vox, grid, mode="bilinear", align_corners=False, padding_mode="border"
    )  # [B, C, 1, 1, N]
    return sampled.view(B, C, N).transpose(1, 2)


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


class VoxelBottleneckAttn(nn.Module):
    """Self-attention over flattened bottleneck voxels for global mixing."""

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        self.attn = SelfAttn(dim, heads=heads, dim_head=dim_head)
        self.ff = FeedForward(dim)

    def forward(self, x):
        B, C, D, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, D*H*W, C]
        tokens = self.attn(tokens)
        tokens = self.ff(tokens)
        return tokens.transpose(1, 2).reshape(B, C, D, H, W)


class PhysicsAttention(nn.Module):
    """Transolver-style Physics-Attention: soft-cluster N points into M slices,
    attend on M tokens, scatter back. Linear in N. From Wu et al. ICML 2024."""

    def __init__(self, dim, heads=8, dim_head=32, slice_num=64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num
        inner = heads * dim_head
        self.in_proj_fx = nn.Linear(dim, inner)
        self.in_proj_x = nn.Linear(dim, inner)
        self.in_proj_slice = nn.Linear(dim_head, slice_num)
        self.qkv = nn.Linear(dim_head, dim_head * 3)
        self.to_out = nn.Linear(inner, dim)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)

    def forward(self, x):
        B, N, _ = x.shape
        H, D, M = self.heads, self.dim_head, self.slice_num
        fx = self.in_proj_fx(x).view(B, N, H, D).transpose(1, 2)  # [B,H,N,D]
        xm = self.in_proj_x(x).view(B, N, H, D).transpose(1, 2)   # [B,H,N,D]
        tau = self.temperature.clamp(0.1, 5.0)
        w = (self.in_proj_slice(xm) / tau).softmax(dim=-1)          # [B,H,N,M]

        z = torch.einsum("bhnc,bhng->bhgc", fx, w)                  # [B,H,M,D]
        z = z / (w.sum(dim=2, keepdim=False).unsqueeze(-1) + 1e-6)

        q, k, v = self.qkv(z).chunk(3, dim=-1)                      # each [B,H,M,D]
        z_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        out = torch.einsum("bhgc,bhng->bhnc", z_out, w)             # [B,H,N,D]
        out = out.transpose(1, 2).reshape(B, N, H * D)
        return self.to_out(out)


class TransolverBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, slice_num=64, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PhysicsAttention(dim, heads, dim_head, slice_num)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(), nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VoxelUNet(nn.Module):
    """3D conv U-Net over voxelized point features.

    Pipeline: per-point encoder MLP → mean-pool into [G,G,G] voxel grid →
    3-level 3D U-Net (G → G/2 → G/4 → G/2 → G) with skip concats →
    trilinear sample back at every input point → pointwise head concats
    voxel-sampled and original point features to predict a residual delta
    on top of v_in[-1] (velocity-normalized). Physics: velocity
    normalization, residual anchor, hard no-slip BC on airfoil points,
    time conditioning broadcast per-point. Optional self-attention at the
    bottleneck for global context.
    """

    def __init__(
        self,
        grid_size=48,
        base_ch=96,
        point_dim=192,
        head_hidden=320,
        fourier_bands=12,
        blocks_per_level=2,
        bottleneck_attn=False,
        attn_heads=8,
        attn_dim_head=64,
        transolver_depth=0,
        transolver_heads=8,
        transolver_dim_head=32,
        transolver_slice_num=64,
        vel_mean=None,
        vel_std=None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.fourier_bands = fourier_bands

        in_feat = 3 * 2 * fourier_bands + T_IN * 3 + 3 + 1
        self.point_enc = nn.Sequential(
            nn.Linear(in_feat, point_dim), nn.GELU(),
            nn.Linear(point_dim, base_ch),
        )

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.down1 = nn.Sequential(*[ResConv3d(c1) for _ in range(blocks_per_level)])
        self.pool1 = nn.Conv3d(c1, c2, 3, stride=2, padding=1)
        self.down2 = nn.Sequential(*[ResConv3d(c2) for _ in range(blocks_per_level)])
        self.pool2 = nn.Conv3d(c2, c3, 3, stride=2, padding=1)
        self.bottleneck = nn.Sequential(*[ResConv3d(c3) for _ in range(blocks_per_level)])
        self.bottleneck_attn = (
            VoxelBottleneckAttn(c3, heads=attn_heads, dim_head=attn_dim_head)
            if bottleneck_attn else None
        )
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.up_block2 = nn.Sequential(
            nn.Conv3d(c2 * 2, c2, 1), *[ResConv3d(c2) for _ in range(blocks_per_level)]
        )
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.up_block1 = nn.Sequential(
            nn.Conv3d(c1 * 2, c1, 1), *[ResConv3d(c1) for _ in range(blocks_per_level)]
        )

        self.t_dim = 64
        self.t_proj = nn.Sequential(
            nn.Linear(10 * self.t_dim, point_dim), nn.GELU(),
            nn.Linear(point_dim, base_ch),
        )

        head_in = base_ch * 2  # voxel-sampled + per-point skip
        self.transolver = nn.ModuleList([
            TransolverBlock(
                head_in,
                heads=transolver_heads,
                dim_head=transolver_dim_head,
                slice_num=transolver_slice_num,
            )
            for _ in range(transolver_depth)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, head_hidden), nn.GELU(),
            nn.Linear(head_hidden, T_OUT * 3),
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

        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_last_norm = v_norm[:, -1:, :, :]
        v_time_mean = v_norm.mean(dim=1)

        # Per-sample bounding-box normalization → [-1, 1], slight shrink to keep
        # edge points strictly inside the voxel grid.
        pos_min = pos.amin(dim=1, keepdim=True)
        pos_max = pos.amax(dim=1, keepdim=True)
        pos_norm = 2.0 * (pos - pos_min) / (pos_max - pos_min).clamp_min(1e-6) - 1.0
        pos_norm = pos_norm * 0.98

        pos_feat = fourier_pos_embed(pos_norm, num_bands=self.fourier_bands)

        airfoil_mask = torch.zeros(B, N, 1, device=device, dtype=pos.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            airfoil_mask[b, idcs.to(device), 0] = 1.0

        point_in = torch.cat([
            pos_feat,
            v_norm.permute(0, 2, 1, 3).reshape(B, N, T * 3),
            v_time_mean,
            airfoil_mask,
        ], dim=-1)
        feat = self.point_enc(point_in)  # [B, N, base_ch]

        t_emb = sinusoidal_embed(t, self.t_dim).reshape(B, -1)
        t_cond = self.t_proj(t_emb).unsqueeze(1)  # [B, 1, base_ch]
        feat = feat + t_cond

        vox, _ = voxelize(pos_norm, feat, self.grid_size)
        x1 = self.down1(vox)
        x2 = self.down2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))
        if self.bottleneck_attn is not None:
            x3 = self.bottleneck_attn(x3)
        u2 = self.up_block2(torch.cat([self.up2(x3), x2], dim=1))
        u1 = self.up_block1(torch.cat([self.up1(u2), x1], dim=1))

        sampled = sample_voxel(u1, pos_norm)
        combined = torch.cat([sampled, feat], dim=-1)
        for block in self.transolver:
            combined = block(combined)
        delta_norm = self.head(combined).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        pred_norm = v_last_norm + delta_norm
        pred = pred_norm * self.vel_std + self.vel_mean

        for b, idcs in enumerate(idcs_airfoil):
            pred[b, :, idcs.to(device), :] = 0.0
        return pred


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
    grid_size=48,
    base_ch=128,
    point_dim=256,
    head_hidden=384,
    fourier_bands=12,
    blocks_per_level=3,
    bottleneck_attn=True,
    attn_heads=8,
    attn_dim_head=64,
    transolver_depth=2,
    transolver_heads=8,
    transolver_dim_head=32,
    transolver_slice_num=32,
)

GRAD_ACCUM = 4
WARMUP_STEPS = 300


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 30
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

    model = VoxelUNet(**MODEL_CFG, vel_mean=stats["vel_mean"], vel_std=stats["vel_std"]).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine schedule over the actual number of optimizer steps we expect to run.
    steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_steps = steps_per_epoch * MAX_EPOCHS
    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
        accum_idx = 0
        optimizer.zero_grad()

        for v_in, v_out, pos, t, idcs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            # y-flip augmentation: F1 front wing ~ symmetric about y=0.
            if torch.rand(1, device=v_in.device).item() < 0.5:
                v_in = v_in.clone(); v_in[..., 1].neg_()
                v_out = v_out.clone(); v_out[..., 1].neg_()
                pos = pos.clone(); pos[..., 1].neg_()

            pred = model(v_in, pos, t, idcs)
            loss = (pred - v_out).pow(2).mean()
            (loss / GRAD_ACCUM).backward()

            accum_idx += 1
            if accum_idx == GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                accum_idx = 0
                global_step += 1
                wandb.log({"train/loss": loss.item(),
                           "train/lr": optimizer.param_groups[0]["lr"],
                           "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss,
                   "lr": optimizer.param_groups[0]["lr"],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            # Save locally first to avoid PVC hangs stalling training; mirror to PVC at end.
            torch.save(model.state_dict(), git_ckpt_path)
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
        # Mirror final best ckpt from local to PVC for durability / predict.py.
        shutil.copyfile(git_ckpt_path, model_path)

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
