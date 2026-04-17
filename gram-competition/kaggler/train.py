"""Train a 3D airflow velocity predictor.

Transolver: per-point encoder -> L Transolver blocks with soft slice attention
-> per-point decoder that predicts a residual delta from the last input step.
Zero velocity at airfoil (no-slip BC) is enforced.

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
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# Global domain bbox (in world coords, meters) — all geometries share this.
DOMAIN_MIN = torch.tensor([0.0, -0.41, 0.0])
DOMAIN_MAX = torch.tensor([2.10, 0.41, 1.22])


# ---------------------------------------------------------------------------
# Fourier features
# ---------------------------------------------------------------------------


def fourier_features(x: torch.Tensor, num_freqs: int, scale: float = 1.0) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype) * math.pi * scale
    xf = x.unsqueeze(-1) * freqs  # [..., C, F]
    enc = torch.cat([xf.sin(), xf.cos()], dim=-1)  # [..., C, 2F]
    enc = enc.reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


# ---------------------------------------------------------------------------
# Transolver block
# ---------------------------------------------------------------------------


class TransolverBlock(nn.Module):
    """Soft-slice attention:
       1) Point -> M slices via softmax(Wx)
       2) Aggregate points into slice features
       3) Self-attention over M slices
       4) Broadcast slice features back to points using same assignments
    """

    def __init__(self, d, heads=8, slices=64, mlp_mult=2, dropout=0.0):
        super().__init__()
        self.d = d
        self.heads = heads
        self.slices = slices
        self.head_dim = d // heads
        assert d % heads == 0

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        # Per-head slice assignment logits.
        self.to_slice_logits = nn.Linear(d, heads * slices)
        self.temp = nn.Parameter(torch.tensor(1.0))

        self.to_v = nn.Linear(d, d)

        # Slice self-attention: each head runs its own QKV over its slice set.
        self.to_qkv = nn.Linear(self.head_dim, self.head_dim * 3)

        self.out = nn.Linear(d, d)

        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_mult),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d * mlp_mult, d),
        )

    def forward(self, x):
        # x: [B, N, d]
        B, N, d = x.shape
        H, M = self.heads, self.slices

        h0 = self.ln1(x)
        logits = self.to_slice_logits(h0) / self.temp.clamp(min=0.1)
        logits = rearrange(logits, 'b n (h m) -> b h n m', h=H, m=M)
        assign = logits.softmax(dim=-1)  # point -> slice, sums to 1 per point

        v = self.to_v(h0)
        v = rearrange(v, 'b n (h c) -> b h n c', h=H)  # [B, H, N, C]

        # Aggregate points into slices (weighted mean).
        # slice_j = sum_i (w_ij * v_i) / sum_i w_ij
        num = torch.einsum('bhnm,bhnc->bhmc', assign, v)                      # [B, H, M, C]
        denom = assign.sum(dim=2).unsqueeze(-1).clamp(min=1e-4)               # [B, H, M, 1]
        slices = num / denom

        # Self-attention among M slices (per head).
        qkv = self.to_qkv(slices)  # [B, H, M, 3C]
        q, k, v2 = qkv.chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        slices_out = attn @ v2  # [B, H, M, C]

        # Broadcast slice -> points using same assignment.
        out = torch.einsum('bhnm,bhmc->bhnc', assign, slices_out)
        out = rearrange(out, 'b h n c -> b n (h c)')
        out = self.out(out)

        x = x + out
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Voxel U-Net
# ---------------------------------------------------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class VoxelUNet(nn.Module):
    """Voxelize point features into a regular 3D grid, run U-Net, devoxelize."""

    def __init__(self, hidden, grid=(64, 32, 32), ch_base=64):
        super().__init__()
        self.hidden = hidden
        self.grid = tuple(grid)
        self.register_buffer("_grid_buf", torch.tensor(list(grid), dtype=torch.long))
        self.enc1 = DoubleConv(hidden, ch_base)
        self.enc2 = DoubleConv(ch_base, ch_base * 2)
        self.enc3 = DoubleConv(ch_base * 2, ch_base * 4)
        self.dec2 = DoubleConv(ch_base * 4 + ch_base * 2, ch_base * 2)
        self.dec1 = DoubleConv(ch_base * 2 + ch_base, ch_base)
        self.out = nn.Conv3d(ch_base, hidden, 1)
        self.pool = nn.MaxPool3d(2)

    def _voxelize(self, x, pos01):
        B, N, C = x.shape
        Gx, Gy, Gz = self.grid
        idx = pos01.clamp(0, 1 - 1e-6)
        idx = torch.stack([
            (idx[..., 0] * Gx).floor().long(),
            (idx[..., 1] * Gy).floor().long(),
            (idx[..., 2] * Gz).floor().long(),
        ], dim=-1)
        flat = idx[..., 0] * (Gy * Gz) + idx[..., 1] * Gz + idx[..., 2]

        voxel = torch.zeros(B, C, Gx * Gy * Gz, device=x.device, dtype=x.dtype)
        count = torch.zeros(B, 1, Gx * Gy * Gz, device=x.device, dtype=x.dtype)
        x_t = x.transpose(1, 2)
        voxel.scatter_add_(2, flat.unsqueeze(1).expand(-1, C, -1), x_t)
        ones = torch.ones(B, 1, N, device=x.device, dtype=x.dtype)
        count.scatter_add_(2, flat.unsqueeze(1), ones)
        voxel = voxel / count.clamp(min=1.0)
        return voxel.view(B, C, Gx, Gy, Gz)

    def _devoxelize(self, voxel, pos01):
        B, C, Gx, Gy, Gz = voxel.shape
        N = pos01.shape[1]
        p = pos01 * 2.0 - 1.0
        grid = p[..., [2, 1, 0]].view(B, 1, 1, N, 3)
        out = F.grid_sample(voxel, grid, mode='bilinear',
                            padding_mode='border', align_corners=False)
        return out.squeeze(2).squeeze(2).transpose(1, 2)

    def forward(self, x, pos01):
        v0 = self._voxelize(x, pos01)
        e1 = self.enc1(v0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-3:], mode='trilinear', align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-3:], mode='trilinear', align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        y = self.out(d1)
        z = self._devoxelize(y, pos01)
        return z


class ResMLP(nn.Module):
    def __init__(self, d, mult=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d * mult),
            nn.GELU(),
            nn.Linear(d * mult, d),
        )

    def forward(self, x):
        return x + self.net(x)


class VoxelUNetModel(nn.Module):
    """Point-cloud model with a voxel-grid 3D U-Net backbone."""

    def __init__(self, hidden=256, num_pre=2, num_post=4,
                 grid=(64, 32, 32), ch_base=64,
                 num_pos_freqs=10, num_vel_freqs=3, num_dist_freqs=6,
                 vel_mean=None, vel_std=None):
        super().__init__()
        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vin_dim = T_IN * 3
        vin_fourier_dim = 3 * 2 * num_vel_freqs
        dist_dim = 2 + 2 * num_dist_freqs
        in_dim = pos_dim + vin_dim + vin_fourier_dim + dist_dim + 1

        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.ModuleList([ResMLP(hidden) for _ in range(num_pre)])
        self.unet = VoxelUNet(hidden, grid=grid, ch_base=ch_base)
        self.blocks_post = nn.ModuleList([ResMLP(hidden) for _ in range(num_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, T_OUT * 3)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        self.register_buffer("domain_min", DOMAIN_MIN.view(1, 1, 3))
        self.register_buffer("domain_max", DOMAIN_MAX.view(1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil, dist_airfoil):
        B, T, N, C = velocity_in.shape
        last_v = velocity_in[:, -1]
        v_norm = (velocity_in - self.vel_mean) / self.vel_std

        pos_feat = fourier_features(pos, self.num_pos_freqs, scale=1.0)
        vin_flat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T_IN * 3)
        last_norm = v_norm[:, -1]
        vfreqs = 2.0 ** torch.arange(self.num_vel_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        vf = last_norm.unsqueeze(-1) * vfreqs
        vin_fourier = torch.cat([vf.sin(), vf.cos()], dim=-1).reshape(B, N, -1)

        airfoil_ind = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            airfoil_ind[b, idx.to(pos.device).long(), 0] = 1.0

        d = dist_airfoil.unsqueeze(-1)
        d_log = torch.log1p(d)
        dfreqs = 2.0 ** torch.arange(self.num_dist_freqs, device=pos.device, dtype=pos.dtype) * math.pi
        df = d_log * dfreqs
        d_feat = torch.cat([d, d_log, df.sin(), df.cos()], dim=-1)

        feat = torch.cat([pos_feat, vin_flat, vin_fourier, d_feat, airfoil_ind], dim=-1)
        h = self.proj_in(feat)
        for blk in self.blocks_pre:
            h = blk(h)

        pos01 = (pos - self.domain_min) / (self.domain_max - self.domain_min)
        h = h + self.unet(h, pos01)

        for blk in self.blocks_post:
            h = blk(h)

        h = self.norm_out(h)
        delta_norm = self.proj_out(h).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_v.unsqueeze(1) + delta

        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx.to(pred.device).long(), :] = 0.0

        return pred


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class TransolverModel(nn.Module):
    def __init__(self, hidden=256, n_blocks=6, heads=8, slices=64,
                 num_pos_freqs=10, num_vel_freqs=3, num_dist_freqs=6,
                 dropout=0.0, vel_mean=None, vel_std=None):
        super().__init__()
        pos_dim = 3 * (1 + 2 * num_pos_freqs)
        vin_dim = T_IN * 3
        vin_fourier_dim = 3 * 2 * num_vel_freqs
        # distance-to-airfoil: raw + log + fourier (num_dist_freqs freqs, sin/cos)
        dist_dim = 2 + 2 * num_dist_freqs
        in_dim = pos_dim + vin_dim + vin_fourier_dim + dist_dim + 1

        self.num_pos_freqs = num_pos_freqs
        self.num_vel_freqs = num_vel_freqs
        self.num_dist_freqs = num_dist_freqs

        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([
            TransolverBlock(hidden, heads=heads, slices=slices, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, T_OUT * 3),
        )
        nn.init.zeros_(self.proj_out[1].weight)
        nn.init.zeros_(self.proj_out[1].bias)

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil, dist_airfoil):
        B, T, N, C = velocity_in.shape
        last_v = velocity_in[:, -1]

        v_norm = (velocity_in - self.vel_mean) / self.vel_std

        pos_feat = fourier_features(pos, self.num_pos_freqs, scale=1.0)
        vin_flat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T_IN * 3)
        last_norm = v_norm[:, -1]
        freqs = 2.0 ** torch.arange(
            self.num_vel_freqs, device=pos.device, dtype=pos.dtype
        ) * math.pi
        vf = last_norm.unsqueeze(-1) * freqs
        vin_fourier = torch.cat([vf.sin(), vf.cos()], dim=-1).reshape(B, N, -1)

        airfoil_ind = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            airfoil_ind[b, idx.to(pos.device).long(), 0] = 1.0

        # Distance-to-airfoil features: raw, log1p, and fourier on log1p(dist).
        d = dist_airfoil.unsqueeze(-1)                                    # [B, N, 1]
        d_log = torch.log1p(d)
        dfreqs = 2.0 ** torch.arange(
            self.num_dist_freqs, device=pos.device, dtype=pos.dtype
        ) * math.pi
        df = d_log * dfreqs
        d_feat = torch.cat([d, d_log, df.sin(), df.cos()], dim=-1)       # [B, N, 2+2*F]

        x = torch.cat([pos_feat, vin_flat, vin_fourier, d_feat, airfoil_ind], dim=-1)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)

        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = last_v.unsqueeze(1) + delta

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
                dist = compute_dist_to_airfoil(pos, idcs)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(v_in, pos, t, idcs, dist)
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
# Config + training
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1
    epochs: int = 120
    # Subsample points during training for speed + regularization.
    subsample_points: int = 16384
    hidden: int = 256
    n_blocks: int = 6
    heads: int = 8
    slices: int = 64
    num_pos_freqs: int = 10
    num_vel_freqs: int = 3
    num_dist_freqs: int = 6
    dropout: float = 0.0
    # Random y-axis reflection augmentation (wing is y-symmetric).
    yflip_aug: bool = False
    # Optional warm-start from a checkpoint file before training.
    init_from: str | None = None
    # "transolver" or "unet"
    model_type: str = "unet"
    # VoxelUNet hyperparameters
    unet_grid_x: int = 96
    unet_grid_y: int = 48
    unet_grid_z: int = 48
    unet_ch_base: int = 64
    unet_num_pre: int = 2
    unet_num_post: int = 4
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


def subsample_batch(v_in, v_out, pos, idcs_airfoil, dist_airfoil, k):
    B, T, N, C = v_in.shape
    if k is None or k >= N:
        return v_in, v_out, pos, idcs_airfoil, dist_airfoil
    new_idcs = []
    perm = torch.stack([torch.randperm(N, device=v_in.device)[:k] for _ in range(B)])
    v_in_s = torch.gather(v_in, 2, perm[:, None, :, None].expand(-1, T, -1, C))
    v_out_s = torch.gather(v_out, 2, perm[:, None, :, None].expand(-1, T_OUT, -1, C))
    pos_s = torch.gather(pos, 1, perm[:, :, None].expand(-1, -1, C))
    dist_s = torch.gather(dist_airfoil, 1, perm)
    for b in range(B):
        af = idcs_airfoil[b].to(v_in.device).long()
        mask_full = torch.zeros(N, dtype=torch.bool, device=v_in.device)
        mask_full[af] = True
        mask_sub = mask_full[perm[b]]
        new_idcs.append(mask_sub.nonzero(as_tuple=False).squeeze(-1))
    return v_in_s, v_out_s, pos_s, new_idcs, dist_s


_DIST_CACHE: dict = {}


def _geom_key(pos_b: torch.Tensor, idcs_b: torch.Tensor) -> tuple:
    # Geometry is determined by pos + airfoil idcs; fingerprint cheaply.
    p = pos_b.flatten()[:6].detach().cpu().tolist()
    return (len(idcs_b), int(idcs_b.numel()), tuple(round(v, 4) for v in p))


def compute_dist_to_airfoil(pos: torch.Tensor, idcs_airfoil: list) -> torch.Tensor:
    """pos: [B, N, 3]; returns [B, N] unsigned distance to nearest airfoil point."""
    B, N, _ = pos.shape
    out = torch.empty(B, N, device=pos.device, dtype=pos.dtype)
    for b in range(B):
        idx = idcs_airfoil[b].to(pos.device).long()
        key = _geom_key(pos[b], idx)
        cached = _DIST_CACHE.get(key)
        if cached is not None:
            out[b] = cached.to(pos.device, non_blocking=True)
            continue
        af = pos[b].index_select(0, idx)                 # [M, 3]
        # Chunked min-distance to avoid OOM for 100k x 15k.
        chunks = []
        for chunk in pos[b].split(8192):
            d2 = torch.cdist(chunk, af)                  # [chunk, M]
            chunks.append(d2.min(dim=1).values)
        dist = torch.cat(chunks, dim=0)                  # [N]
        _DIST_CACHE[key] = dist.detach().to("cpu")
        out[b] = dist
    return out


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

    if cfg.model_type == "unet":
        model = VoxelUNetModel(
            hidden=cfg.hidden, num_pre=cfg.unet_num_pre, num_post=cfg.unet_num_post,
            grid=(cfg.unet_grid_x, cfg.unet_grid_y, cfg.unet_grid_z),
            ch_base=cfg.unet_ch_base,
            num_pos_freqs=cfg.num_pos_freqs, num_vel_freqs=cfg.num_vel_freqs,
            num_dist_freqs=cfg.num_dist_freqs,
            vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
        ).to(device)
    else:
        model = TransolverModel(
            hidden=cfg.hidden, n_blocks=cfg.n_blocks, heads=cfg.heads, slices=cfg.slices,
            num_pos_freqs=cfg.num_pos_freqs, num_vel_freqs=cfg.num_vel_freqs,
            num_dist_freqs=cfg.num_dist_freqs,
            dropout=cfg.dropout,
            vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M params")

    if cfg.init_from:
        sd = torch.load(cfg.init_from, map_location=device, weights_only=True)
        # Allow legacy checkpoints without grid_buf / domain_min / domain_max.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"Unexpected keys in init_from: {unexpected[:5]}")
        if missing:
            print(f"Missing keys in init_from: {missing[:5]}")
        print(f"Warm-started from {cfg.init_from}")

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

            dist = compute_dist_to_airfoil(pos, idcs)

            v_in_s, v_out_s, pos_s, idcs_s, dist_s = subsample_batch(
                v_in, v_out, pos, idcs, dist, cfg.subsample_points
            )

            # y-flip: wing is symmetric about y=0. Distance-to-airfoil is invariant.
            if cfg.yflip_aug and torch.rand(1).item() < 0.5:
                pos_s = pos_s.clone(); pos_s[..., 1] = -pos_s[..., 1]
                v_in_s = v_in_s.clone(); v_in_s[..., 1] = -v_in_s[..., 1]
                v_out_s = v_out_s.clone(); v_out_s[..., 1] = -v_out_s[..., 1]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred = model(v_in_s, pos_s, t, idcs_s, dist_s)
                # Normalize residual per-channel so Ux (large std) doesn't dominate
                # and Uy/Uz get equal training weight — matches leaderboard L2 metric.
                err = (pred.float() - v_out_s) / model.vel_std
                loss = err.pow(2).mean()

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
        if cfg.yflip_aug:
            pred_cmd += ["--yflip_tta", "True"]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()


if __name__ == "__main__":
    main()
