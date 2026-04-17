"""Model architecture for GRAM airflow prediction.

Kept in its own module so predict.py can import the class without triggering
train.py's argparser.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import T_IN, T_OUT


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


class BaselineMLP(nn.Module):
    """Point-wise ResMLP: residual prediction (delta from last input frame) +
    input/output normalization + no-slip BC enforcement at airfoil surface.
    """

    def __init__(self, hidden=512, n_blocks=8, vel_mean=None, vel_std=None):
        super().__init__()
        in_dim = 3 + T_IN * 3
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_feat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)
        x = torch.cat([pos, v_feat], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        last = velocity_in[:, -1:, :, :]
        pred = last + delta_norm * self.vel_std
        for b, idc in enumerate(idcs_airfoil):
            pred[b, :, idc, :] = 0.0
        return pred


class VoxelFlowNet(nn.Module):
    """Voxel-grid 3D CNN + point-wise MLP head.

    Pipeline:
      1. Scatter-mean v_in features onto a regular [G,G,G] grid (+occupancy).
      2. Dilated 3D convs expand spatial receptive field across the grid.
      3. Trilinear-sample voxel features back to each point.
      4. Concat per-point feats (pos + v_in_norm + voxel_feat) -> ResMLP.
      5. Residual prediction in normalized space; no-slip BC.
    """

    def __init__(
        self,
        vel_mean,
        vel_std,
        grid_res: int = 48,
        grid_ch: int = 64,
        grid_dilations: tuple = (1, 2, 4, 8),
        point_hidden: int = 384,
        n_point_blocks: int = 6,
        fourier_L: int = 8,
    ):
        super().__init__()
        self.grid_res = grid_res
        self.fourier_L = fourier_L
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        # Fixed bbox (from dataset inspection), padded slightly.
        self.register_buffer("pos_min", torch.tensor([-0.1, -0.5, -0.1]))
        self.register_buffer("pos_max", torch.tensor([2.2, 0.5, 1.3]))
        # Fourier frequencies: 2^k * pi for k in [0, L)
        self.register_buffer("fourier_freqs", (2.0 ** torch.arange(fourier_L)) * torch.pi)

        # mean-pool: T_IN*3 + 1 (occupancy); max-pool: T_IN*3 extra
        in_ch = 2 * T_IN * 3 + 1
        self.grid_in = nn.Conv3d(in_ch, grid_ch, 1)
        self.grid_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(grid_ch, grid_ch, 3, padding=d, dilation=d),
                nn.GroupNorm(8, grid_ch),
                nn.GELU(),
                nn.Conv3d(grid_ch, grid_ch, 3, padding=1),
                nn.GroupNorm(8, grid_ch),
                nn.GELU(),
            )
            for d in grid_dilations
        ])

        pos_feat_dim = 3 + 3 * 2 * fourier_L  # raw pos + sin/cos at L scales
        point_in = pos_feat_dim + T_IN * 3 + grid_ch
        point_out = T_OUT * 3
        self.proj_in = nn.Linear(point_in, point_hidden)
        self.blocks = nn.Sequential(*[ResBlock(point_hidden) for _ in range(n_point_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(point_hidden), nn.Linear(point_hidden, point_out))

    def _voxelize(self, v_in_norm, pos):
        """Scatter per-point features onto a [B, C, G, G, G] grid using BOTH
        mean- and max-pooling, concatenated along channels.

        Mean captures typical flow direction in each cell; max captures the
        most extreme component (useful for turbulent/separation regions).
        """
        B, T, N, C = v_in_norm.shape
        G = self.grid_res
        pos_norm = ((pos - self.pos_min) / (self.pos_max - self.pos_min)).clamp(0, 1 - 1e-6)
        idx = (pos_norm * G).long()  # [B, N, 3]
        flat_idx = idx[..., 0] * G * G + idx[..., 1] * G + idx[..., 2]  # [B, N]

        v_feat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)  # [B, N, 15]
        ones = torch.ones(B, N, 1, device=v_feat.device, dtype=v_feat.dtype)
        feat_mean_in = torch.cat([v_feat, ones], dim=-1)  # [B, N, 16] — +occupancy
        nch_mean = feat_mean_in.shape[-1]

        # Scatter mean
        grid_sum = torch.zeros(B, G * G * G, nch_mean, device=v_feat.device, dtype=v_feat.dtype)
        counts = torch.zeros(B, G * G * G, 1, device=v_feat.device, dtype=v_feat.dtype)
        idx_m = flat_idx.unsqueeze(-1).expand(-1, -1, nch_mean)
        grid_sum.scatter_add_(1, idx_m, feat_mean_in)
        counts.scatter_add_(1, flat_idx.unsqueeze(-1), ones)
        grid_mean = grid_sum / counts.clamp(min=1.0)

        # Scatter max (velocities only — occupancy max == mean, skip)
        grid_max = torch.zeros(B, G * G * G, T * C, device=v_feat.device, dtype=v_feat.dtype)
        idx_x = flat_idx.unsqueeze(-1).expand(-1, -1, T * C)
        grid_max.scatter_reduce_(1, idx_x, v_feat, reduce="amax", include_self=False)

        grid = torch.cat([grid_mean, grid_max], dim=-1)  # [B, G³, 2*T*C+1]
        nch = grid.shape[-1]
        return grid.reshape(B, G, G, G, nch).permute(0, 4, 1, 2, 3).contiguous()

    def _pos_enc(self, pos):
        """Fourier feature encoding: concat [pos, sin(w*pos), cos(w*pos)] for w in 2^k * pi."""
        pos_norm = ((pos - self.pos_min) / (self.pos_max - self.pos_min)) * 2 - 1  # [-1, 1]
        xw = pos_norm.unsqueeze(-1) * self.fourier_freqs  # [B, N, 3, L]
        sin_f = torch.sin(xw).flatten(-2)  # [B, N, 3*L]
        cos_f = torch.cos(xw).flatten(-2)
        return torch.cat([pos, sin_f, cos_f], dim=-1)  # [B, N, 3 + 2*3*L]

    def _interp(self, grid, pos):
        """Trilinear-sample voxel features at each point position."""
        B, C, _, _, _ = grid.shape
        pos_norm = ((pos - self.pos_min) / (self.pos_max - self.pos_min)).clamp(0, 1)
        # grid_sample (5D) coord order is (W, H, D) i.e. (z, y, x) for a grid
        # laid out as [B, C, G_x, G_y, G_z]. Flip (x,y,z) -> (z,y,x).
        coords = (2 * pos_norm - 1).flip(-1)
        coords = coords.view(B, 1, 1, -1, 3)
        sampled = F.grid_sample(grid, coords, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std

        grid = self._voxelize(v_in_norm, pos)
        g = self.grid_in(grid)
        for blk in self.grid_blocks:
            g = g + blk(g)
        voxel_feat = self._interp(g, pos)

        v_feat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)
        pos_feat = self._pos_enc(pos)
        x = torch.cat([pos_feat, v_feat, voxel_feat], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        pred = velocity_in[:, -1:, :, :] + delta_norm * self.vel_std
        for b, idc in enumerate(idcs_airfoil):
            pred[b, :, idc, :] = 0.0
        return pred
