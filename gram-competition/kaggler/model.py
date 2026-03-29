"""Model definitions for GRAM airflow prediction."""

import torch
import torch.nn as nn
from data import T_IN, T_OUT


class FourierFeatures(nn.Module):
    """Random Fourier features for positional encoding."""
    def __init__(self, in_dim, n_freqs=64):
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freqs) * 2.0)

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


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


class ResidualMLP(nn.Module):
    """
    Improved MLP with:
    - Fourier positional features
    - Time conditioning
    - Input normalization
    - Residual prediction (predict delta from last input timestep)
    - No-slip boundary enforcement
    """

    def __init__(self, hidden=512, n_blocks=10, n_freqs=128, dropout=0.0,
                 vel_mean=None, vel_std=None):
        super().__init__()
        self.n_freqs = n_freqs

        # Register normalization stats
        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

        # Fourier features for position
        self.pos_ff = FourierFeatures(3, n_freqs)
        pos_dim = 2 * n_freqs

        # Time embedding
        self.time_ff = FourierFeatures(1, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(64 * 10, 256),
            nn.GELU(),
            nn.Linear(256, hidden),
        )

        # Input: pos_fourier + pos_raw(3) + normalized velocity_in(5*3=15)
        # + velocity temporal features (3: last timestep, 3: mean, 3: std, 3: delta from t0 to t-1)
        in_dim = pos_dim + 3 + T_IN * 3 + 12
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)

        # Two stages of blocks with a bottleneck
        self.blocks1 = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks // 2)])
        self.mid_norm = nn.LayerNorm(hidden)
        self.blocks2 = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks - n_blocks // 2)])

        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocity
        v_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)

        # Fourier position features
        pos_feat = self.pos_ff(pos)  # [B, N, 2*n_freqs]

        # Velocity features
        vel_flat = v_norm.reshape(B, N, T * C)  # [B, N, 15]

        # Temporal features per point
        last_vel = v_norm[:, -1]  # [B, N, 3]
        vel_mean = v_norm.mean(dim=1)  # [B, N, 3]
        vel_std_t = v_norm.std(dim=1)  # [B, N, 3]
        vel_delta = v_norm[:, -1] - v_norm[:, 0]  # [B, N, 3] temporal trend

        # Combine inputs
        x = torch.cat([pos, pos_feat, vel_flat, last_vel, vel_mean, vel_std_t, vel_delta], dim=-1)
        x = self.proj_in(x)

        # Time conditioning
        t_feat = self.time_ff(t.unsqueeze(-1))  # [B, 10, 64]
        t_feat = t_feat.reshape(B, -1)  # [B, 640]
        t_cond = self.time_mlp(t_feat)  # [B, hidden]
        x = x + t_cond.unsqueeze(1)

        # Two-stage processing
        x = self.blocks1(x)
        x = self.mid_norm(x)
        x = self.blocks2(x)

        delta = self.proj_out(x).reshape(B, T_OUT, N, 3)

        # Residual: predict delta in normalized space, then denormalize
        last_vel_raw = velocity_in[:, -1:]  # [B, 1, N, 3]
        pred = last_vel_raw + delta * (self.vel_std.squeeze(1) + 1e-8)

        # No-slip boundary condition
        for i, idcs in enumerate(idcs_airfoil):
            pred[i, :, idcs, :] = 0.0

        return pred
