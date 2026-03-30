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


class ResidualMLP(nn.Module):
    """
    Enhanced MLP with residual prediction, normalization, time conditioning,
    Fourier features, and no-slip BC.
    """

    def __init__(self, hidden=512, n_blocks=10, n_freqs=128,
                 vel_mean=None, vel_std=None):
        super().__init__()

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

        self.pos_ff = FourierFeatures(3, n_freqs)
        pos_dim = 2 * n_freqs

        self.time_ff = FourierFeatures(1, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(64 * 10, 256),
            nn.GELU(),
            nn.Linear(256, hidden),
        )

        # Input: pos_fourier + pos(3) + vel_in(5*3) + temporal_feats(12)
        in_dim = pos_dim + 3 + T_IN * 3 + 12

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks1 = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks // 2)])
        self.mid_norm = nn.LayerNorm(hidden)
        self.blocks2 = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks - n_blocks // 2)])
        # Per-timestep output heads for better gradient flow
        self.proj_out = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
            for _ in range(T_OUT)
        ])

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        v_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)

        pos_feat = self.pos_ff(pos)
        vel_flat = v_norm.reshape(B, N, T * C)
        last_vel = v_norm[:, -1]
        vel_mean_t = v_norm.mean(dim=1)
        vel_std_t = v_norm.std(dim=1)
        vel_delta = v_norm[:, -1] - v_norm[:, 0]

        x = torch.cat([pos, pos_feat, vel_flat, last_vel, vel_mean_t, vel_std_t, vel_delta], dim=-1)
        x = self.proj_in(x)

        t_feat = self.time_ff(t.unsqueeze(-1)).reshape(B, -1)
        t_cond = self.time_mlp(t_feat).unsqueeze(1)
        x = x + t_cond

        x = self.blocks1(x)
        x = self.mid_norm(x)
        x = self.blocks2(x)

        # Per-timestep heads predict raw delta
        last_vel_raw = velocity_in[:, -1]  # [B, N, 3]
        preds = []
        for head in self.proj_out:
            delta = head(x)  # [B, N, 3]
            pred = last_vel_raw + delta
            preds.append(pred)

        pred = torch.stack(preds, dim=1)  # [B, T_OUT, N, 3]

        # No-slip BC
        for i, idcs in enumerate(idcs_airfoil):
            pred[i, :, idcs, :] = 0.0

        return pred
