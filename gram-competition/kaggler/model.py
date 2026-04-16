"""Model definitions.

Kept separate from train.py so predict.py can import without triggering
train.py's argv parsing.
"""

import torch
import torch.nn as nn

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


class ResidualMLP(nn.Module):
    """Predict residual from last-input velocity, with normalization and no-slip BC.

    Key ideas:
      * Normalize velocity_in by dataset stats to stabilize training.
      * Predict a per-timestep delta on top of velocity_in[:, -1] (strong prior).
      * Hard-zero the prediction at airfoil surface indices (no-slip).
    """

    def __init__(self, hidden=384, n_blocks=8, vel_mean=None, vel_std=None):
        super().__init__()
        in_dim = 3 + T_IN * 3  # pos(3) + normalized velocity_in(15)
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.float().view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.float().view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        x = torch.cat([pos, v_norm.reshape(B, N, T * C)], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        delta = self.proj_out(x).reshape(B, T_OUT, N, 3)
        last = velocity_in[:, -1:, :, :]
        pred = last + delta * self.vel_std
        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx, :] = 0.0
        return pred
