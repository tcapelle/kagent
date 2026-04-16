"""Model architectures for GRAM airflow prediction."""

import torch
import torch.nn as nn

from data import T_IN, T_OUT


class ResBlock(nn.Module):
    def __init__(self, dim, expand=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class FlowMLP(nn.Module):
    """Pointwise ResMLP with physical priors.

    - normalizes velocity with dataset stats
    - predicts a delta off the last input timestep (residual / persistence prior)
    - enforces no-slip BC at airfoil surface points
    """

    def __init__(self, vel_mean, vel_std, hidden=384, n_blocks=8, expand=4, dropout=0.0):
        super().__init__()
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

        in_dim = 3 + T_IN * 3
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, expand, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        # small init so we start near the persistence baseline
        nn.init.zeros_(self.proj_out[-1].bias)
        nn.init.normal_(self.proj_out[-1].weight, std=1e-3)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T_in, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_flat = v_norm.transpose(1, 2).reshape(B, N, T_in * C)

        x = torch.cat([pos, v_flat], dim=-1)
        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x)
        delta_norm = delta_norm.reshape(B, N, T_OUT, C).transpose(1, 2)

        last_norm = v_norm[:, -1:, :, :]
        out_norm = last_norm + delta_norm
        out = out_norm * self.vel_std + self.vel_mean

        for b, idx in enumerate(idcs_airfoil):
            out[b, :, idx] = 0.0
        return out


def build_model(stats, hidden=384, n_blocks=8, expand=4, dropout=0.0):
    return FlowMLP(
        vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
        hidden=hidden, n_blocks=n_blocks, expand=expand, dropout=dropout,
    )
