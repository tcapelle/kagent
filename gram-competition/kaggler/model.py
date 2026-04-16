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


class GlobalFuseBlock(nn.Module):
    """Mean+max pool across points, project, and broadcast back as residual.

    Gives each point access to a shared global context — crucial for flow
    problems where freestream conditions and global wake structure matter.
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):  # x: [B, N, C]
        h = self.norm(x)
        g_mean = h.mean(dim=1, keepdim=True)
        g_max = h.amax(dim=1, keepdim=True)
        g = self.proj(torch.cat([g_mean, g_max], dim=-1))  # [B, 1, C]
        return x + g.expand_as(x)


class ResidualMLP(nn.Module):
    """Predict residual from last-input velocity, with normalization and no-slip BC.

    Architecture:
      * Per-point input: pos(3) + v_norm(15) + is_airfoil(1) + dist_airfoil(1) = 20
      * ResBlock + GlobalFuseBlock interleaved for local MLP + global context
      * Output: delta scaled by vel_std added to velocity_in[:, -1]
      * Hard-zero at airfoil surface indices (no-slip)
    """

    def __init__(self, hidden=512, n_blocks=10, vel_mean=None, vel_std=None,
                 n_airfoil_probe=512):
        super().__init__()
        in_dim = 3 + T_IN * 3 + 2  # pos + v_norm + airfoil mask + dist
        out_dim = T_OUT * 3
        self.n_airfoil_probe = n_airfoil_probe
        self.proj_in = nn.Linear(in_dim, hidden)
        self.res_blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.global_blocks = nn.ModuleList([GlobalFuseBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.float().view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.float().view(1, 1, 1, 3))

    def _airfoil_features(self, pos, idcs_airfoil):
        """Per-point: [is_airfoil, dist_to_airfoil_surface]. pos: [B, N, 3]."""
        B, N, _ = pos.shape
        feats = torch.zeros(B, N, 2, device=pos.device, dtype=pos.dtype)
        for b, idx in enumerate(idcs_airfoil):
            feats[b, idx, 0] = 1.0
            # Probe subset of airfoil points for min-distance (fast enough).
            m = idx.numel()
            n_probe = min(self.n_airfoil_probe, m)
            stride = max(1, m // n_probe)
            probe_idx = idx[::stride][:n_probe]
            probes = pos[b, probe_idx]                                # [P, 3]
            d = torch.cdist(pos[b].unsqueeze(0), probes.unsqueeze(0)) # [1, N, P]
            feats[b, :, 1] = d.squeeze(0).min(dim=-1).values
        return feats

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        af = self._airfoil_features(pos, idcs_airfoil)  # [B, N, 2]
        x = torch.cat([pos, v_norm.reshape(B, N, T * C), af], dim=-1)
        x = self.proj_in(x)
        for rb, gb in zip(self.res_blocks, self.global_blocks):
            x = rb(x)
            x = gb(x)
        delta = self.proj_out(x).reshape(B, T_OUT, N, 3)
        last = velocity_in[:, -1:, :, :]
        pred = last + delta * self.vel_std
        for b, idx in enumerate(idcs_airfoil):
            pred[b, :, idx, :] = 0.0
        return pred
