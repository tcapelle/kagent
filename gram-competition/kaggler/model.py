"""Model for GRaM airflow prediction.

Per-point ResMLP with:
  * residual prediction (delta from last input step, in normalized space)
  * multi-scale voxel-pooled neighbor velocities (spatial context)
  * signed distance to the airfoil surface (boundary-layer coord)
  * temporal velocity differences (Δv over input frames)
  * airfoil mask + global time conditioning
  * hard no-slip BC enforcement at airfoil indices.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from data import T_IN, T_OUT


VOXEL_SCALES = (0.05, 0.20)  # metres; F1 wing domain is ~2m x 0.8m x 1.2m


def voxel_pool_mean(pos, feats, voxel_size):
    """Per-voxel mean of `feats`, broadcast back to each input point.

    pos: [N, 3], feats: [N, C]. Returns [N, C].
    Voxel ids are built by integer-quantising pos at `voxel_size`.
    """
    grid = torch.floor(pos / voxel_size).long()  # [N,3]
    g_min = grid.min(dim=0).values
    grid = grid - g_min
    rng = grid.max(dim=0).values + 1
    key = grid[:, 0] * (rng[1] * rng[2]) + grid[:, 1] * rng[2] + grid[:, 2]
    _, inv = torch.unique(key, return_inverse=True)
    n_vox = int(inv.max().item() + 1)
    mean = scatter(feats, inv, dim=0, dim_size=n_vox, reduce="mean")
    return mean[inv]


def min_distance_to(pos, subset_pos, chunk=4096):
    """For each point in `pos`, distance to the nearest point in `subset_pos`.

    Returns [N] tensor. Chunked to keep memory bounded.
    """
    N = pos.shape[0]
    out = torch.empty(N, device=pos.device)
    for i in range(0, N, chunk):
        d = torch.cdist(pos[i : i + chunk], subset_pos)  # [c, M]
        out[i : i + chunk] = d.min(dim=1).values
    return out


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
    """Pointwise residual predictor with spatial + temporal context features."""

    # Feature dim breakdown:
    #   pos(3) + v_in_norm_flat(T_IN*3) + v_in_diff_flat((T_IN-1)*3)
    #   + airfoil_mask(1) + log_dist_airfoil(1)
    #   + voxel_mean_v per scale (len(VOXEL_SCALES) * 3)
    IN_DIM = 3 + T_IN * 3 + (T_IN - 1) * 3 + 1 + 1 + len(VOXEL_SCALES) * 3

    def __init__(self, vel_mean, vel_std, hidden=512, n_blocks=10):
        super().__init__()
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

        self.proj_in = nn.Linear(self.IN_DIM, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, T_OUT * 3))

        t_dim = T_IN + T_OUT + T_OUT
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

    def _build_features(self, velocity_in, pos, idcs_airfoil):
        B, T, N, _ = velocity_in.shape
        device = velocity_in.device

        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std      # [B,T,N,3]
        v_in_flat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * 3)
        v_in_diff = (v_in_norm[:, 1:] - v_in_norm[:, :-1])              # [B,T-1,N,3]
        v_in_diff_flat = v_in_diff.permute(0, 2, 1, 3).reshape(B, N, (T - 1) * 3)

        # Per-sample voxel features + distance-to-airfoil + mask.
        mask = torch.zeros(B, N, 1, device=device)
        log_dist = torch.zeros(B, N, 1, device=device)
        vox_feats_per_scale = [torch.zeros(B, N, 3, device=device) for _ in VOXEL_SCALES]

        v_last_norm = v_in_norm[:, -1]  # [B,N,3]
        for i in range(B):
            idc = idcs_airfoil[i].to(device)
            mask[i, idc, 0] = 1.0

            # Signed-distance-like feature to the nearest airfoil point.
            dist = min_distance_to(pos[i], pos[i, idc])
            log_dist[i, :, 0] = torch.log1p(dist * 10.0)

            # Multi-scale voxel-mean velocity (last input frame).
            for s, vs in enumerate(VOXEL_SCALES):
                vox_feats_per_scale[s][i] = voxel_pool_mean(pos[i], v_last_norm[i], vs)

        vox = torch.cat(vox_feats_per_scale, dim=-1)  # [B,N,3*len(scales)]
        feats = torch.cat(
            [pos, v_in_flat, v_in_diff_flat, mask, log_dist, vox],
            dim=-1,
        )
        return feats, v_in_norm

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, _ = velocity_in.shape
        device = velocity_in.device

        feats, v_in_norm = self._build_features(velocity_in, pos, idcs_airfoil)
        h = self.proj_in(feats)

        t_in, t_out = t[:, :T_IN], t[:, T_IN:]
        dt = t_out - t[:, T_IN - 1 : T_IN]
        t_feat = torch.cat([t_in, t_out, dt], dim=-1)
        h = h + self.t_mlp(t_feat).unsqueeze(1)

        for blk in self.blocks:
            h = blk(h)

        delta_norm = self.out(h).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3).contiguous()
        last_norm = v_in_norm[:, -1:].expand(-1, T_OUT, -1, -1)
        pred_norm = last_norm + delta_norm
        pred = pred_norm * self.vel_std + self.vel_mean

        for i, idc in enumerate(idcs_airfoil):
            pred[i, :, idc.to(device), :] = 0.0
        return pred


def build_from_checkpoint(state_dict, stats, device):
    hidden = state_dict["proj_in.weight"].shape[0]
    n_blocks = sum(
        1 for k in state_dict.keys()
        if k.startswith("blocks.") and k.endswith(".net.1.weight")
    )
    model = ResidualMLP(
        vel_mean=stats["vel_mean"],
        vel_std=stats["vel_std"],
        hidden=hidden,
        n_blocks=n_blocks,
    ).to(device)
    model.load_state_dict(state_dict)
    return model
