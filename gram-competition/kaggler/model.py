"""Model for GRaM airflow prediction.

v4: per-point residual MLP with 4-scale voxel statistics (mean, std,
self-deviation) + within-voxel offset per scale, laplacian proxy
(coarse−fine), temporal Δv, airfoil mask, log-distance-to-airfoil,
global time conditioning, and hard no-slip BC enforcement.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from data import T_IN, T_OUT


VOXEL_SCALES = (0.03, 0.08, 0.20, 0.50)  # fine → coarse


def min_distance_to(pos, subset_pos, chunk=4096):
    """Per-point min distance from `pos` to any point in `subset_pos`. Chunked."""
    N = pos.shape[0]
    out = torch.empty(N, device=pos.device)
    for i in range(0, N, chunk):
        d = torch.cdist(pos[i : i + chunk], subset_pos)
        out[i : i + chunk] = d.min(dim=1).values
    return out


def voxel_pool_mean(pos, feats, voxel_size):
    """Per-voxel mean of `feats`, broadcast back to each point."""
    grid = torch.floor(pos / voxel_size).long()
    g_min = grid.min(dim=0).values
    grid = grid - g_min
    rng = grid.max(dim=0).values + 1
    key = grid[:, 0] * (rng[1] * rng[2]) + grid[:, 1] * rng[2] + grid[:, 2]
    _, inv = torch.unique(key, return_inverse=True)
    n_vox = int(inv.max().item() + 1)
    mean = scatter(feats, inv, dim=0, dim_size=n_vox, reduce="mean")
    return mean[inv]


def voxel_stats(pos, feats, voxel_size):
    """Returns (mean_at_point, std_at_point, self_dev_at_point, offset_at_point).

    mean_at[i]  = mean(feats) over the voxel containing point i.
    std_at[i]   = std(feats) over the voxel.
    self_dev[i] = feats[i] - mean_at[i].
    offset[i]   = pos[i] - mean_pos_of_voxel(i), normalized by voxel_size
                  (gives sub-voxel position in [-0.5, 0.5]-ish range).
    """
    grid = torch.floor(pos / voxel_size).long()
    g_min = grid.min(dim=0).values
    grid = grid - g_min
    rng = grid.max(dim=0).values + 1
    key = grid[:, 0] * (rng[1] * rng[2]) + grid[:, 1] * rng[2] + grid[:, 2]
    _, inv = torch.unique(key, return_inverse=True)
    n_vox = int(inv.max().item() + 1)

    m = scatter(feats, inv, dim=0, dim_size=n_vox, reduce="mean")
    m2 = scatter(feats * feats, inv, dim=0, dim_size=n_vox, reduce="mean")
    var = (m2 - m * m).clamp_min(0)
    std = var.sqrt()

    pos_mean = scatter(pos, inv, dim=0, dim_size=n_vox, reduce="mean")
    offset = (pos - pos_mean[inv]) / voxel_size

    mean_at = m[inv]
    std_at = std[inv]
    dev = feats - mean_at
    return mean_at, std_at, dev, offset


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
    # Per-point input features:
    #   pos(3) + v_in_norm_flat(T_IN*3) + v_in_diff_flat((T_IN-1)*3)
    #   + airfoil_mask(1) + log_dist_airfoil(1)
    #   + [mean(3), std(3), self-dev(3), offset(3)] per scale × len(VOXEL_SCALES)
    #   + laplacian(3)  (coarsest mean − finest mean)
    IN_DIM = 3 + T_IN * 3 + (T_IN - 1) * 3 + 1 + 1 + len(VOXEL_SCALES) * 12 + 3

    def __init__(self, vel_mean, vel_std, hidden=512, n_blocks=12):
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

        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std       # [B,T,N,3]
        v_in_flat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * 3)
        v_in_diff = v_in_norm[:, 1:] - v_in_norm[:, :-1]
        v_in_diff_flat = v_in_diff.permute(0, 2, 1, 3).reshape(B, N, (T - 1) * 3)

        mask = torch.zeros(B, N, 1, device=device)
        log_dist = torch.zeros(B, N, 1, device=device)
        per_scale_feats = [torch.zeros(B, N, 12, device=device) for _ in VOXEL_SCALES]
        laplacian = torch.zeros(B, N, 3, device=device)

        v_last = v_in_norm[:, -1]
        for i in range(B):
            idc = idcs_airfoil[i].to(device)
            mask[i, idc, 0] = 1.0

            dist = min_distance_to(pos[i], pos[i, idc])
            log_dist[i, :, 0] = torch.log1p(dist * 10.0)

            means = []
            for s, vs in enumerate(VOXEL_SCALES):
                mean_at, std_at, dev, offset = voxel_stats(pos[i], v_last[i], vs)
                per_scale_feats[s][i] = torch.cat([mean_at, std_at, dev, offset], dim=-1)
                means.append(mean_at)
            laplacian[i] = means[-1] - means[0]

        vox_cat = torch.cat(per_scale_feats, dim=-1)
        feats = torch.cat(
            [pos, v_in_flat, v_in_diff_flat, mask, log_dist, vox_cat, laplacian],
            dim=-1,
        )
        return feats, v_in_norm

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, _, N, _ = velocity_in.shape
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
