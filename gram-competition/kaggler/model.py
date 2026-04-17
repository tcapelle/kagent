"""Model for GRaM airflow prediction.

v5: per-point residual MLP with multi-scale voxel input features (mean,
std, self-dev, offset) + iterative voxel-mix message passing between
ResBlocks. Each block does: ResBlock → multi-scale scatter-mean →
per-scale zero-init linear on voxel features → gather back. Gives
effective receptive field that grows with depth, unlike v4 where
spatial context was injected only once at the input.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from data import T_IN, T_OUT


VOXEL_SCALES = (0.03, 0.08, 0.20, 0.50)         # input feature scales
VOXEL_MIX_SCALES = (0.12,)                        # iterative mix scales
MIX_EVERY = 2                                      # apply mix every N blocks


def min_distance_to(pos, subset_pos, chunk=4096):
    N = pos.shape[0]
    out = torch.empty(N, device=pos.device)
    for i in range(0, N, chunk):
        d = torch.cdist(pos[i : i + chunk], subset_pos)
        out[i : i + chunk] = d.min(dim=1).values
    return out


def voxel_inv(pos, voxel_size):
    """Return (inv [N], n_vox) for a single-sample pos [N, 3]."""
    grid = torch.floor(pos / voxel_size).long()
    g_min = grid.min(dim=0).values
    grid = grid - g_min
    rng = grid.max(dim=0).values + 1
    key = grid[:, 0] * (rng[1] * rng[2]) + grid[:, 1] * rng[2] + grid[:, 2]
    _, inv = torch.unique(key, return_inverse=True)
    return inv, int(inv.max().item() + 1)


def voxel_stats(pos, feats, voxel_size):
    """Returns (mean, std, self_dev, offset) broadcast to each point.

    offset[i] = (pos[i] − voxel_centroid) / voxel_size   (sub-voxel position).
    """
    inv, n_vox = voxel_inv(pos, voxel_size)
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


import torch.nn.functional as F


class ResBlock(nn.Module):
    # Dropout applied functionally so state_dict keys match v6 checkpoints
    # (enables loading old weights and ensembling). dropout_p=0 disables.
    def __init__(self, dim, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        # Split net into pre/post; drop between them only during training
        h = self.net[0](x)          # LN
        h = self.net[1](h)          # Linear(d, 2d)
        h = self.net[2](h)          # GELU
        if self.training and self.dropout_p > 0:
            h = F.dropout(h, p=self.dropout_p, training=True)
        h = self.net[3](h)          # Linear(2d, d)
        return x + h


class VoxelMix(nn.Module):
    """Lightweight scatter-mean message passing with LN + bounded gate.

    m = LN(scatter_mean(h, inv));   out = tanh(gate) ⊙ gather(m).
    tanh bounds the per-channel scale so iterative mixing can't blow up.
    """
    def __init__(self, dim, n_scales):
        super().__init__()
        self.lns = nn.ModuleList([nn.LayerNorm(dim) for _ in range(n_scales)])
        self.gate = nn.Parameter(torch.zeros(n_scales, dim))

    def forward(self, h, invs_per_scale):
        B = h.shape[0]
        out = torch.zeros_like(h)
        for s_idx, per_batch in enumerate(invs_per_scale):
            g = torch.tanh(self.gate[s_idx])
            ln = self.lns[s_idx]
            for b in range(B):
                inv, n_vox = per_batch[b]
                m = scatter(h[b], inv, dim=0, dim_size=n_vox, reduce="mean")
                m = ln(m)
                out[b] = out[b] + g * m[inv]
        return out


# ---------------------------------------------------------------------------
# v16 additions: Fourier positional embedding + voxel-token global attention
# ---------------------------------------------------------------------------

FOURIER_FREQS = 6                 # NeRF-style, 2^0..2^5 = 1..32
VOXEL_ATTN_SCALE = 0.15           # voxel size for attention tokens (~200-800 tokens)


class FourierPosEmbed(nn.Module):
    """Maps pos [..., 3] → [..., 3*(1 + 2*K)] via sin/cos at K frequencies.

    Standard NeRF-style positional encoding. Unlike a learned Linear(3→d)
    projection, this gives the network exposure to many spatial frequencies
    at the input, which helps modeling of sharp turbulent features.
    """
    def __init__(self, n_freqs=FOURIER_FREQS):
        super().__init__()
        freqs = 2.0 ** torch.arange(n_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim_mult(self):
        return 1 + 2 * self.freqs.numel()

    def forward(self, x):
        # x: [..., 3]
        angles = x.unsqueeze(-1) * self.freqs  # [..., 3, K]
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [..., 3, 2K]
        emb = emb.flatten(-2)  # [..., 3*2K]
        return torch.cat([x, emb], dim=-1)


class VoxelTokenAttn(nn.Module):
    """Global self-attention across voxel tokens with LayerScale residual.

    Pool point features → voxel tokens (scatter_mean + voxel-centroid
    positional encoding), multi-head self-attention among tokens,
    scatter back to points via gather. Residual is LayerScaled (CaiT-style)
    so the block starts as near-identity and can't destabilize early
    training the way v8's tanh-gated residual did.
    """
    def __init__(self, dim, n_heads=4, layer_scale=1e-4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.pos_mlp = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, h, pos, invs_and_centroids):
        # h: [B, N, d], pos: [B, N, 3]
        # invs_and_centroids: list[(inv, n_vox, centroids)] per batch sample
        B, N, d = h.shape
        out = torch.zeros_like(h)
        h_norm = self.norm(h)
        for b in range(B):
            inv, n_vox, centroids = invs_and_centroids[b]
            pooled = scatter(h_norm[b], inv, dim=0, dim_size=n_vox, reduce="mean")
            tokens = pooled + self.pos_mlp(centroids)  # [n_vox, d]
            qkv = self.qkv(tokens)  # [n_vox, 3d]
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(n_vox, self.n_heads, self.head_dim).transpose(0, 1)  # [H, n_vox, hd]
            k = k.view(n_vox, self.n_heads, self.head_dim).transpose(0, 1)
            v = v.view(n_vox, self.n_heads, self.head_dim).transpose(0, 1)
            attended = F.scaled_dot_product_attention(q, k, v)  # [H, n_vox, hd]
            attended = attended.transpose(0, 1).reshape(n_vox, d)  # [n_vox, d]
            broadcast = attended[inv]  # [N, d]
            out[b] = self.gamma * self.out(broadcast)
        return h + out


class ResidualMLP(nn.Module):
    # Per-point input features:
    #   pos(3) + v_in_norm_flat(T_IN*3) + v_in_diff_flat((T_IN-1)*3)
    #   + airfoil_mask(1) + log_dist_airfoil(1)
    #   + [mean(3), std(3), dev(3), offset(3)] per scale × len(VOXEL_SCALES)
    #   + laplacian(3)
    IN_DIM = 3 + T_IN * 3 + (T_IN - 1) * 3 + 1 + 1 + len(VOXEL_SCALES) * 12 + 3

    def __init__(self, vel_mean, vel_std, hidden=512, n_blocks=12, dropout_p=0.0):
        super().__init__()
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

        self.proj_in = nn.Linear(self.IN_DIM, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout_p=dropout_p) for _ in range(n_blocks)])
        # One mix per every MIX_EVERY blocks
        self.n_mixes = n_blocks // MIX_EVERY
        self.mixes = nn.ModuleList(
            [VoxelMix(hidden, len(VOXEL_MIX_SCALES)) for _ in range(self.n_mixes)]
        )
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

        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std
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

        # Precompute voxel indices for mix scales (once per forward)
        mix_invs = [[voxel_inv(pos[b], vs) for b in range(B)] for vs in VOXEL_MIX_SCALES]

        feats, v_in_norm = self._build_features(velocity_in, pos, idcs_airfoil)
        h = self.proj_in(feats)

        t_in, t_out = t[:, :T_IN], t[:, T_IN:]
        dt = t_out - t[:, T_IN - 1 : T_IN]
        t_feat = torch.cat([t_in, t_out, dt], dim=-1)
        h = h + self.t_mlp(t_feat).unsqueeze(1)

        mix_iter = iter(self.mixes)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if (i + 1) % MIX_EVERY == 0:
                h = h + next(mix_iter)(h, mix_invs)

        delta_norm = self.out(h).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3).contiguous()
        last_norm = v_in_norm[:, -1:].expand(-1, T_OUT, -1, -1)
        pred_norm = last_norm + delta_norm
        pred = pred_norm * self.vel_std + self.vel_mean

        for i, idc in enumerate(idcs_airfoil):
            pred[i, :, idc.to(device), :] = 0.0
        return pred


class ResidualMLPv16(nn.Module):
    """v16: v6 architecture + Fourier pos embedding + voxel-token attention.

    Changes vs ResidualMLP (v6):
      - Input replaces raw pos (3) with Fourier pos embed (3 + 3*2*K = 39 dims for K=6).
      - Adds N_ATTN_BLOCKS VoxelTokenAttn modules distributed through depth.
        Each attends globally across ~200-800 voxel tokens (scale 0.15) and
        uses LayerScale residual so the block is near-identity at init.

    Distinct state_dict keys (`fourier_pos`, `attns.*`, `proj_in` different
    in_features) let `build_from_checkpoint` dispatch to the right class.
    """
    IN_DIM_V16 = 3 * (1 + 2 * FOURIER_FREQS) + T_IN * 3 + (T_IN - 1) * 3 + 1 + 1 + len(VOXEL_SCALES) * 12 + 3

    def __init__(self, vel_mean, vel_std, hidden=512, n_blocks=12,
                 dropout_p=0.0, n_attn_blocks=2, layer_scale=1e-4):
        super().__init__()
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

        self.fourier_pos = FourierPosEmbed(FOURIER_FREQS)
        self.proj_in = nn.Linear(self.IN_DIM_V16, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout_p=dropout_p) for _ in range(n_blocks)])
        self.n_mixes = n_blocks // MIX_EVERY
        self.mixes = nn.ModuleList(
            [VoxelMix(hidden, len(VOXEL_MIX_SCALES)) for _ in range(self.n_mixes)]
        )
        # Attention blocks inserted at evenly-spaced depths
        self.n_attn_blocks = n_attn_blocks
        self.attns = nn.ModuleList(
            [VoxelTokenAttn(hidden, n_heads=4, layer_scale=layer_scale)
             for _ in range(n_attn_blocks)]
        )
        # Depths at which to run attention (1-indexed block positions)
        step = max(1, n_blocks // (n_attn_blocks + 1))
        self.attn_after_block = [(i + 1) * step for i in range(n_attn_blocks)]

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

        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std
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
        pos_embed = self.fourier_pos(pos)
        feats = torch.cat(
            [pos_embed, v_in_flat, v_in_diff_flat, mask, log_dist, vox_cat, laplacian],
            dim=-1,
        )
        return feats, v_in_norm

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, _, N, _ = velocity_in.shape
        device = velocity_in.device

        # Precompute voxel indices for mix scales (once per forward)
        mix_invs = [[voxel_inv(pos[b], vs) for b in range(B)] for vs in VOXEL_MIX_SCALES]
        # Precompute voxel indices + centroids for attention
        attn_invs = []
        for b in range(B):
            inv_b, n_vox_b = voxel_inv(pos[b], VOXEL_ATTN_SCALE)
            centroids = scatter(pos[b], inv_b, dim=0, dim_size=n_vox_b, reduce="mean")
            attn_invs.append((inv_b, n_vox_b, centroids))

        feats, v_in_norm = self._build_features(velocity_in, pos, idcs_airfoil)
        h = self.proj_in(feats)

        t_in, t_out = t[:, :T_IN], t[:, T_IN:]
        dt = t_out - t[:, T_IN - 1 : T_IN]
        t_feat = torch.cat([t_in, t_out, dt], dim=-1)
        h = h + self.t_mlp(t_feat).unsqueeze(1)

        mix_iter = iter(self.mixes)
        attn_schedule = set(self.attn_after_block)
        attn_iter = iter(self.attns)
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if (i + 1) % MIX_EVERY == 0:
                h = h + next(mix_iter)(h, mix_invs)
            if (i + 1) in attn_schedule:
                h = next(attn_iter)(h, pos, attn_invs)

        delta_norm = self.out(h).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3).contiguous()
        last_norm = v_in_norm[:, -1:].expand(-1, T_OUT, -1, -1)
        pred_norm = last_norm + delta_norm
        pred = pred_norm * self.vel_std + self.vel_mean

        for i, idc in enumerate(idcs_airfoil):
            pred[i, :, idc.to(device), :] = 0.0
        return pred


def build_from_checkpoint(state_dict, stats, device):
    # Dispatch by presence of v16-only keys in state_dict
    is_v16 = any(k.startswith("attns.") for k in state_dict.keys())
    hidden = state_dict["proj_in.weight"].shape[0]
    n_blocks = sum(
        1 for k in state_dict.keys()
        if k.startswith("blocks.") and k.endswith(".net.1.weight")
    )
    if is_v16:
        n_attn_blocks = sum(
            1 for k in state_dict.keys()
            if k.startswith("attns.") and k.endswith(".qkv.weight")
        )
        model = ResidualMLPv16(
            vel_mean=stats["vel_mean"],
            vel_std=stats["vel_std"],
            hidden=hidden,
            n_blocks=n_blocks,
            n_attn_blocks=n_attn_blocks,
        ).to(device)
    else:
        model = ResidualMLP(
            vel_mean=stats["vel_mean"],
            vel_std=stats["vel_std"],
            hidden=hidden,
            n_blocks=n_blocks,
        ).to(device)
    model.load_state_dict(state_dict)
    return model
