"""Model architectures for GRAM airflow prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


@torch.no_grad()
def knn_edge_index(pos: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """Chunked KNN: returns edge_index [2, N*k] with rows (src, dst). Excludes self-loops."""
    N = pos.shape[0]
    idx_rows = []
    for i in range(0, N, chunk):
        d = torch.cdist(pos[i:i + chunk], pos)  # [c, N]
        _, idx = torch.topk(d, k + 1, dim=-1, largest=False)
        idx_rows.append(idx[:, 1:])  # drop self
    idx = torch.cat(idx_rows, dim=0)  # [N, k]
    dst = torch.arange(N, device=pos.device).repeat_interleave(k)
    src = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


class EdgeConvBlock(nn.Module):
    """Single KNN edge-conv block with max-aggregation.

    For each point i, message = MLP([x_i, x_j - x_i, rel_pos_ij]); aggregate max over j.
    """

    def __init__(self, dim, pos_dim=3, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.msg = nn.Sequential(
            nn.Linear(2 * dim + pos_dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x, pos, edge_index):
        # x: [N, D], pos: [N, 3], edge_index: [2, E] (src, dst)
        x_n = self.norm(x)
        src, dst = edge_index[0], edge_index[1]
        rel_pos = pos[src] - pos[dst]
        msg = torch.cat([x_n[dst], x_n[src] - x_n[dst], rel_pos], dim=-1)
        msg = self.msg(msg)
        # scatter max over dst
        out = torch.full((x.shape[0], x.shape[1]), float("-inf"), device=x.device, dtype=x.dtype)
        out = out.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), msg, reduce="amax", include_self=False)
        out = torch.where(out.isinf(), torch.zeros_like(out), out)
        return x + out


class FlowGNN(nn.Module):
    """ResMLP encoder + KNN EdgeConv spatial mixing + residual-to-time-mean head."""

    def __init__(self, vel_mean, vel_std, hidden=256, n_pre=2, n_gnn=3, n_post=2,
                 k=16, expand=4, prior: str = "mean"):
        super().__init__()
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        self.k = k
        self.prior = prior  # "mean" or "last"
        self._knn_cache: dict = {}

        in_dim = 3 + T_IN * 3
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.pre = nn.Sequential(*[ResBlock(hidden, expand) for _ in range(n_pre)])
        self.gnn = nn.ModuleList([EdgeConvBlock(hidden) for _ in range(n_gnn)])
        self.post = nn.Sequential(*[ResBlock(hidden, expand) for _ in range(n_post)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        nn.init.zeros_(self.proj_out[-1].bias)
        nn.init.normal_(self.proj_out[-1].weight, std=1e-3)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T_in, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_flat = v_norm.transpose(1, 2).reshape(B, N, T_in * C)

        x = torch.cat([pos, v_flat], dim=-1)
        x = self.proj_in(x)
        x = self.pre(x)

        xs = []
        for b in range(B):
            # cheap content fingerprint so we only compute KNN once per unique geometry
            fp = pos[b, ::1000].flatten()  # ~100 floats
            key = (N, float(fp.sum().item()), float(fp[:5].prod().item()))
            edge_index = self._knn_cache.get(key)
            if edge_index is None:
                edge_index = knn_edge_index(pos[b], k=self.k)
                self._knn_cache[key] = edge_index
            xb = x[b]
            for layer in self.gnn:
                xb = layer(xb, pos[b], edge_index)
            xs.append(xb)
        x = torch.stack(xs, dim=0)

        x = self.post(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, C).transpose(1, 2)
        prior_norm = v_norm.mean(dim=1, keepdim=True) if self.prior == "mean" else v_norm[:, -1:, :, :]
        out_norm = prior_norm + delta_norm
        out = out_norm * self.vel_std + self.vel_mean
        for b, idx in enumerate(idcs_airfoil):
            out[b, :, idx] = 0.0
        return out


class TransolverBlock(nn.Module):
    """Linear-attention-style block via learned soft slicing.

    Project N points to M slice tokens with softmax-normalized assignments,
    apply self-attention over slices, then scatter back to points.
    """

    def __init__(self, dim, n_slices=64, n_heads=4, expand=4):
        super().__init__()
        self.n_slices = n_slices
        self.n_heads = n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.to_w = nn.Linear(dim, n_slices * n_heads)
        self.to_v = nn.Linear(dim, dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.GELU(),
            nn.Linear(dim * expand, dim),
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        H, M = self.n_heads, self.n_slices
        Dh = D // H
        x_n = self.norm1(x)
        w = self.to_w(x_n).reshape(B, N, H, M).softmax(dim=1)       # N->M assignment per head
        v = self.to_v(x_n).reshape(B, N, H, Dh)
        slice_tok = torch.einsum("bnhm,bnhd->bmhd", w, v).reshape(B, M, D)  # [B, M, D]

        qkv = self.attn_qkv(slice_tok).reshape(B, M, 3, H, Dh).permute(2, 0, 3, 1, 4)  # [3, B, H, M, Dh]
        q, k, vv = qkv[0], qkv[1], qkv[2]
        a = torch.einsum("bhmd,bhnd->bhmn", q, k) / Dh ** 0.5
        a = a.softmax(dim=-1)
        out_slices = torch.einsum("bhmn,bhnd->bhmd", a, vv)
        out_slices = out_slices.permute(0, 2, 1, 3).reshape(B, M, D)
        out_slices = self.attn_proj(out_slices)                       # [B, M, D]

        # scatter back to points: for each point n, weighted sum over slices per head
        out_h = out_slices.reshape(B, M, H, Dh)
        back = torch.einsum("bnhm,bmhd->bnhd", w, out_h).reshape(B, N, D)
        x = x + back
        x = x + self.mlp(self.norm2(x))
        return x


class FlowTransolver(nn.Module):
    def __init__(self, vel_mean, vel_std, hidden=256, n_blocks=4, n_slices=64, n_heads=4, expand=4, prior="mean"):
        super().__init__()
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        self.prior = prior
        in_dim = 3 + T_IN * 3
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([TransolverBlock(hidden, n_slices, n_heads, expand) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.proj_out[-1].bias)
        nn.init.normal_(self.proj_out[-1].weight, std=1e-3)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T_in, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_flat = v_norm.transpose(1, 2).reshape(B, N, T_in * C)
        x = torch.cat([pos, v_flat], dim=-1)
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, C).transpose(1, 2)
        prior_norm = v_norm.mean(dim=1, keepdim=True) if self.prior == "mean" else v_norm[:, -1:, :, :]
        out_norm = prior_norm + delta_norm
        out = out_norm * self.vel_std + self.vel_mean
        for b, idx in enumerate(idcs_airfoil):
            out[b, :, idx] = 0.0
        return out


def build_model(stats, arch="mlp", **kw):
    if arch == "mlp":
        defaults = dict(hidden=384, n_blocks=8, expand=4, dropout=0.0)
        defaults.update(kw)
        return FlowMLP(vel_mean=stats["vel_mean"], vel_std=stats["vel_std"], **defaults)
    if arch == "gnn":
        defaults = dict(hidden=256, n_pre=2, n_gnn=3, n_post=2, k=16, expand=4)
        defaults.update(kw)
        return FlowGNN(vel_mean=stats["vel_mean"], vel_std=stats["vel_std"], **defaults)
    if arch == "transolver":
        defaults = dict(hidden=256, n_blocks=4, n_slices=64, n_heads=4, expand=4)
        defaults.update(kw)
        return FlowTransolver(vel_mean=stats["vel_mean"], vel_std=stats["vel_std"], **defaults)
    raise ValueError(f"unknown arch: {arch}")
