"""Model definitions for GRAM airflow prediction."""

import torch
import torch.nn as nn
from data import T_IN, T_OUT
from torch_geometric.nn import knn_graph


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


class GraphConvBlock(nn.Module):
    """Simple edge-conditioned graph convolution using scatter_mean."""
    def __init__(self, dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim),  # source + target + rel_pos
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.node_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim * 2, dim * 4),  # node + aggregated msg
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x, edge_index, rel_pos):
        """
        x: [N_total, dim]
        edge_index: [2, E]
        rel_pos: [E, 3]
        """
        src, dst = edge_index
        # Compute edge messages
        edge_feat = torch.cat([x[src], x[dst], rel_pos], dim=-1)
        msg = self.edge_mlp(edge_feat)  # [E, dim]

        # Aggregate messages to destination nodes (mean)
        agg = torch.zeros_like(x)
        count = torch.zeros(x.shape[0], 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(dst.shape[0], 1, device=x.device))
        agg = agg / (count + 1e-8)

        # Update node
        out = self.node_mlp(torch.cat([x, agg], dim=-1))
        return x + out


class GNNModel(nn.Module):
    """
    Graph Neural Network for 3D airflow prediction.
    - k-NN graph on spatial coordinates
    - Message passing for spatial interaction
    - Residual prediction with no-slip BC
    """

    def __init__(self, hidden=256, n_mlp_blocks=4, n_gnn_blocks=4, n_freqs=64,
                 k_neighbors=16, vel_mean=None, vel_std=None):
        super().__init__()
        self.k = k_neighbors

        # Normalization stats
        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

        # Fourier position features
        self.pos_ff = FourierFeatures(3, n_freqs)
        pos_dim = 2 * n_freqs

        # Time embedding
        self.time_ff = FourierFeatures(1, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(64 * 10, 256),
            nn.GELU(),
            nn.Linear(256, hidden),
        )

        # Input: pos_fourier + pos(3) + vel_in(5*3) + temporal_feats(12)
        in_dim = pos_dim + 3 + T_IN * 3 + 12

        self.proj_in = nn.Linear(in_dim, hidden)

        # Pre-GNN MLP blocks (per-point processing)
        self.pre_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_mlp_blocks)])

        # GNN blocks (spatial message passing)
        self.gnn_blocks = nn.ModuleList([GraphConvBlock(hidden) for _ in range(n_gnn_blocks)])

        # Post-GNN MLP blocks
        self.post_blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_mlp_blocks)])

        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, T_OUT * 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        # Normalize velocity input
        v_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)

        # Per-point features
        pos_feat = self.pos_ff(pos)  # [B, N, 2*n_freqs]
        vel_flat = v_norm.reshape(B, N, T * C)
        last_vel = v_norm[:, -1]
        vel_mean = v_norm.mean(dim=1)
        vel_std_t = v_norm.std(dim=1)
        vel_delta = v_norm[:, -1] - v_norm[:, 0]

        x = torch.cat([pos, pos_feat, vel_flat, last_vel, vel_mean, vel_std_t, vel_delta], dim=-1)
        x = self.proj_in(x)  # [B, N, hidden]

        # Time conditioning
        t_feat = self.time_ff(t.unsqueeze(-1)).reshape(B, -1)
        t_cond = self.time_mlp(t_feat).unsqueeze(1)
        x = x + t_cond

        # Pre-GNN per-point processing
        x = self.pre_blocks(x)

        # Flatten batch for graph operations
        x_flat = x.reshape(B * N, -1)  # [B*N, hidden]
        pos_flat = pos.reshape(B * N, 3)

        # Build k-NN graph (batch-aware)
        batch_idx = torch.arange(B, device=x.device).repeat_interleave(N)
        edge_index = knn_graph(pos_flat, k=self.k, batch=batch_idx, loop=False)

        # Relative positions for edges
        src, dst = edge_index
        rel_pos = pos_flat[src] - pos_flat[dst]

        # GNN message passing
        for gnn in self.gnn_blocks:
            x_flat = gnn(x_flat, edge_index, rel_pos)

        # Unflatten
        x = x_flat.reshape(B, N, -1)

        # Post-GNN processing
        x = self.post_blocks(x)

        delta = self.proj_out(x).reshape(B, T_OUT, N, 3)

        # Residual: predict delta in normalized space, denormalize
        last_vel_raw = velocity_in[:, -1:]
        pred = last_vel_raw + delta * (self.vel_std.squeeze(1) + 1e-8)

        # No-slip boundary condition
        for i, idcs in enumerate(idcs_airfoil):
            pred[i, :, idcs, :] = 0.0

        return pred


class ResidualMLP(nn.Module):
    """Fallback MLP model (kept for compatibility)."""

    def __init__(self, hidden=512, n_blocks=10, n_freqs=128, dropout=0.0,
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

        in_dim = pos_dim + 3 + T_IN * 3 + 12
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks1 = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks // 2)])
        self.mid_norm = nn.LayerNorm(hidden)
        self.blocks2 = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks - n_blocks // 2)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, T_OUT * 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / (self.vel_std + 1e-8)
        pos_feat = self.pos_ff(pos)
        vel_flat = v_norm.reshape(B, N, T * C)
        last_vel = v_norm[:, -1]
        vel_mean = v_norm.mean(dim=1)
        vel_std_t = v_norm.std(dim=1)
        vel_delta = v_norm[:, -1] - v_norm[:, 0]
        x = torch.cat([pos, pos_feat, vel_flat, last_vel, vel_mean, vel_std_t, vel_delta], dim=-1)
        x = self.proj_in(x)
        t_feat = self.time_ff(t.unsqueeze(-1)).reshape(B, -1)
        t_cond = self.time_mlp(t_feat).unsqueeze(1)
        x = x + t_cond
        x = self.blocks1(x)
        x = self.mid_norm(x)
        x = self.blocks2(x)
        delta = self.proj_out(x).reshape(B, T_OUT, N, 3)
        last_vel_raw = velocity_in[:, -1:]
        pred = last_vel_raw + delta * (self.vel_std.squeeze(1) + 1e-8)
        for i, idcs in enumerate(idcs_airfoil):
            pred[i, :, idcs, :] = 0.0
        return pred
