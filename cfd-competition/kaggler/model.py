"""Model definitions for CFD surrogate."""

import copy

import torch
import torch.nn as nn


class FiLMResMLPBlock(nn.Module):
    """ResMLPBlock with FiLM conditioning from global features."""
    def __init__(self, dim, cond_dim, expansion=2, dropout=0.0):
        super().__init__()
        inner = dim * expansion
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # FiLM: condition produces scale and shift
        self.film = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        gamma_beta = self.film(cond)
        if gamma_beta.dim() == 2:
            gamma_beta = gamma_beta.unsqueeze(1)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = h * (1 + gamma) + beta
        return x + self.drop(self.fc2(self.act(self.fc1(h))))


class ResMLP(nn.Module):
    def __init__(self, in_dim=24, hidden=256, n_blocks=12, out_dim=3, expansion=2,
                 dropout=0.0, local_dim=13, global_dim=11):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.proj_in = nn.Linear(in_dim, hidden)
        cond_hidden = 64
        self.cond_encoder = nn.Sequential(
            nn.Linear(global_dim, cond_hidden), nn.GELU(),
            nn.Linear(cond_hidden, cond_hidden), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FiLMResMLPBlock(hidden, cond_hidden, expansion, dropout) for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(hidden)
        # Separate heads for each output channel
        self.head_ux = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.head_uy = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.head_p = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, data, **kwargs):
        x_full = data["x"]
        cond = self.cond_encoder(x_full[:, 0, self.local_dim:])
        x = self.proj_in(x_full)
        for block in self.blocks:
            x = block(x, cond)
        x = self.norm_out(x)
        ux = self.head_ux(x)
        uy = self.head_uy(x)
        p = self.head_p(x)
        return {"preds": torch.cat([ux, uy, p], dim=-1)}


class EMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
