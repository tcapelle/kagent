"""Model definitions for CFD surrogate."""

import copy

import torch
import torch.nn as nn


class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


class ResMLP(nn.Module):
    def __init__(self, in_dim=24, hidden=256, n_blocks=8, out_dim=3, dropout=0.0):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden, dropout) for _ in range(n_blocks)])
        self.norm_out = nn.LayerNorm(hidden)
        # Separate heads for each output channel
        self.head_ux = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.head_uy = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.head_p = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, data, **kwargs):
        x = self.proj_in(data["x"])
        for block in self.blocks:
            x = block(x)
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
