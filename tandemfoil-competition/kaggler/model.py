"""Transolver model — shared between train.py and predict.py."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True):
        super().__init__()
        act_fn = ACTIVATION[act]
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            [nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn()) for _ in range(n_layers)]
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            x = self.linears[i](x) + x if self.res else self.linears[i](x)
        return self.linear_post(x)


class PhysicsAttention(nn.Module):
    """Physics-aware attention for irregular meshes."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        B, N, _ = x.shape

        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        out_slice = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        )

        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act="gelu",
                 mlp_ratio=4, last_layer=False, out_dim=1, slice_num=32,
                 film_dim: int = 0):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = PhysicsAttention(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim,
                        n_layers=0, res=False, act=act)
        if film_dim > 0:
            # AdaLN-style FiLM: per-block gamma/beta for each LN, conditioned on
            # a global scalar (log Re here). Zero-init so the warm-started model
            # behaves identically until SGD learns to use the conditioning.
            self.film = nn.Linear(film_dim, 4 * hidden_dim)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, fx, cond=None):
        if cond is not None and hasattr(self, "film"):
            g1, b1, g2, b2 = self.film(cond).chunk(4, dim=-1)
            # cond shape: [B, film_dim]; reshape to [B, 1, hidden_dim] to broadcast over N.
            g1, b1, g2, b2 = g1.unsqueeze(1), b1.unsqueeze(1), g2.unsqueeze(1), b2.unsqueeze(1)
            fx = self.attn(self.ln_1(fx) * (1 + g1) + b1) + fx
            fx = self.mlp(self.ln_2(fx) * (1 + g2) + b2) + fx
        else:
            fx = self.attn(self.ln_1(fx)) + fx
            fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class Transolver(nn.Module):
    def __init__(self, space_dim=1, n_layers=5, n_hidden=256, dropout=0.0,
                 n_head=8, act="gelu", mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False,
                 fourier_dim: int = 0, fourier_sigma: float = 5.0,
                 film_re: bool = False, film_emb_dim: int = 64,
                 output_fields: list[str] | None = None,
                 output_dims: list[int] | None = None):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.fourier_dim = fourier_dim
        self.film_re = film_re
        self.output_fields = output_fields or []
        self.output_dims = output_dims or []

        # Fixed random Fourier-feature frequencies on spatial coords.
        # 2 cols (x, z) → fourier_dim freqs → 2*fourier_dim sin/cos features.
        if fourier_dim > 0:
            B = torch.randn(2, fourier_dim) * fourier_sigma
            self.register_buffer("fourier_B", B)
            extra = 2 * fourier_dim
        else:
            extra = 0

        if self.unified_pos:
            in_dim = fun_dim + ref**3 + extra
        else:
            in_dim = fun_dim + space_dim + extra
        self.preprocess = MLP(in_dim, n_hidden * 2, n_hidden,
                               n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        film_dim = film_emb_dim if film_re else 0
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout,
                act=act, mlp_ratio=mlp_ratio, out_dim=out_dim,
                slice_num=slice_num, last_layer=(i == n_layers - 1),
                film_dim=film_dim,
            )
            for i in range(n_layers)
        ])
        self.placeholder = nn.Parameter((1 / n_hidden) * torch.rand(n_hidden))
        if film_re:
            # Tiny MLP that maps log(Re) → film_emb_dim. Used as global cond per sample.
            self.re_embed = nn.Sequential(
                nn.Linear(1, film_emb_dim), nn.GELU(),
                nn.Linear(film_emb_dim, film_emb_dim),
            )
        self.apply(self._init_weights)
        # Re-zero the FiLM linears AFTER apply() (which would normally trunc-init them).
        if film_re:
            for blk in self.blocks:
                if hasattr(blk, "film"):
                    nn.init.zeros_(blk.film.weight)
                    nn.init.zeros_(blk.film.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data, **kwargs):
        x = data["x"]
        cond = None
        if self.film_re:
            # log(Re) is a sample-constant column at index 13 — take the first node.
            re = x[:, 0:1, 13]  # [B, 1]
            cond = self.re_embed(re)  # [B, film_emb_dim]
        if self.fourier_dim > 0:
            ff = x[..., :2] @ self.fourier_B  # [B, N, fourier_dim]
            ff = 2 * math.pi * ff
            x = torch.cat([x, torch.sin(ff), torch.cos(ff)], dim=-1)
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            fx = block(fx, cond=cond)
        return {"preds": fx}
