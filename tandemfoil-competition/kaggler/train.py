"""ResMLP surrogate for TandemFoilSet.

Per-node MLP with residual blocks and Fourier features on position. Inputs
already encode geometry context (saf + dsdf), so a point-wise ResMLP can
capture much of the field structure without explicit neighbour aggregation.

Run:
  python train.py --agent <name> --wandb_name "<name>/<desc>"
"""

import os
import time
import random
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from einops import rearrange
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from viz import visualize


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    """Random Gaussian Fourier features on 2D positions."""
    def __init__(self, n_freqs: int = 16, sigma: float = 5.0):
        super().__init__()
        B = torch.randn(2, n_freqs) * sigma
        self.register_buffer("B", B)

    def forward(self, pos):
        proj = pos @ self.B
        return torch.cat([torch.sin(proj * 2 * torch.pi),
                          torch.cos(proj * 2 * torch.pi)], dim=-1)


class MultiScaleFourier(nn.Module):
    """Multi-scale sin/cos features on 2D position (Edward's pattern).

    Output dim = 2 (sin/cos) * 2 (axes x,z) * num_scales.
    """
    def __init__(self, num_scales: int = 8, max_freq: float = 16.0):
        super().__init__()
        freqs = 2 ** torch.linspace(0.0, float(np.log2(max_freq)), num_scales)
        self.register_buffer("freqs", freqs)

    def out_dim(self):
        return 2 * 2 * self.freqs.numel()

    def forward(self, pos):
        x = pos.unsqueeze(-1) * self.freqs * torch.pi
        return torch.cat([x.sin(), x.cos()], dim=-1).flatten(-2)


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class GlobalFiLM(nn.Module):
    """Mean-pool a global descriptor over masked nodes, regress (γ, β) to
    modulate per-node features. Initialised to identity so adding to a
    pre-trained network is a no-op at init time.
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        h = hidden or 2 * dim
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, h), nn.GELU(),
            nn.Linear(h, 2 * dim),
        )
        # Zero-init the last layer → γ=0, β=0 → output = x at init.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, mask=None):
        if mask is None:
            g = x.mean(dim=1)
        else:
            mf = mask.unsqueeze(-1).to(x.dtype)
            g = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
        g = self.norm(g)
        gb = self.mlp(g)  # [B, 2D]
        gamma, beta = gb.chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class SliceAttention(nn.Module):
    """Transolver-style physics attention via learned slice tokens.

    Each node is softly assigned to S slices; we self-attend on slice tokens,
    then redistribute back. Cost is O(N*S + S^2) instead of O(N^2).
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 32, slice_num: int = 64):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_x = nn.Linear(dim, inner)
        self.in_project_fx = nn.Linear(dim, inner)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Linear(inner, dim)

    def forward(self, x):
        B, N, _ = x.shape
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()
        slice_w = (self.in_project_slice(x_mid) / self.temperature).softmax(dim=-1)  # [B, h, N, S]
        slice_n = slice_w.sum(dim=2)  # [B, h, S]
        slice_t = torch.einsum("bhnc,bhns->bhsc", fx_mid, slice_w)
        slice_t = slice_t / (slice_n.unsqueeze(-1) + 1e-5)
        q = self.to_q(slice_t)
        k = self.to_k(slice_t)
        v = self.to_v(slice_t)
        out_s = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out_x = torch.einsum("bhsc,bhns->bhnc", out_s, slice_w)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class TransolverBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, slice_num: int,
                 mlp_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = SliceAttention(dim, heads, dim_head, slice_num)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransolverNet(nn.Module):
    def __init__(self, in_dim: int = 24, hidden: int = 256, n_layers: int = 8,
                 heads: int = 8, dim_head: int = 32, slice_num: int = 64,
                 mlp_ratio: int = 2, out_dim: int = 3, dropout: float = 0.0,
                 n_freqs: int = 32, fourier_sigma: float = 4.0):
        super().__init__()
        self.fourier = FourierFeatures(n_freqs=n_freqs, sigma=fourier_sigma)
        embed_in = in_dim + 2 * n_freqs
        self.embed = nn.Sequential(nn.Linear(embed_in, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([
            TransolverBlock(hidden, heads, dim_head, slice_num, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, data):
        x = data["x"]
        ff = self.fourier(x[..., :2])
        h = torch.cat([x, ff], dim=-1)
        h = self.embed(h)
        for block in self.blocks:
            h = block(h)
        return {"preds": self.head(h)}


class ResMLP(nn.Module):
    """Per-node MLP with residual blocks.

    If `use_global=True`, every block is followed by a zero-init `GlobalFiLM`
    that injects a sample-level mean-pooled descriptor via FiLM modulation.
    Zero-init means the layer is identity at construction, so warm-starting
    from a checkpoint without these layers (strict=False) is safe.
    """

    def __init__(self, in_dim: int = 24, hidden: int = 512, n_blocks: int = 8,
                 out_dim: int = 3, expansion: int = 4, dropout: float = 0.0,
                 n_freqs: int = 32, fourier_sigma: float = 4.0,
                 ff_kind: str = "random",
                 fourier_scales: int = 8, fourier_max_freq: float = 16.0,
                 use_global: bool = False):
        super().__init__()
        self.ff_kind = ff_kind
        self.use_global = use_global
        if ff_kind == "multiscale":
            self.fourier = MultiScaleFourier(num_scales=fourier_scales,
                                             max_freq=fourier_max_freq)
            embed_in = in_dim + self.fourier.out_dim()
        else:
            self.fourier = FourierFeatures(n_freqs=n_freqs, sigma=fourier_sigma)
            embed_in = in_dim + 2 * n_freqs
        self.embed = nn.Sequential(nn.Linear(embed_in, hidden), nn.GELU())
        self.blocks = nn.ModuleList([
            ResMLPBlock(hidden, expansion, dropout) for _ in range(n_blocks)
        ])
        if use_global:
            self.globals = nn.ModuleList([GlobalFiLM(hidden) for _ in range(n_blocks)])
        else:
            self.globals = None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, data):
        x = data["x"]
        mask = data.get("mask")
        ff = self.fourier(x[..., :2])
        h = torch.cat([x, ff], dim=-1)
        h = self.embed(h)
        for i, block in enumerate(self.blocks):
            h = block(h)
            if self.globals is not None:
                h = self.globals[i](h, mask)
        return {"preds": self.head(h)}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    surf_weight: float = 10.0
    p_weight: float = 3.0     # extra weighting on pressure channel in loss
    loss_type: str = "l1"     # "l1" or "l2"
    epochs: int = 60
    warmup_steps: int = 300
    train_subsample: int = 40000
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    warm_start: str | None = None
    # model
    arch: str = "resmlp"  # "transolver" or "resmlp"
    hidden: int = 384
    n_layers: int = 6
    heads: int = 8
    dim_head: int = 40
    slice_num: int = 64
    mlp_ratio: int = 4
    dropout: float = 0.0
    # Fourier features
    ff_kind: str = "multiscale"   # "multiscale" or "random"
    fourier_scales: int = 8
    fourier_max_freq: float = 16.0
    n_freqs: int = 32
    fourier_sigma: float = 4.0
    # bf16 autocast
    amp: bool = True
    # validate every N epochs (after epoch 1)
    val_every: int = 2
    # cosine LR target ~= this many epochs
    cosine_epochs: int = 60
    # EMA
    ema_decay: float = 0.9995
    ema_start_step: int = 500
    # Global FiLM modulation per block (zero-init, safe to warm-start)
    use_global: bool = False


def subsample_batch(x, y, is_surface, mask, k_vol):
    """Keep all surface nodes; randomly sample k_vol volume nodes per sample."""
    B, _, D = x.shape
    new_x_list, new_y_list, new_s_list = [], [], []
    max_len = 0
    for b in range(B):
        surf_idx = (mask[b] & is_surface[b]).nonzero(as_tuple=False).squeeze(-1)
        vol_idx = (mask[b] & ~is_surface[b]).nonzero(as_tuple=False).squeeze(-1)
        if vol_idx.numel() > k_vol:
            perm = torch.randperm(vol_idx.numel(), device=vol_idx.device)[:k_vol]
            vol_idx = vol_idx[perm]
        keep = torch.cat([surf_idx, vol_idx])
        new_x_list.append(x[b, keep])
        new_y_list.append(y[b, keep])
        new_s_list.append(is_surface[b, keep])
        max_len = max(max_len, keep.numel())

    out_x = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
    out_y = torch.zeros(B, max_len, y.shape[-1], device=y.device, dtype=y.dtype)
    out_s = torch.zeros(B, max_len, dtype=torch.bool, device=is_surface.device)
    out_m = torch.zeros(B, max_len, dtype=torch.bool, device=mask.device)
    for b in range(B):
        n = new_x_list[b].shape[0]
        out_x[b, :n] = new_x_list[b]
        out_y[b, :n] = new_y_list[b]
        out_s[b, :n] = new_s_list[b]
        out_m[b, :n] = True
    return out_x, out_y, out_s, out_m


def _validate(model, val_loaders, stats, device, surf_weight, amp=False):
    model.eval()
    val_loss_sum = 0.0
    surf_p_sum = 0.0
    split_metrics: dict[str, dict] = {}
    amp_dtype = torch.bfloat16 if amp else torch.float32

    for split_name, vloader in val_loaders.items():
        val_vol = val_surf = 0.0
        mae_surf = torch.zeros(3, device=device)
        mae_vol = torch.zeros(3, device=device)
        n_surf = n_vol = n_vb = 0

        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x = (x - stats["x_mean"]) / stats["x_std"]
                y_norm = (y - stats["y_mean"]) / stats["y_std"]

                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
                    pred = model({"x": x, "mask": mask})["preds"]
                pred = pred.float()
                sq_err = (pred - y_norm) ** 2

                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                val_vol += (sq_err * vol_mask.unsqueeze(-1)).sum().item() / vol_mask.sum().clamp(min=1).item()
                val_surf += (sq_err * surf_mask.unsqueeze(-1)).sum().item() / surf_mask.sum().clamp(min=1).item()
                n_vb += 1

                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
                n_vol += vol_mask.sum().item()

        val_vol /= max(n_vb, 1)
        val_surf /= max(n_vb, 1)
        split_loss = val_vol + surf_weight * val_surf
        mae_surf /= max(n_surf, 1)
        mae_vol /= max(n_vol, 1)

        split_metrics[split_name] = {
            f"{split_name}/vol_loss": val_vol,
            f"{split_name}/surf_loss": val_surf,
            f"{split_name}/loss": split_loss,
            f"{split_name}/mae_vol_Ux": mae_vol[0].item(),
            f"{split_name}/mae_vol_Uy": mae_vol[1].item(),
            f"{split_name}/mae_vol_p": mae_vol[2].item(),
            f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
            f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
            f"{split_name}/mae_surf_p": mae_surf[2].item(),
        }
        val_loss_sum += split_loss
        surf_p_sum += mae_surf[2].item()

    return val_loss_sum / len(val_loaders), surf_p_sum / len(val_loaders), split_metrics


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats, sample_weights = load_data(cfg.splits_dir, debug=cfg.debug)
    stats = {k: v.to(device) for k, v in stats.items()}

    loader_kwargs = dict(collate_fn=pad_collate, num_workers=4, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2)

    if cfg.debug:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, **loader_kwargs)
    else:
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  sampler=sampler, **loader_kwargs)

    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    if cfg.arch == "transolver":
        model_config = dict(
            arch="transolver",
            in_dim=X_DIM,
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            heads=cfg.heads,
            dim_head=cfg.dim_head,
            slice_num=cfg.slice_num,
            mlp_ratio=cfg.mlp_ratio,
            out_dim=3,
            dropout=cfg.dropout,
            n_freqs=cfg.n_freqs,
            fourier_sigma=cfg.fourier_sigma,
        )
        model_kwargs = {k: v for k, v in model_config.items() if k != "arch"}
        model = TransolverNet(**model_kwargs).to(device)
    else:
        model_config = dict(
            arch="resmlp",
            in_dim=X_DIM,
            hidden=cfg.hidden,
            n_blocks=cfg.n_layers,
            out_dim=3,
            expansion=cfg.mlp_ratio,
            dropout=cfg.dropout,
            ff_kind=cfg.ff_kind,
            fourier_scales=cfg.fourier_scales,
            fourier_max_freq=cfg.fourier_max_freq,
            n_freqs=cfg.n_freqs,
            fourier_sigma=cfg.fourier_sigma,
            use_global=cfg.use_global,
        )
        model_kwargs = {k: v for k, v in model_config.items() if k != "arch"}
        model = ResMLP(**model_kwargs).to(device)

    if cfg.warm_start:
        state = torch.load(cfg.warm_start, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Warm-started from {cfg.warm_start} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * (cfg.cosine_epochs if not cfg.debug else MAX_EPOCHS)
    warmup = cfg.warmup_steps if not cfg.debug else 5

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v2"),
        group=cfg.wandb_group,
        name=cfg.wandb_name,
        tags=[cfg.agent] if cfg.agent else [],
        config={
            **asdict(cfg),
            "model": "ResMLP+Fourier",
            "model_config": model_config,
            "n_params": n_params,
            "train_samples": len(train_ds),
            "val_samples": {k: len(v) for k, v in val_splits.items()},
        },
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")
    for _name in VAL_SPLIT_NAMES:
        wandb.define_metric(f"{_name}/*", step_metric="global_step")
    wandb.define_metric("lr", step_metric="global_step")

    model_dir = Path(f"models/model-{run.id}")
    model_dir.mkdir(parents=True)
    model_path = model_dir / "checkpoint.pt"
    with open(model_dir / "config.yaml", "w") as f:
        yaml.dump(model_config, f)

    # EMA shadow weights — averaged params used at val/inference time.
    ema_state: dict[str, torch.Tensor] | None = None
    if cfg.ema_decay > 0:
        ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    chan_w = torch.tensor([1.0, 1.0, cfg.p_weight], device=device)

    def channel_loss(pred, y_norm, mask_pts):
        diff = pred - y_norm
        err = diff.abs() if cfg.loss_type == "l1" else diff ** 2
        err = err * chan_w
        masked = err * mask_pts.unsqueeze(-1)
        return masked.sum() / mask_pts.sum().clamp(min=1) / chan_w.sum()

    best_surf_p = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT - 2.0:
            print(f"Approaching timeout. Stopping at epoch {epoch}.")
            break

        t0 = time.time()
        model.train()
        epoch_vol = epoch_surf = 0.0
        n_batches = 0

        amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
        for x, y, is_surface, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if cfg.train_subsample > 0:
                x, y, is_surface, mask = subsample_batch(x, y, is_surface, mask, cfg.train_subsample)

            x = (x - stats["x_mean"]) / stats["x_std"]
            y_norm = (y - stats["y_mean"]) / stats["y_std"]

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=cfg.amp):
                pred = model({"x": x, "mask": mask})["preds"]
            pred = pred.float()

            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            vol_loss = channel_loss(pred, y_norm, vol_mask)
            surf_loss = channel_loss(pred, y_norm, surf_mask)
            loss = vol_loss + cfg.surf_weight * surf_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # EMA update (after the optimiser step, only after warmup).
            if ema_state is not None and global_step >= cfg.ema_start_step:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema_state[k].mul_(cfg.ema_decay).add_(v.detach(), alpha=1.0 - cfg.ema_decay)
                        else:
                            ema_state[k].copy_(v.detach())
            global_step += 1
            wandb.log({
                "train/loss": loss.item(),
                "train/vol_loss_step": vol_loss.item(),
                "train/surf_loss_step": surf_loss.item(),
                "lr": scheduler.get_last_lr()[0],
                "global_step": global_step,
            })

            epoch_vol += vol_loss.item()
            epoch_surf += surf_loss.item()
            n_batches += 1

        epoch_vol /= n_batches
        epoch_surf /= n_batches

        # Validate on epoch 1 always, then every val_every epochs.
        do_val = (epoch == 0) or ((epoch + 1) % cfg.val_every == 0)
        if do_val:
            # Validate using EMA weights once EMA has begun; otherwise live weights.
            if ema_state is not None and global_step >= cfg.ema_start_step:
                live_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema_state)
                mean_val_loss, mean_surf_p, split_metrics = _validate(
                    model, val_loaders, stats, device, cfg.surf_weight, amp=cfg.amp,
                )
                model.load_state_dict(live_state)
            else:
                mean_val_loss, mean_surf_p, split_metrics = _validate(
                    model, val_loaders, stats, device, cfg.surf_weight, amp=cfg.amp,
                )
        else:
            mean_val_loss, mean_surf_p, split_metrics = float("nan"), float("nan"), {}

        dt = time.time() - t0

        metrics = {
            "train/vol_loss": epoch_vol,
            "train/surf_loss": epoch_surf,
            "epoch_time_s": dt,
            "global_step": global_step,
        }
        if do_val:
            metrics.update({"val/loss": mean_val_loss, "val/mean_surf_p": mean_surf_p})
            for sm in split_metrics.values():
                metrics.update(sm)
        wandb.log(metrics)

        tag = ""
        if do_val and mean_surf_p < best_surf_p:
            best_surf_p = mean_surf_p
            best_metrics = {"epoch": epoch + 1, "val_loss": mean_val_loss, "mean_surf_p": mean_surf_p}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            # Save EMA weights if available, else live weights.
            save_state = ema_state if (ema_state is not None and global_step >= cfg.ema_start_step) else model.state_dict()
            torch.save(save_state, model_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        if do_val:
            split_summary = "  ".join(
                f"{name.replace('val_', '')}_p={split_metrics[name][f'{name}/mae_surf_p']:.2f}"
                for name in VAL_SPLIT_NAMES
            )
            print(
                f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
                f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  "
                f"surf_p[{split_summary}] mean={mean_surf_p:.2f}{tag}"
            )
        else:
            print(
                f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
                f"train[vol={epoch_vol:.4f} surf={epoch_surf:.4f}]  (no val)"
            )

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/loss={best_metrics['val_loss']:.4f}, "
              f"mean_surf_p={best_metrics['mean_surf_p']:.2f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        plot_dir = Path("plots") / run.id
        n = 1 if cfg.debug else 4
        for split_name, split_ds in val_splits.items():
            images = visualize(model, split_ds, stats, device, n_samples=n,
                               out_dir=plot_dir / split_name)
            if images:
                wandb.log({
                    f"val_predictions/{split_name}": [wandb.Image(str(p)) for p in images],
                    "global_step": global_step,
                })

        repo_ckpt_dir = Path("checkpoints")
        repo_ckpt_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), repo_ckpt_dir / "best.pt")
        research_tag = os.environ.get("RESEARCH_TAG", "default")
        kaggler_name = os.environ.get("KAGGLER_NAME", cfg.agent or "unknown")
        pvc_dir = Path(f"/mnt/new-pvc/kagent/{research_tag}/{kaggler_name}/checkpoints/model-{run.id}")
        pvc_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), pvc_dir / "checkpoint.pt")
        with open(pvc_dir / "config.yaml", "w") as f:
            yaml.dump(model_config, f)
        print(f"Mirrored checkpoint to {pvc_dir}")

    if best_metrics and not cfg.debug:
        # Free GPU memory before spawning predict.py subprocess (otherwise both
        # processes try to fit on one GPU and the child OOMs).
        import gc
        del model, optimizer, scheduler
        gc.collect()
        torch.cuda.empty_cache()

        print("\nGenerating test predictions...")
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-1500:]}")

    wandb.finish()


if __name__ == "__main__":
    main()
