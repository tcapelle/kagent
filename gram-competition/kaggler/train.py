"""Train a 3D airflow velocity predictor.

Template — fill in your model architecture.
The training loop, loss, validation, and W&B logging are provided.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Baseline MLP — replace with your own architecture
#
# Model contract:
#   Input:  velocity_in [B, 5, N, 3], pos [B, N, 3], t [B, 10], idcs_airfoil list[tensor]
#   Output: velocity_out [B, 5, N, 3]  (predicted future velocity field)
#
# Note: the real competition uses model(t, pos, idcs_airfoil, velocity_in) —
#       different arg order. If you submit to the real comp, wrap accordingly.
# ---------------------------------------------------------------------------


def fourier_encode(x: torch.Tensor, n_freqs: int = 6) -> torch.Tensor:
    """NeRF-style positional encoding.

    Concatenates raw x plus sin/cos at frequencies pi, 2pi, 4pi, ..., 2^(n_freqs-1)*pi.
    Input shape [..., D], output shape [..., D + D*2*n_freqs].
    Positions are ~2m in extent; highest freq gives ~6cm wavelength.
    """
    freqs = (2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)) * torch.pi
    scaled = x.unsqueeze(-1) * freqs  # [..., D, F]
    enc = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [..., D, 2F]
    enc = enc.reshape(*x.shape[:-1], -1)
    return torch.cat([x, enc], dim=-1)


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


def sdpa(q, k, v):
    """Scaled-dot-product attention using PyTorch's fused kernel (Flash on CUDA).

    Shapes: q [B, H, Lq, D], k [B, H, Lk, D], v [B, H, Lk, D] -> [B, H, Lq, D].
    """
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class PhysicsAttentionBlock(nn.Module):
    """Transolver Physics-Attention (ICML'24).

    Each point is softly assigned to M learnable "slices" via a per-head softmax over
    slice weights w[n, m]. Each slice aggregates its assigned points' features (weighted
    mean), runs self-attention across the M slice tokens, then deslices back via the
    same weights. Cost O(N*M + M^2) vs Perceiver's fixed-query O(N*M) — same scaling,
    but the slice assignments are *data-dependent*, i.e. each sample gets its own
    clusters that adapt to the geometry (wake vs freestream vs boundary layer).

    Followed by a small point-wise FFN for post-mixing.

    Ref: github.com/thuml/Transolver, `Physics_Attention.py`.
    """

    def __init__(self, dim: int, n_slices: int = 32, n_heads: int = 8):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_slices = n_slices

        self.ln1 = nn.LayerNorm(dim)
        # Orthogonal init on slice projection breaks slice symmetry.
        self.to_slice_logits = nn.Linear(dim, n_heads * n_slices, bias=False)
        nn.init.orthogonal_(self.to_slice_logits.weight)
        # Learnable per-head softmax temperature (init 0.5 per paper).
        self.temperature = nn.Parameter(torch.full((1, n_heads, 1, 1), 0.5))

        # Per-head feature projection for slice values.
        self.to_v = nn.Linear(dim, dim, bias=False)
        # Self-attention over slice tokens (standard MHSA at full dim D).
        self.ln_s = nn.LayerNorm(dim)
        self.to_qkv_s = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]
        B, N, D = h.shape
        H, M, Dh = self.n_heads, self.n_slices, self.head_dim

        x = self.ln1(h)

        # Slice assignment: [B, N, D] -> logits [B, H, N, M] -> softmax over M.
        logits = self.to_slice_logits(x).reshape(B, N, H, M).permute(0, 2, 1, 3)  # [B,H,N,M]
        w = torch.softmax(logits / self.temperature, dim=-1)                      # [B,H,N,M]

        # Slice values: per-head projection of x -> [B, H, N, Dh].
        v = self.to_v(x).reshape(B, N, H, Dh).permute(0, 2, 1, 3)                 # [B,H,N,Dh]

        # Aggregate points into slice tokens via weighted mean.
        # slice_sum[b,h,m,d] = sum_n w[b,h,n,m] * v[b,h,n,d]
        w_sum = w.sum(dim=2, keepdim=True).clamp_min(1e-5)                         # [B,H,1,M]
        w_t = w.transpose(-1, -2)                                                  # [B,H,M,N]
        slice_tokens = torch.einsum('bhmn,bhnd->bhmd', w_t, v) / w_sum.transpose(-1, -2)  # [B,H,M,Dh]

        # Self-attend across slice tokens. Reshape per-head tokens to [B, M, D]
        # then run standard MHSA with separate Q/K/V projections.
        s_flat = slice_tokens.permute(0, 2, 1, 3).reshape(B, M, D)                 # [B,M,D]
        s_ln = self.ln_s(s_flat)
        qkv = self.to_qkv_s(s_ln).reshape(B, M, 3, H, Dh).permute(2, 0, 3, 1, 4)   # [3,B,H,M,Dh]
        q, k, vv = qkv[0], qkv[1], qkv[2]
        s_attn = sdpa(q, k, vv)                                                    # [B,H,M,Dh]
        s_out = slice_tokens + s_attn                                              # [B,H,M,Dh]

        # Deslice: scatter slice token back to every point using the same weights.
        out_heads = torch.einsum('bhnm,bhmd->bhnd', w, s_out)                      # [B,H,N,Dh]
        out = out_heads.permute(0, 2, 1, 3).reshape(B, N, D)                       # [B,N,D]

        h = h + self.proj(out)
        h = h + self.ffn(self.ln2(h))
        return h


def compute_sdf(pos: torch.Tensor, idcs_airfoil, cache: dict, chunk: int = 4096) -> torch.Tensor:
    """Distance from each point to the nearest airfoil-surface point.

    Computed on GPU in chunks. Results cached by a simple pos-hash so each unique
    geometry (146 in train) is computed at most once across all epochs.

    pos: [B, N, 3], idcs_airfoil: list of length-B tensors.
    returns: [B, N]
    """
    B, N, _ = pos.shape
    out = torch.empty(B, N, device=pos.device, dtype=pos.dtype)
    for b in range(B):
        key = (float(pos[b].sum()), float(pos[b].view(-1)[0]), float(pos[b].view(-1)[-1]))
        if key in cache:
            out[b] = cache[key].to(pos.device, non_blocking=True)
            continue
        pos_a = pos[b, idcs_airfoil[b]]  # [n_a, 3]
        sdf_b = torch.empty(N, device=pos.device, dtype=pos.dtype)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(pos[b:b + 1, s:e], pos_a.unsqueeze(0))  # [1, chunk, n_a]
            sdf_b[s:e] = d[0].min(dim=-1).values
        cache[key] = sdf_b.detach().cpu()
        out[b] = sdf_b
    return out


def compute_sdf_and_rel(pos: torch.Tensor, idcs_airfoil, cache: dict, chunk: int = 4096):
    """Distance AND relative vector from each point to its nearest airfoil point.

    Returns (sdf [B,N], rel [B,N,3]) where rel = pos - nearest_airfoil_pos. rel
    encodes both distance AND orientation (e.g. upstream vs downstream, above vs
    below the suction side) — strictly richer than the scalar SDF. Cached
    per-geometry keyed by pos-hash (same key as compute_sdf but stores both).
    """
    B, N, _ = pos.shape
    sdf = torch.empty(B, N, device=pos.device, dtype=pos.dtype)
    rel = torch.empty(B, N, 3, device=pos.device, dtype=pos.dtype)
    for b in range(B):
        key = ("rel", float(pos[b].sum()), float(pos[b].view(-1)[0]), float(pos[b].view(-1)[-1]))
        if key in cache:
            cached = cache[key]
            sdf[b] = cached[0].to(pos.device, non_blocking=True)
            rel[b] = cached[1].to(pos.device, non_blocking=True)
            continue
        pos_a = pos[b, idcs_airfoil[b]]  # [n_a, 3]
        sdf_b = torch.empty(N, device=pos.device, dtype=pos.dtype)
        nearest_b = torch.empty(N, 3, device=pos.device, dtype=pos.dtype)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            d = torch.cdist(pos[b:b + 1, s:e], pos_a.unsqueeze(0))[0]  # [chunk, n_a]
            min_d, min_i = d.min(dim=-1)
            sdf_b[s:e] = min_d
            nearest_b[s:e] = pos_a[min_i]
        rel_b = pos[b] - nearest_b
        cache[key] = (sdf_b.detach().cpu(), rel_b.detach().cpu())
        sdf[b] = sdf_b
        rel[b] = rel_b
    return sdf, rel


class BaselineMLP(nn.Module):
    """Residual per-point ResMLP with Fourier pos, temporal diffs, and Perceiver context.

    Tricks vs. the raw baseline:
      - Fourier (NeRF) encoding of positions.
      - Normalized velocity + normalized inter-step diffs (acceleration proxy) as input.
      - Predict delta = v_out - v_in[-1]; add v_in[-1] back.
      - Hard no-slip BC: zero velocity at airfoil indices.
      - Transolver Physics-Attention blocks (data-dependent slice tokens, M=32) provide
        geometry-aware global context at O(N*M) cost. Stacked as the main trunk.
      - Time-conditioned per-step decoder: a shared MLP takes (point_feat, time_embed(k))
        and predicts delta for each output step k separately. Replaces the monolithic
        Linear(hidden -> 5*3) that forced all steps through one predictor.
    """

    def __init__(
        self,
        hidden: int = 384,
        n_blocks: int = 8,
        n_pos_freqs: int = 6,
        n_slices: int = 32,
        n_heads: int = 8,
        time_embed_dim: int = 32,
        vel_mean: torch.Tensor | None = None,
        vel_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_pos_freqs = n_pos_freqs
        self.time_embed_dim = time_embed_dim
        pos_feat_dim = 3 + 3 * 2 * n_pos_freqs      # 39
        vel_feat_dim = T_IN * 3 + (T_IN - 1) * 3     # 15 + 12 = 27
        in_dim = pos_feat_dim + vel_feat_dim         # 66

        self.proj_in = nn.Linear(in_dim, hidden)
        # SDF-to-airfoil branch: small MLP that lifts the per-point distance feature
        # into the trunk dim and adds it residually. Zero-init on the final layer
        # means the branch is exactly a no-op at warm-start time and the model
        # learns how much to use it as training progresses.
        sdf_feat_dim = 1 + 2 * 4  # raw + Fourier (L=4)
        self.sdf_n_freqs = 4
        self.sdf_embed = nn.Sequential(
            nn.Linear(sdf_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.sdf_embed[-1].weight)
        nn.init.zeros_(self.sdf_embed[-1].bias)
        # Relative-vector-to-nearest-airfoil-point branch. Strictly richer than the
        # scalar SDF branch (same distance info + direction). Kept as a SEPARATE
        # zero-init additive branch so warm-start from an iter-15 checkpoint (which
        # has no rel branch) produces exactly identical output at init.
        self.rel_n_freqs = 4
        rel_feat_dim = 3 + 3 * 2 * self.rel_n_freqs  # 3 raw + Fourier = 27
        self.rel_embed = nn.Sequential(
            nn.Linear(rel_feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        nn.init.zeros_(self.rel_embed[-1].weight)
        nn.init.zeros_(self.rel_embed[-1].bias)
        self._sdf_cache: dict = {}

        # Trunk: stack of Physics-Attention blocks (each has its own FFN + residual).
        self.blocks = nn.ModuleList([
            PhysicsAttentionBlock(hidden, n_slices=n_slices, n_heads=n_heads)
            for _ in range(n_blocks)
        ])
        # Time-conditioned decoder: shared across output steps, but each step gets
        # its own time embedding as additional input.
        self.time_embed = nn.Embedding(T_OUT, time_embed_dim)
        self.ln_dec = nn.LayerNorm(hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + time_embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

        if vel_mean is None:
            vel_mean = torch.zeros(3)
        if vel_std is None:
            vel_std = torch.ones(3)
        self.register_buffer("vel_mean", vel_mean.reshape(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.reshape(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil, sdf=None, rel=None):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std  # [B, T, N, 3]
        v_diff = v_norm[:, 1:] - v_norm[:, :-1]                # [B, T-1, N, 3]
        v_last = velocity_in[:, -1]                            # [B, N, 3]

        pos_feat = fourier_encode(pos, self.n_pos_freqs)        # [B, N, 39]
        vel_feat = torch.cat([
            v_norm.permute(0, 2, 1, 3).reshape(B, N, T * C),
            v_diff.permute(0, 2, 1, 3).reshape(B, N, (T - 1) * C),
        ], dim=-1)                                              # [B, N, 27]
        x = torch.cat([pos_feat, vel_feat], dim=-1)             # [B, N, 66]

        x = self.proj_in(x)
        # SDF + relative-vector features (distance AND direction to nearest airfoil).
        # Both branches are zero-init on their final Linear, so warm-starting from a
        # checkpoint that has either (or neither) keeps forward exactly equivalent.
        # Caller can pre-compute (e.g. after subsampling); otherwise we compute here.
        if sdf is None or rel is None:
            sdf, rel = compute_sdf_and_rel(pos, idcs_airfoil, self._sdf_cache)
        sdf_feat = fourier_encode(sdf.unsqueeze(-1), self.sdf_n_freqs)  # [B, N, 9]
        x = x + self.sdf_embed(sdf_feat)
        rel_feat = fourier_encode(rel, self.rel_n_freqs)                # [B, N, 27]
        x = x + self.rel_embed(rel_feat)
        for block in self.blocks:
            x = block(x)
        x = self.ln_dec(x)  # [B, N, hidden]

        # Decode per output time step with a shared MLP + step embedding.
        step_ids = torch.arange(T_OUT, device=x.device)                 # [T_OUT]
        step_emb = self.time_embed(step_ids)                             # [T_OUT, D_t]
        # Expand: x -> [B, T_OUT, N, H], step_emb -> [1, T_OUT, 1, D_t]
        x_exp = x.unsqueeze(1).expand(B, T_OUT, N, -1)
        step_exp = step_emb.reshape(1, T_OUT, 1, self.time_embed_dim).expand(B, T_OUT, N, -1)
        dec_in = torch.cat([x_exp, step_exp], dim=-1)                    # [B, T_OUT, N, H+D_t]
        delta = self.decoder(dec_in)                                     # [B, T_OUT, N, 3]
        out = v_last.unsqueeze(1) + delta
        for b in range(B):
            out[b, :, idcs_airfoil[b], :] = 0.0
        return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def sobolev_anchor_loss(pred, gt, pos, n_anchors: int = 1024, k: int = 8):
    """Stochastic gradient-matching loss.

    For a random subset of `n_anchors` anchor points, compute the kNN (in position
    space), then penalise ||(pred[nei] - pred[anchor]) - (gt[nei] - gt[anchor])||.
    This is a Sobolev-style loss: getting the spatial gradient right matters as
    much as the point value. Matches the local flow shear structure.

    Anchors are resampled every call so the loss covers the full point cloud in
    expectation without paying the O(N^2) cost of computing the full kNN.

    pred, gt: [B, T, N, 3]
    pos:      [B, N, 3]
    """
    B, T, N, C = pred.shape
    anchor_idx = torch.randperm(N, device=pos.device)[:n_anchors]           # [M]
    pos_a = pos[:, anchor_idx]                                              # [B, M, 3]
    d = torch.cdist(pos_a, pos)                                             # [B, M, N]
    _, knn = d.topk(k + 1, dim=-1, largest=False)                           # [B, M, k+1]
    knn = knn[..., 1:]                                                      # drop self -> [B, M, k]

    pred_a = pred[:, :, anchor_idx, :]                                      # [B, T, M, 3]
    gt_a = gt[:, :, anchor_idx, :]

    knn_flat = knn.reshape(B, -1)                                           # [B, M*k]
    idx_exp = knn_flat.unsqueeze(1).unsqueeze(-1).expand(B, T, n_anchors * k, C)
    pred_n = torch.gather(pred, 2, idx_exp).reshape(B, T, n_anchors, k, C)
    gt_n = torch.gather(gt, 2, idx_exp).reshape(B, T, n_anchors, k, C)

    pred_diff = pred_n - pred_a.unsqueeze(3)
    gt_diff = gt_n - gt_a.unsqueeze(3)
    return (pred_diff - gt_diff).norm(dim=-1).mean()


def subsample_batch(v_in, v_out, pos, idcs_airfoil, sdf, rel, n_keep: int):
    """Train-time point subsampling for geometry-level data augmentation.

    Keep all airfoil-surface points + random selection of the rest. Subsampled
    `idcs_airfoil` become contiguous indices [0, n_air-1]. The model is permutation-
    invariant so ordering doesn't matter for the trunk; idcs are only used for the
    output no-slip mask.

    NOTE: iter 16 v1 tried --train_n_points=40000 warm-started from iter 15 and val
    jumped 1.066 -> 1.100 — Transolver slice statistics shift with N, and the
    warm-started weights were tuned for 100k. Left in as a flag for future fresh-
    train experiments; default off.
    """
    B, T, N, C = v_in.shape
    device = v_in.device
    new_v_in = torch.empty(B, T, n_keep, C, device=device, dtype=v_in.dtype)
    new_v_out = torch.empty(B, T, n_keep, C, device=device, dtype=v_out.dtype)
    new_pos = torch.empty(B, n_keep, 3, device=device, dtype=pos.dtype)
    new_sdf = torch.empty(B, n_keep, device=device, dtype=sdf.dtype)
    new_rel = torch.empty(B, n_keep, 3, device=device, dtype=rel.dtype)
    new_idcs = []
    for b in range(B):
        air = idcs_airfoil[b].to(device)
        n_air = min(air.shape[0], n_keep)
        air = air[:n_air]
        n_rest = n_keep - n_air
        if n_rest > 0:
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[air] = False
            non_air = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            perm = torch.randperm(non_air.shape[0], device=device)[:n_rest]
            keep = torch.cat([air, non_air[perm]])
        else:
            keep = air
        new_v_in[b] = v_in[b, :, keep]
        new_v_out[b] = v_out[b, :, keep]
        new_pos[b] = pos[b, keep]
        new_sdf[b] = sdf[b, keep]
        new_rel[b] = rel[b, keep]
        new_idcs.append(torch.arange(n_air, device=device))
    return new_v_in, new_v_out, new_pos, new_idcs, new_sdf, new_rel


def reflect_y(v_in, v_out, pos):
    """Apply y-axis reflection to the whole sample (in-place safe on clones).

    Flips pos_y and the y-component of velocities. The F1 wing geometry and its
    flow field are near-perfectly y-symmetric, so this produces physically valid
    augmented samples and doubles the effective dataset size.
    """
    v_in = v_in.clone();   v_in[..., 1] = -v_in[..., 1]
    v_out = v_out.clone(); v_out[..., 1] = -v_out[..., 1]
    pos = pos.clone();     pos[..., 1] = -pos[..., 1]
    return v_in, v_out, pos


def predict_tta(model, v_in, pos, t, idcs):
    """Test-time augmentation: average prediction of original and y-mirrored inputs.

    The mirror operation + un-mirror of the prediction enforces exact y-symmetry
    on the output and is a cheap ensemble (2 forward passes).
    """
    pred_orig = model(v_in, pos, t, idcs)
    v_in_m = v_in.clone();   v_in_m[..., 1] = -v_in_m[..., 1]
    pos_m = pos.clone();     pos_m[..., 1] = -pos_m[..., 1]
    pred_m = model(v_in_m, pos_m, t, idcs)
    pred_m = pred_m.clone()
    pred_m[..., 1] = -pred_m[..., 1]
    return 0.5 * (pred_orig + pred_m)


def validate(model, val_loaders, device, global_step, use_tta: bool = True):
    """Run validation, log to W&B. Returns mean val metric (L2 velocity error)."""
    model.eval()
    val_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        total_l2 = 0.0
        total_mae = torch.zeros(3, device=device, dtype=torch.float64)
        n_samples = 0

        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if use_tta:
                        pred = predict_tta(model, v_in, pos, t, idcs).float()
                    else:
                        pred = model(v_in, pos, t, idcs).float()  # [B, 5, N, 3]

                # L2 velocity error (competition hint metric)
                l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))  # [B]
                total_l2 += l2_err.sum().item()

                # Per-component MAE
                mae = (pred - v_out).abs().mean(dim=(1, 2))  # [B, 3]
                total_mae += mae.double().sum(dim=0)
                n_samples += v_in.shape[0]

        mean_l2 = total_l2 / max(n_samples, 1)
        mean_mae = total_mae / max(n_samples, 1)

        val_metrics[split_name] = {
            f"{split_name}/l2_error": mean_l2,
            f"{split_name}/mae_Ux": mean_mae[0].item(),
            f"{split_name}/mae_Uy": mean_mae[1].item(),
            f"{split_name}/mae_Uz": mean_mae[2].item(),
        }

    mean_val = sum(m[f"{k}/l2_error"] for k, m in val_metrics.items()) / len(val_metrics)

    metrics = {"val/l2_error": mean_val, "global_step": global_step}
    for sm in val_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    return mean_val, val_metrics


# ---------------------------------------------------------------------------
# Config + data loading
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))  # minutes


@dataclass
class Config:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    batch_size: int = 1
    epochs: int = 25  # tuned to what fits in 30 min so the cosine actually anneals
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    resume: str | None = None  # path to a checkpoint to warm-start from
    sobolev_lambda: float = 0.0  # weight for stochastic Sobolev/gradient-matching loss
    sobolev_anchors: int = 1024
    sobolev_k: int = 8
    train_n_points: int = 0  # if >0, subsample each train sample to this many points (val unchanged)


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = BaselineMLP(
        hidden=384,
        n_blocks=8,
        vel_mean=stats["vel_mean"],
        vel_std=stats["vel_std"],
    ).to(device)

    if cfg.resume:
        state = torch.load(cfg.resume, map_location=device, weights_only=True)
        # strict=False lets us warm-start even when the architecture has new
        # zero-initialized branches (e.g. the SDF embedding) that don't exist in
        # the older checkpoint.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Warm-started from {cfg.resume} (missing={len(missing)}, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Warmup (linear 0 -> lr over warmup_steps) then cosine anneal over remaining steps.
    steps_per_epoch = len(train_loader)
    total_steps = MAX_EPOCHS * steps_per_epoch
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, cfg.warmup_steps)
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - cfg.warmup_steps)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_steps]
    )

    RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-gram"),
        group=cfg.wandb_group or RESEARCH_TAG,
        name=cfg.wandb_name,
        tags=[t for t in [cfg.agent, RESEARCH_TAG] if t],
        config={**asdict(cfg), "n_params": n_params,
                "train_samples": len(train_ds),
                "val_samples": {k: len(v) for k, v in val_splits.items()}},
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    KAGGLER_NAME = os.environ.get("KAGGLER_NAME", cfg.agent or "local")
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}/{KAGGLER_NAME}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    model_path = pvc_dir / "checkpoint.pt"

    git_ckpt_path = Path("checkpoints/best.pt")
    git_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    # When warm-starting, eval the resumed model before training so we never
    # overwrite the in-repo best.pt with a worse checkpoint (which silently
    # happened on iter 16 E1 at 1.10 clobbering iter 15 at 1.066).
    if cfg.resume:
        init_val, _ = validate(model, val_loaders, device, global_step)
        best_val = init_val
        print(f"Warm-start init val/l2={init_val:.4f} (floor for saving checkpoints)")

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT:
            print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
            break

        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for v_in, v_out, pos, t, idcs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            # Reflection augmentation: with p=0.5 flip y-axis of pos + velocities.
            # F1 wing is y-symmetric so the mirrored sample is physically valid.
            if torch.rand(1).item() < 0.5:
                v_in, v_out, pos = reflect_y(v_in, v_out, pos)

            # With optional point subsampling, precompute SDF+rel at FULL pos (cached
            # per-geometry) before subsampling — the cache key depends on full pos.
            sdf_in = rel_in = None
            if cfg.train_n_points > 0 and cfg.train_n_points < pos.shape[1]:
                sdf_full, rel_full = compute_sdf_and_rel(pos, idcs, model._sdf_cache)
                v_in, v_out, pos, idcs, sdf_in, rel_in = subsample_batch(
                    v_in, v_out, pos, idcs, sdf_full, rel_full, cfg.train_n_points
                )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(v_in, pos, t, idcs, sdf=sdf_in, rel=rel_in)  # [B, 5, N, 3]
                # L2-per-point loss: matches the val metric; less outlier-dominated than MSE.
                l2_loss = (pred - v_out).norm(dim=3).mean()
                if cfg.sobolev_lambda > 0:
                    sob = sobolev_anchor_loss(
                        pred, v_out, pos, cfg.sobolev_anchors, cfg.sobolev_k
                    )
                    loss = l2_loss + cfg.sobolev_lambda * sob
                else:
                    loss = l2_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": scheduler.get_last_lr()[0],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(model.state_dict(), model_path)
            # Skip the in-repo copy in debug mode so a quick sanity run can't
            # clobber a real, long-trained checkpoint.
            if not cfg.debug:
                shutil.copyfile(model_path, git_ckpt_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
            f"train={epoch_loss:.4f}  val/l2={mean_val:.4f}{tag}"
        )

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/l2_error={best_metrics['val_l2_error']:.4f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    if best_metrics and not cfg.debug:
        import subprocess
        print("\nGenerating test predictions...")
        # If an ensemble directory exists on PVC, submit the ensemble (current
        # checkpoint + every prior checkpoint in the ensemble dir) instead of
        # the lone current checkpoint. Iter 16 found this bought ~0.9% val drop
        # for free (1.0628 single -> 1.0537 5-model). Any file matching iter*.pt
        # in that dir is included plus the just-trained model_path.
        ensemble_dir = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}/{KAGGLER_NAME}/ensemble_ckpts")
        ensemble_members = sorted(ensemble_dir.glob("iter*.pt")) if ensemble_dir.exists() else []
        if ensemble_members:
            # Keep only members strictly better than the (single-) current ckpt?  No —
            # including iter 12 (val 1.14) still helped the 5-model ensemble; blanket
            # include everything. The leaderboard metric is robust to weak members.
            ckpt_list = [str(p) for p in ensemble_members] + [str(model_path)]
            pred_cmd = ["python", "predict.py", "--checkpoints", ",".join(ckpt_list)]
            print(f"Ensemble predict: {len(ckpt_list)} checkpoints")
        else:
            pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()


if __name__ == "__main__":
    main()
