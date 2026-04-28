"""Evaluate Re-perturbation TTA on val.

Hypothesis: averaging predictions over small perturbations of log(Re) (feature 13,
in normalized space) regularizes errors, especially on val_re_rand which holds
out OOD Re values. Cheap to apply at inference — no retraining needed.

Tests a single ckpt with N TTA passes at sigma in normalized space.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, load_data, pad_collate
from model import Transolver


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    sigmas: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3])  # log(Re) normalized space
    n_tta: int = 3  # number of TTA passes (e.g., -sigma, 0, +sigma if 3)
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

models = []
for ckpt in cfg.checkpoints:
    config_path = Path(ckpt).parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
    print(f"Loaded: {ckpt}")

if not cfg.weights:
    cfg.weights = [1.0 / len(models)] * len(models)
weights = [w / sum(cfg.weights) for w in cfg.weights]
print(f"Weights: {weights}")

_, val_splits, stats, _ = load_data(splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2, pin_memory=True)
    for name, ds in val_splits.items()
}


def offsets_for(sigma, n_tta):
    """Symmetric grid of offsets in normalized log(Re) space."""
    if sigma == 0 or n_tta == 1:
        return [0.0]
    if n_tta == 3:
        return [-sigma, 0.0, sigma]
    if n_tta == 5:
        return [-2 * sigma, -sigma, 0.0, sigma, 2 * sigma]
    raise ValueError(f"n_tta {n_tta} unsupported")


@torch.no_grad()
def eval_with_tta(sigma):
    """Returns dict {split: surf_p_mae}."""
    offsets = offsets_for(sigma, cfg.n_tta) if sigma > 0 else [0.0]
    out = {}
    for split_name, vloader in val_loaders.items():
        total_err = 0.0
        total_count = 0
        for x, y, is_surface, mask in vloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm_base = (x - stats["x_mean"]) / stats["x_std"]
            surf_mask = mask & is_surface

            ensembled = None
            for off in offsets:
                x_norm = x_norm_base.clone()
                if off != 0.0:
                    x_norm[..., 13] += off
                pred_sum = None
                for m, w in zip(models, weights):
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        p = m({"x": x_norm})["preds"].float()
                    p_phys = p * stats["y_std"] + stats["y_mean"]
                    pred_sum = p_phys * w if pred_sum is None else pred_sum + p_phys * w
                ensembled = pred_sum if ensembled is None else ensembled + pred_sum
            ensembled = ensembled / len(offsets)

            err = (ensembled - y).abs()
            total_err += (err[..., 2] * surf_mask).sum().item()
            total_count += surf_mask.sum().item()
        out[split_name] = total_err / max(total_count, 1)
    return out


print(f"\n=== TTA sweep (n_tta={cfg.n_tta}) ===")
print(f"{'sigma':<8}" + "  ".join(f"{n[4:].replace('_',''):>14}" for n in VAL_SPLIT_NAMES) + f"  {'avg':>8}")
print("-" * 90)
for sigma in cfg.sigmas:
    res = eval_with_tta(sigma)
    avg = sum(res.values()) / len(res)
    print(f"{sigma:<8.3f}" + "  ".join(f"{res[n]:>14.4f}" for n in VAL_SPLIT_NAMES) + f"  {avg:>8.4f}")
