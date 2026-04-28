"""Evaluate many ensemble combos on val to find best per-split optimum.

Caches predictions for all candidate models, then sweeps weight grids
across multiple model subsets to find the lowest avg surface_p MAE.
"""

import itertools
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
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4
    # Subsets to try: comma-separated indices (0-indexed into checkpoints list)
    subsets: list[str] = field(default_factory=list)
    grid_step: float = 0.05  # weight grid step


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
    print(f"Loaded [{len(models)-1}]: {ckpt}")

_, val_splits, stats, _ = load_data(splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2, pin_memory=True)
    for name, ds in val_splits.items()
}


@torch.no_grad()
def cache_predictions():
    cache = {}
    for split_name, vloader in val_loaders.items():
        per_model_preds = [[] for _ in models]
        ys, surf_masks = [], []
        for x, y, is_surface, mask in vloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            for i, m in enumerate(models):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = m({"x": x_norm})["preds"].float()
                p_phys = p * stats["y_std"] + stats["y_mean"]
                per_model_preds[i].append(p_phys.cpu())
            ys.append(y.cpu())
            surf_masks.append((mask & is_surface).cpu())
        cache[split_name] = {
            "preds": per_model_preds,
            "y": ys,
            "surf_mask": surf_masks,
        }
    return cache


print(f"\nCaching predictions for {len(models)} models...")
cache = cache_predictions()
print("Done.")


def eval_per_split(weights, idxs):
    """weights[k] applied to model[idxs[k]]. Returns dict {split: surf_p_mae}."""
    weights = [w / sum(weights) for w in weights]
    out = {}
    for split_name, c in cache.items():
        total_err = 0.0
        total_count = 0
        for batch_idx in range(len(c["y"])):
            ensembled = sum(c["preds"][idxs[k]][batch_idx] * weights[k] for k in range(len(weights)))
            err = (ensembled - c["y"][batch_idx]).abs()
            total_err += (err[..., 2] * c["surf_mask"][batch_idx]).sum().item()
            total_count += c["surf_mask"][batch_idx].sum().item()
        out[split_name] = total_err / max(total_count, 1)
    return out


def grid_weights(n, step):
    """Generate all simplex weights (sum=1) with step granularity."""
    n_steps = int(round(1.0 / step))
    out = []
    if n == 2:
        for a in range(n_steps + 1):
            out.append([a / n_steps, (n_steps - a) / n_steps])
    elif n == 3:
        for a in range(n_steps + 1):
            for b in range(n_steps + 1 - a):
                c = n_steps - a - b
                out.append([a / n_steps, b / n_steps, c / n_steps])
    elif n == 4:
        for a in range(n_steps + 1):
            for b in range(n_steps + 1 - a):
                for c in range(n_steps + 1 - a - b):
                    d = n_steps - a - b - c
                    out.append([a / n_steps, b / n_steps, c / n_steps, d / n_steps])
    return out


print("\n" + "=" * 100)
for subset_str in cfg.subsets:
    idxs = [int(x) for x in subset_str.split(",")]
    label = "+".join(str(i) for i in idxs)
    n = len(idxs)
    weights_grid = grid_weights(n, cfg.grid_step)
    print(f"\n=== Subset {label} (n={n}, {len(weights_grid)} configs) ===")

    # Per-split optimum sweep
    per_split_best: dict = {s: (None, float("inf")) for s in VAL_SPLIT_NAMES}
    uniform_best = (None, float("inf"))
    for w in weights_grid:
        if sum(w) == 0:
            continue
        res = eval_per_split(w, idxs)
        avg = sum(res.values()) / len(res)
        if avg < uniform_best[1]:
            uniform_best = (w, avg)
        for s, v in res.items():
            if v < per_split_best[s][1]:
                per_split_best[s] = (w, v)
    total_per_split = sum(v for _, v in per_split_best.values()) / 4
    print(f"  uniform best: avg={uniform_best[1]:.3f} weights={uniform_best[0]}")
    for s, (w, v) in per_split_best.items():
        print(f"  {s}: {v:.3f} weights={w}")
    print(f"  PER-SPLIT TOTAL: {total_per_split:.3f}")
