"""Evaluate ALL ensemble configs across N models on val per-split.

For each split, find the configuration with lowest surf_p MAE.
Reports per-split optima so we can submit per-split-optimized predictions.
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
    weight_configs: list[str] = field(default_factory=list)
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load models
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

_, val_splits, stats, _ = load_data(splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2, pin_memory=True)
    for name, ds in val_splits.items()
}


@torch.no_grad()
def cache_predictions():
    """For each model and val split, cache predictions as a list-of-batch tensors."""
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
            "preds": per_model_preds,  # list of M lists of B-tensors
            "y": ys,
            "surf_mask": surf_masks,
        }
    return cache


print("Caching predictions...")
cache = cache_predictions()
print("Done.")


def eval_weights_per_split(weights):
    weights = [w / sum(weights) for w in weights]
    out = {}
    for split_name, c in cache.items():
        total_err = 0.0
        total_count = 0
        for batch_idx in range(len(c["y"])):
            ensembled = sum(c["preds"][i][batch_idx] * weights[i] for i in range(len(weights)))
            err = (ensembled - c["y"][batch_idx]).abs()
            total_err += (err[..., 2] * c["surf_mask"][batch_idx]).sum().item()
            total_count += c["surf_mask"][batch_idx].sum().item()
        out[split_name] = total_err / max(total_count, 1)
    return out


print("\n=== Per-split sweep ===")
results = []
for cfg_str in cfg.weight_configs:
    weights = [float(w) for w in cfg_str.split(",")]
    if len(weights) != len(models):
        continue
    splits_data = eval_weights_per_split(weights)
    avg = sum(splits_data.values()) / len(splits_data)
    results.append((cfg_str, avg, splits_data))

# Print per-split table
print(f"{'Weights':<30} {'avg':>7}  " + "  ".join(f"{n[4:].replace('_',''):>12}" for n in VAL_SPLIT_NAMES))
print("-" * 100)
for cfg_str, avg, splits_data in results:
    splits_str = "  ".join(f"{splits_data[n]:>12.3f}" for n in VAL_SPLIT_NAMES)
    print(f"{cfg_str:<30} {avg:>7.3f}  {splits_str}")

# Find per-split optimum
print("\n=== Per-split optimum ===")
total = 0.0
for split in VAL_SPLIT_NAMES:
    best_cfg, best_val = None, float("inf")
    for cfg_str, avg, splits_data in results:
        if splits_data[split] < best_val:
            best_val = splits_data[split]
            best_cfg = cfg_str
    print(f"{split}: {best_cfg} → {best_val:.3f}")
    total += best_val
print(f"\nIf using per-split optima: avg = {total / 4:.3f}")
