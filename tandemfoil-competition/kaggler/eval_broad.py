"""Evaluate broad ensembles with different weighting strategies.

Goal: test whether using more models with simpler (uniform-ish) weights
generalizes better than per-split aggressive optimization.
"""

import math
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
    val_scores: list[float] = field(default_factory=list)  # known val/avg_surf_p per model
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4
    temperature: float = 5.0  # for softmax-weighted ensemble


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"[{len(models)-1}] {ckpt}  val={cfg.val_scores[len(models)-1] if cfg.val_scores else '?'}")

_, val_splits, stats, _ = load_data(cfg.splits_dir)
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


print("\nCaching predictions...")
cache = cache_predictions()
print("Done.")


def eval_weights(weights):
    """Returns dict {split: surf_p_mae} given a weights array of len(models)."""
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


def report(name, weights):
    res = eval_weights(weights)
    avg = sum(res.values()) / len(res)
    parts = "  ".join(f"{n[4:].replace('_',''):>14}={res[n]:.3f}" for n in VAL_SPLIT_NAMES)
    weight_str = " ".join(f"{w:.3f}" for w in weights)
    print(f"\n{name}: avg={avg:.3f}  [{weight_str}]\n  {parts}")


# 1. Uniform
n = len(models)
report("uniform", [1.0 / n] * n)

# 2. Performance-weighted (softmax over -val/T)
if cfg.val_scores:
    # softmax over -val/T
    logits = [-v / cfg.temperature for v in cfg.val_scores]
    m_max = max(logits)
    weights = [math.exp(l - m_max) for l in logits]
    s = sum(weights)
    weights = [w / s for w in weights]
    report(f"perf-softmax T={cfg.temperature}", weights)

    # Linear inverse: w ∝ 1/val
    weights = [1.0 / v for v in cfg.val_scores]
    s = sum(weights)
    weights = [w / s for w in weights]
    report("perf-inverse", weights)

# 3. Top-3-equal: only the 3 best models, equal weight
if cfg.val_scores:
    top3 = sorted(range(n), key=lambda i: cfg.val_scores[i])[:3]
    weights = [0.0] * n
    for i in top3:
        weights[i] = 1.0 / 3
    report(f"top3-equal {top3}", weights)

# 4. Top-5-equal
if cfg.val_scores:
    top5 = sorted(range(n), key=lambda i: cfg.val_scores[i])[:5]
    weights = [0.0] * n
    for i in top5:
        weights[i] = 1.0 / 5
    report(f"top5-equal {top5}", weights)
