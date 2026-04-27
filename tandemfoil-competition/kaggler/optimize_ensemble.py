"""Validation-based ensemble weight optimization.

For each candidate model, run inference on the validation splits, save predictions.
Then grid-search (or scipy-optimize) weights to minimize avg surface pressure MAE.

Usage:
  python optimize_ensemble.py --checkpoints ck1.pt ck2.pt ... --agent edward
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
import yaml
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, load_data, pad_collate
from model import Transolver

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)  # paths to checkpoint.pt
    sources: list[str] = field(default_factory=list)  # commit hashes for test predictions
    agent: str = "edward"
    cache_dir: str = "/tmp/edward_val_preds"
    grid_steps: int = 5  # grid resolution for sweep


cfg = sp.parse(Config)
assert len(cfg.checkpoints) == len(cfg.sources), "checkpoints and sources must match"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load val splits + stats
_, val_splits, stats, _ = load_data(SPLITS_DIR)
stats = {k: v.to(device) for k, v in stats.items()}

cache_dir = Path(cfg.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)


def model_val_preds(ckpt_path: str, tag: str) -> dict[str, dict]:
    """Run model on each val split. Returns {split: {pred_phys, y, surf_mask, vol_mask}}.
    Cached by ckpt path tag.
    """
    cache_file = cache_dir / f"{tag}.pt"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=False)
    config_path = Path(ckpt_path).parent / "config.yaml"
    with open(config_path) as f:
        model_cfg = yaml.safe_load(f)
    model = Transolver(**model_cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    out = {}
    for split_name, ds in val_splits.items():
        loader = torch.utils.data.DataLoader(
            ds, batch_size=4, shuffle=False, collate_fn=pad_collate,
            num_workers=2, pin_memory=True,
        )
        all_pred = []
        all_y = []
        all_surf = []
        all_vol = []
        with torch.no_grad():
            for x, y, is_surface, mask in loader:
                x = x.to(device); y = y.to(device)
                is_surface = is_surface.to(device); mask = mask.to(device)
                x_norm = (x - stats["x_mean"]) / stats["x_std"]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_norm = model({"x": x_norm, "mask": mask})["preds"]
                pred_phys = pred_norm.float() * stats["y_std"] + stats["y_mean"]
                vol_mask = mask & ~is_surface
                surf_mask = mask & is_surface
                # collect per-sample (variable length)
                for i in range(x.shape[0]):
                    n = mask[i].sum().item()
                    all_pred.append(pred_phys[i, :n].cpu())
                    all_y.append(y[i, :n].cpu())
                    all_surf.append(is_surface[i, :n].cpu())
                    all_vol.append((~is_surface[i, :n] & mask[i, :n]).cpu())
        out[split_name] = {"pred": all_pred, "y": all_y, "surf": all_surf, "vol": all_vol}
    torch.save(out, cache_file)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# Get predictions for each model
all_preds = []
for ckpt, src in zip(cfg.checkpoints, cfg.sources):
    print(f"Computing val preds for {src}...")
    all_preds.append(model_val_preds(ckpt, src))


def ensemble_metric(weights):
    """Compute weighted avg surface pressure MAE across all val splits."""
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    total_surf_p_mae = 0.0
    for split_name in VAL_SPLIT_NAMES:
        n_surf = 0
        sum_err = 0.0
        n_samples = len(all_preds[0][split_name]["pred"])
        for i in range(n_samples):
            blended = sum(w * all_preds[k][split_name]["pred"][i] for k, w in enumerate(weights))
            y = all_preds[0][split_name]["y"][i]
            surf = all_preds[0][split_name]["surf"][i]
            # surface pressure: channel 2
            err_p = (blended[:, 2] - y[:, 2]).abs()
            sum_err += err_p[surf].sum().item()
            n_surf += surf.sum().item()
        total_surf_p_mae += sum_err / max(n_surf, 1)
    return total_surf_p_mae / len(VAL_SPLIT_NAMES)


# Try uniform first
w_uniform = [1.0] * len(cfg.sources)
print(f"\nUniform: avg_surf_p = {ensemble_metric(w_uniform):.4f}")

# Try inverse-rank (favor first model)
single_scores = []
for k, src in enumerate(cfg.sources):
    score = 0.0
    n_surf = 0
    for split in VAL_SPLIT_NAMES:
        for i in range(len(all_preds[k][split]["pred"])):
            err = (all_preds[k][split]["pred"][i][:, 2] - all_preds[k][split]["y"][i][:, 2]).abs()
            surf = all_preds[k][split]["surf"][i]
            score += err[surf].sum().item()
            n_surf += surf.sum().item()
    single_scores.append(score / max(n_surf, 1))
print("\nSingle-model val avg_surf_p:")
for src, s in zip(cfg.sources, single_scores):
    print(f"  {src}: {s:.4f}")

# Inverse-score weights
w_inv = [1.0 / s for s in single_scores]
print(f"\nInverse-weighted: avg_surf_p = {ensemble_metric(w_inv):.4f}")

# Greedy coordinate descent on weights
best_w = w_inv[:]
best = ensemble_metric(best_w)
print(f"\nStarting greedy: {best:.4f}")
for _ in range(3):
    for k in range(len(best_w)):
        for delta in [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]:
            new_w = best_w[:]
            new_w[k] = max(0.0, new_w[k] * (1 + delta) if delta >= 0 else new_w[k] * (1 + delta))
            score = ensemble_metric(new_w)
            if score < best:
                best = score
                best_w = new_w
                print(f"  k={k} delta={delta}: {score:.4f}")
print(f"Final: {best:.4f}")
print(f"Best weights: {[f'{w:.3f}' for w in best_w]}")

# Normalize and submit
total = sum(best_w)
norm_w = [w / total for w in best_w]
print(f"\nNormalized weights:")
for src, w in zip(cfg.sources, norm_w):
    print(f"  {src}: {w:.3f}")

# Use ensemble.py with these weights
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
out_dir = PREDICTIONS_DIR / cfg.agent / commit
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSubmitting test ensemble to {out_dir}...")
for split in TEST_SPLITS:
    parts = [torch.load(PREDICTIONS_DIR / cfg.agent / s / f"{split}.pt", weights_only=True)
             for s in cfg.sources]
    n = len(parts[0])
    blended = []
    for i in tqdm(range(n), desc=split, leave=False):
        acc = sum(w * parts[k][i] for k, w in enumerate(norm_w))
        blended.append(acc)
    torch.save(blended, out_dir / f"{split}.pt")
    print(f"  → {split}.pt ({n} samples)")

print(f"\nDone. Best val avg_surf_p = {best:.4f}")
