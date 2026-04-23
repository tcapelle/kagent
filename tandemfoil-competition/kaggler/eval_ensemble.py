"""Evaluate ensemble of checkpoints on validation splits.

Reports l2_error (same metric as leaderboard) and per-split losses for
a single checkpoint or an ensemble (comma-separated list).

Usage:
  python eval_ensemble.py --checkpoints p1,p2,...
"""

import json
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from model import Transolver


@dataclass
class Config:
    checkpoints: str
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 2


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt_paths = [Path(p.strip()) for p in cfg.checkpoints.split(",") if p.strip()]


def load_model(path: Path) -> Transolver:
    config_path = path.parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


models = [load_model(p) for p in ckpt_paths]
for p in ckpt_paths:
    print(f"Loaded: {p}")
print(f"Ensemble size: {len(models)}")

_, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=4, pin_memory=True)
    for name, ds in val_splits.items()
}

print()
mean_l2 = 0.0
with torch.no_grad():
    for split_name, vloader in val_loaders.items():
        l2_sum = 0.0
        n_all = 0
        for x, y, is_surface, mask in vloader:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_norm = sum(m({"x": x_norm})["preds"] for m in models) / len(models)
            pred_phys = pred_norm.float() * stats["y_std"] + stats["y_mean"]
            vel_diff_sq = (pred_phys[..., :2] - y[..., :2]) ** 2
            l2_vel = vel_diff_sq.sum(-1).sqrt()
            l2_sum += (l2_vel * mask).sum().item()
            n_all += mask.sum().item()
        split_l2 = l2_sum / max(n_all, 1)
        mean_l2 += split_l2
        print(f"  {split_name}: l2={split_l2:.4f}")

mean_l2 /= len(val_loaders)
print(f"\nEnsemble val/l2_error (mean across splits): {mean_l2:.4f}")
