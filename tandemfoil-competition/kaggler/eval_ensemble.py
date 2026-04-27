"""Evaluate ensemble of checkpoints on the val splits.

Reports avg surface pressure MAE for each weight configuration so we can
pick the optimal ensemble weights before submitting to the test server.
"""

import json
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
    weight_configs: list[str] = field(default_factory=lambda: [
        "1.0,0.0", "0.0,1.0", "0.5,0.5", "0.6,0.4", "0.7,0.3", "0.8,0.2",
    ])
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load models
models = []
for ckpt in cfg.checkpoints:
    ckpt_path = Path(ckpt)
    config_path = ckpt_path.parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
    print(f"Loaded: {ckpt}")

# Load val splits and stats
_, val_splits, stats, _ = load_data(splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2, pin_memory=True)
    for name, ds in val_splits.items()
}


@torch.no_grad()
def eval_weights(weights):
    weights = [w / sum(weights) for w in weights]
    split_metrics = {}
    for split_name, vloader in val_loaders.items():
        mae_surf = torch.zeros(3, device=device)
        n_surf = 0
        for x, y, is_surface, mask in vloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            ensembled = None
            for m, w in zip(models, weights):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = m({"x": x_norm})["preds"].float()
                p_phys = p * stats["y_std"] + stats["y_mean"]
                ensembled = p_phys * w if ensembled is None else ensembled + p_phys * w
            err = (ensembled - y).abs()
            surf_mask = mask & is_surface
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += surf_mask.sum().item()
        mae_surf /= max(n_surf, 1)
        split_metrics[split_name] = mae_surf[2].item()
    avg = sum(split_metrics.values()) / len(split_metrics)
    return avg, split_metrics


print("\n=== Ensemble weight sweep (val avg surface pressure MAE) ===")
print(f"{'Weights':<25} {'avg':>7}  " + "  ".join(f"{n[4:].replace('_', ''):>10}" for n in VAL_SPLIT_NAMES))
print("-" * 90)
for cfg_str in cfg.weight_configs:
    weights = [float(w) for w in cfg_str.split(",")]
    if len(weights) != len(models):
        continue
    avg, splits_data = eval_weights(weights)
    splits_str = "  ".join(f"{splits_data[n]:>10.3f}" for n in VAL_SPLIT_NAMES)
    print(f"{cfg_str:<25} {avg:>7.3f}  {splits_str}")
