"""Evaluate one or more checkpoints (averaged) on the val splits.

Reports avg_surf_p (the leaderboard metric) plus per-split breakdown.

Run:
  python eval_ensemble.py --checkpoints a.pt,b.pt
"""

import json
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, pad_collate, load_data, X_DIM
from model import Transolver


@dataclass
class Config:
    checkpoints: str
    weights: str | None = None
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

ckpt_paths = [Path(p.strip()) for p in cfg.checkpoints.split(",") if p.strip()]
if cfg.weights:
    weights = [float(w) for w in cfg.weights.split(",")]
else:
    weights = [1.0] * len(ckpt_paths)
total_w = sum(weights)
weights = [w / total_w for w in weights]


def load(path: Path) -> Transolver:
    mc_path = path.parent / "config.yaml"
    if mc_path.exists():
        with open(mc_path) as f:
            mc = yaml.safe_load(f)
    else:
        mc = dict(space_dim=0, fun_dim=X_DIM, out_dim=3,
                  n_hidden=192, n_layers=6, n_head=6, slice_num=128, mlp_ratio=2,
                  output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1])
    m = Transolver(**mc).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


models = [load(p) for p in ckpt_paths]
print(f"Loaded {len(models)} models, weights={weights}")

_, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

avg_surf_p = 0.0
per_split = {}
for name, ds in val_splits.items():
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=2)
    mae = torch.zeros(3, device=device)
    n_surf = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x, y = x.to(device), y.to(device)
            is_surface = is_surface.to(device)
            mask = mask.to(device)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]

            pred_avg = None
            for m, w in zip(models, weights):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pn = m({"x": x_norm})["preds"].float()
                pp = pn * stats["y_std"] + stats["y_mean"]
                pred_avg = pp * w if pred_avg is None else pred_avg + pp * w

            err = (pred_avg - y).abs()
            surf_mask = mask & is_surface
            mae += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += surf_mask.sum().item()
    mae = mae / max(n_surf, 1)
    per_split[name] = mae[2].item()
    avg_surf_p += mae[2].item()
    print(f"{name}: surf_p={mae[2].item():.3f} surf_Ux={mae[0].item():.3f} surf_Uy={mae[1].item():.3f}")

avg_surf_p /= len(val_splits)
print(f"\navg_surf_p = {avg_surf_p:.3f}")
