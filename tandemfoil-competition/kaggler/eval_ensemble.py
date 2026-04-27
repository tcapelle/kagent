"""Compute avg_surf_p (leaderboard metric) on val splits for an ensemble.

Run:
  python eval_ensemble.py --checkpoints /tmp/ens/iter6/checkpoint.pt,/tmp/ens/iter7/checkpoint.pt,...
"""

from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, pad_collate, load_data
from train import Transolver


@dataclass
class Config:
    checkpoints: str  # comma-separated checkpoint paths
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 8


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load val splits (no need for train data)
_, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=4, pin_memory=True)
    for name, ds in val_splits.items()
}


def load_model(ckpt_path: str) -> Transolver:
    ckpt = Path(ckpt_path)
    with open(ckpt.parent / "config.yaml") as f:
        mc = yaml.safe_load(f)
    m = Transolver(**mc).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


paths = [p.strip() for p in cfg.checkpoints.split(",") if p.strip()]
models = [load_model(p) for p in paths]
print(f"Ensembling {len(models)} models:")
for p in paths:
    print(f"  - {p}")

avg_surf_p = 0.0
per_split = {}
for split_name, loader in val_loaders.items():
    mae_surf = torch.zeros(3, device=device)
    n_surf = 0
    with torch.no_grad():
        for x, y, is_surf, mask in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            is_surf = is_surf.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            x_in = (x - stats["x_mean"]) / stats["x_std"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = [m({"x": x_in})["preds"] for m in models]
            pred_norm = torch.stack(preds, dim=0).mean(dim=0).float()
            pred_phys = pred_norm * stats["y_std"] + stats["y_mean"]
            err = (pred_phys - y).abs()
            surf_mask = mask & is_surf
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += surf_mask.sum().item()
    mae_surf /= max(n_surf, 1)
    per_split[split_name] = mae_surf[2].item()
    avg_surf_p += mae_surf[2].item()
    print(f"  {split_name}: mae_surf_p = {mae_surf[2].item():.4f}")

avg_surf_p /= len(val_loaders)
print(f"\navg_surf_p (val) = {avg_surf_p:.4f}")
