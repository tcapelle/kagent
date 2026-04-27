"""Evaluate ensemble (and individual ckpts) on the 4 val splits.

Run:
  python eval_ensemble.py --checkpoints "p1,p2,..."
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from train import Transolver, MODEL_CONFIG


@dataclass
class Config:
    checkpoints: str
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4


def main():
    cfg = sp.parse(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = [p.strip() for p in cfg.checkpoints.split(",") if p.strip()]
    print(f"Evaluating ensemble of {len(ckpt_paths)} ckpts:")
    for p in ckpt_paths:
        print(f"  {p}")

    _, val_splits, stats, _ = load_data(cfg.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

    models = []
    for p in ckpt_paths:
        m = Transolver(**MODEL_CONFIG).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        m.eval()
        models.append(m)

    from torch.utils.data import DataLoader
    loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=2)
        for name, ds in val_splits.items()
    }

    print("\nPer-split MAE (surf):")
    avg_p = 0.0
    for split_name, vloader in loaders.items():
        mae_surf = torch.zeros(3, device=device)
        n_surf = 0
        with torch.no_grad():
            for x, y, is_surface, mask in tqdm(vloader, desc=split_name, leave=False):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)

                x_n = (x - stats["x_mean"]) / stats["x_std"]
                preds_norm = None
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    for m in models:
                        p = m({"x": x_n, "mask": mask})["preds"].float()
                        preds_norm = p if preds_norm is None else preds_norm + p
                preds_norm = preds_norm / len(models)

                pred_orig = preds_norm * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                surf_mask = (mask & is_surface).unsqueeze(-1).float()
                mae_surf += (err * surf_mask).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()

        mae_surf /= max(n_surf, 1)
        print(f"  {split_name}: Ux={mae_surf[0].item():.2f}  Uy={mae_surf[1].item():.2f}  p={mae_surf[2].item():.2f}")
        avg_p += mae_surf[2].item()

    print(f"\navg/mae_surf_p = {avg_p / len(loaders):.3f}")


if __name__ == "__main__":
    main()
