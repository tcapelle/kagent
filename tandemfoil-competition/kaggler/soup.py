"""Average weights of multiple checkpoints (model soup) and eval/save.

Run:
  python soup.py --checkpoints "p1,p2,..." [--save_to PATH]
"""

import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm

from data import VAL_SPLIT_NAMES, pad_collate, load_data
from train import Transolver, MODEL_CONFIG


@dataclass
class Config:
    checkpoints: str
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4
    save_to: str | None = None
    eval: bool = True


def main():
    cfg = sp.parse(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_paths = [p.strip() for p in cfg.checkpoints.split(",") if p.strip()]
    print(f"Souping {len(ckpt_paths)} ckpts:")
    for p in ckpt_paths:
        print(f"  {p}")

    sds = [torch.load(p, map_location="cpu", weights_only=True) for p in ckpt_paths]
    avg_sd = {}
    for k in sds[0]:
        if sds[0][k].dtype.is_floating_point:
            avg_sd[k] = sum(sd[k] for sd in sds) / len(sds)
        else:
            avg_sd[k] = sds[0][k]

    if cfg.save_to:
        Path(cfg.save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(avg_sd, cfg.save_to)
        print(f"Saved soup to {cfg.save_to}")

    if not cfg.eval:
        return

    model = Transolver(**MODEL_CONFIG).to(device)
    model.load_state_dict(avg_sd)
    model.eval()

    _, val_splits, stats, _ = load_data(cfg.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}

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
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model({"x": x_n, "mask": mask})["preds"].float()
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                err = (pred_orig - y).abs()
                surf_mask = (mask & is_surface).unsqueeze(-1).float()
                mae_surf += (err * surf_mask).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()

        mae_surf /= max(n_surf, 1)
        print(f"  {split_name}: p={mae_surf[2].item():.2f}")
        avg_p += mae_surf[2].item()

    print(f"\navg/mae_surf_p = {avg_p / len(loaders):.3f}")


if __name__ == "__main__":
    main()
