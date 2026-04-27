"""Quick val MAE eval for an ensemble of checkpoints.

Loads multiple ckpts, runs on val splits, averages predictions, computes MAE.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from train import ResMLP, TransolverNet


@dataclass
class Config:
    checkpoints: list[str]
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4


def load_model(ckpt_path: str, device):
    cfg_path = Path(ckpt_path).parent / "config.yaml"
    with open(cfg_path) as f:
        mc = yaml.safe_load(f)
    arch = mc.pop("arch", "resmlp")
    m = (TransolverNet if arch == "transolver" else ResMLP)(**mc).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    m.eval()
    return m


def main():
    cfg = sp.parse(Config)
    device = torch.device("cuda")
    _, val_splits, stats, _ = load_data(cfg.splits_dir)
    stats = {k: v.to(device) for k, v in stats.items()}
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=2)
        for name, ds in val_splits.items()
    }
    models = [load_model(p, device) for p in cfg.checkpoints]
    print(f"Loaded {len(models)} models")
    print(f"Per-model params: {[sum(p.numel() for p in m.parameters())/1e6 for m in models]} M")

    surf_p_per_split = {}
    for split_name, vloader in val_loaders.items():
        mae_surf = torch.zeros(3, device=device)
        n_surf = 0
        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device)
                mask = mask.to(device)
                x_n = (x - stats["x_mean"]) / stats["x_std"]
                pred_sum = None
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for m in models:
                        pred = m({"x": x_n, "mask": mask})["preds"]
                        pred_sum = pred if pred_sum is None else pred_sum + pred
                pred = pred_sum / len(models)
                pred = pred.float() * stats["y_std"] + stats["y_mean"]
                err = (pred - y).abs()
                surf_mask = mask & is_surface
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
        mae_surf /= max(n_surf, 1)
        surf_p_per_split[split_name] = mae_surf[2].item()
        print(f"{split_name:30s} mae_surf_p = {mae_surf[2].item():.3f}")
    mean_surf_p = sum(surf_p_per_split.values()) / len(surf_p_per_split)
    print(f"\nMean surf_p MAE across {len(val_splits)} splits: {mean_surf_p:.3f}")


if __name__ == "__main__":
    main()
