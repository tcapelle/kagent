"""Quick val evaluation of a checkpoint — same metrics as train.py validation phase."""

import json
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from dataclasses import dataclass
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, pad_collate, load_data
from model import Transolver


@dataclass
class Config:
    checkpoint: str
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 2
    bf16: bool = True


cfg = sp.parse(Config)
device = torch.device("cuda")

train_ds, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

ckpt = Path(cfg.checkpoint)
model_config = yaml.safe_load(open(ckpt.parent / "config.yaml"))
model = Transolver(**model_config).to(device).eval()
state = torch.load(ckpt, map_location=device, weights_only=True)
model.load_state_dict(state)
print(f"Loaded {ckpt}")

autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if cfg.bf16 else torch.amp.autocast("cuda", enabled=False)

split_surf_p = {}
for split_name in VAL_SPLIT_NAMES:
    loader = DataLoader(val_splits[split_name], batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=pad_collate, num_workers=2)
    mae_surf = torch.zeros(3, device=device)
    n_surf = 0
    with torch.no_grad():
        for x, y, is_surface, mask in loader:
            x, y = x.to(device), y.to(device)
            is_surface = is_surface.to(device); mask = mask.to(device)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            with autocast_ctx:
                pred = model({"x": x_norm})["preds"].float()
            pred_phys = pred * stats["y_std"] + stats["y_mean"]
            err = (pred_phys - y).abs()
            surf_mask = mask & is_surface
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += surf_mask.sum().item()
    mae_surf = (mae_surf / max(n_surf, 1)).cpu()
    split_surf_p[split_name] = mae_surf[2].item()
    print(f"{split_name}: surf_Ux={mae_surf[0]:.3f}  surf_Uy={mae_surf[1]:.3f}  surf_p={mae_surf[2]:.3f}")

avg_surf_p = sum(split_surf_p.values()) / len(split_surf_p)
print(f"\navg surf_p: {avg_surf_p:.4f}")
