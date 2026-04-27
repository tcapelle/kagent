"""Evaluate a Transolver checkpoint on the 4 val splits.

Reports avg_surf_p plus per-channel/per-split MAE in physical units.
"""

from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, load_data, pad_collate
from model import Transolver


@dataclass
class Config:
    checkpoint: str
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 8
    config_yaml: str | None = None


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = Path(cfg.checkpoint)
config_path = Path(cfg.config_yaml) if cfg.config_yaml else (ckpt_path.parent / "config.yaml")
if not config_path.exists():
    # fallback default
    model_config = dict(space_dim=2, fun_dim=22, out_dim=3,
                        n_hidden=192, n_layers=6, n_head=6, slice_num=64, mlp_ratio=2,
                        output_fields=["Ux","Uy","p"], output_dims=[1,1,1])
else:
    with open(config_path) as f:
        model_config = yaml.safe_load(f)

model = Transolver(**model_config).to(device)
model.load_state_dict(torch.load(cfg.checkpoint, map_location=device, weights_only=True))
model.eval()
print(f"Loaded {cfg.checkpoint}")

_, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2)
    for name, ds in val_splits.items()
}

split_sp: dict[str, float] = {}
for split, vloader in loaders.items():
    mae_surf = torch.zeros(3, device=device)
    n_surf = 0
    with torch.no_grad():
        for x, y, is_surface, mask in vloader:
            x = x.to(device); y = y.to(device)
            is_surface = is_surface.to(device); mask = mask.to(device)
            x_n = (x - stats["x_mean"]) / stats["x_std"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model({"x": x_n})["preds"]
            pred = pred.float() * stats["y_std"] + stats["y_mean"]
            err = (pred - y).abs()
            surf_mask = mask & is_surface
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0,1))
            n_surf += surf_mask.sum().item()
    mae_surf /= max(n_surf, 1)
    split_sp[split] = mae_surf[2].item()
    print(f"{split:30s} mae_surf_Ux={mae_surf[0]:.3f}  mae_surf_Uy={mae_surf[1]:.3f}  mae_surf_p={mae_surf[2]:.2f}")

avg = sum(split_sp.values()) / len(split_sp)
print(f"\navg_surf_p = {avg:.3f}")
