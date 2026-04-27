"""Compute val avg_surf_p for one or more checkpoints (single or ensemble).

Run:
  python eval_val.py --checkpoints A.pt,B.pt
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
    checkpoints: str  # comma-separated
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    batch_size: int = 4
    bf16: bool = True


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(p: str) -> Transolver:
    p = Path(p)
    mc_path = p.parent / "config.yaml"
    if mc_path.exists():
        with open(mc_path) as f:
            mc = yaml.safe_load(f)
    else:
        mc = dict(space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
                  n_hidden=192, n_layers=6, n_head=6, slice_num=128, mlp_ratio=2,
                  output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1])
    m = Transolver(**mc).to(device)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    m.eval()
    return m


ckpts = [c.strip() for c in cfg.checkpoints.split(",") if c.strip()]
print(f"Loading {len(ckpts)} checkpoint(s):", ckpts)
models = [_load_model(c) for c in ckpts]

_, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}
val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2, pin_memory=True)
    for name, ds in val_splits.items()
}

per_split = {}
for split_name, vloader in val_loaders.items():
    mae_surf = torch.zeros(3, device=device)
    mae_vol = torch.zeros(3, device=device)
    n_surf = n_vol = 0
    with torch.no_grad():
        for x, y, is_surface, mask in vloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]

            pred_sum = torch.zeros(*x.shape[:2], 3, device=device)
            for m in models:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    pn = m({"x": x_norm})["preds"]
                pred_sum += pn.float() * stats["y_std"] + stats["y_mean"]
            pred_orig = pred_sum / len(models)

            err = (pred_orig - y).abs()
            vol_mask = mask & ~is_surface
            surf_mask = mask & is_surface
            mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
            mae_vol += (err * vol_mask.unsqueeze(-1)).sum(dim=(0, 1))
            n_surf += surf_mask.sum().item()
            n_vol += vol_mask.sum().item()
    mae_surf /= max(n_surf, 1)
    mae_vol /= max(n_vol, 1)
    per_split[split_name] = (mae_surf[2].item(), mae_surf.tolist(), mae_vol.tolist())

print()
print(f"{'split':30s}  surf_p   surf_Ux  surf_Uy   vol_Ux   vol_Uy   vol_p")
total = 0.0
for name in VAL_SPLIT_NAMES:
    sp_p, surf, vol = per_split[name]
    print(f"{name:30s}  {sp_p:7.3f}  {surf[0]:7.3f}  {surf[1]:7.3f}  {vol[0]:7.3f}  {vol[1]:7.3f}  {vol[2]:7.3f}")
    total += sp_p
print(f"{'AVG surf_p':30s}  {total/len(VAL_SPLIT_NAMES):7.3f}")
