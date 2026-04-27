"""One-off: compute val predictions for a single checkpoint and cache."""

import os
import sys
from pathlib import Path

import torch
import yaml
from data import VAL_SPLIT_NAMES, load_data, pad_collate
from model import Transolver

CACHE = Path("/tmp/edward_val_preds")
CACHE.mkdir(parents=True, exist_ok=True)

ckpt_path = sys.argv[1]
tag = sys.argv[2]
device = torch.device("cuda")

_, val_splits, stats, _ = load_data()
stats = {k: v.to(device) for k, v in stats.items()}

cfg_path = Path(ckpt_path).parent / "config.yaml"
with open(cfg_path) as f:
    model_cfg = yaml.safe_load(f)
model = Transolver(**model_cfg).to(device)
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()

out = {}
for sn, ds in val_splits.items():
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, shuffle=False, collate_fn=pad_collate,
        num_workers=2, pin_memory=True,
    )
    preds, ys, surfs, vols = [], [], [], []
    with torch.no_grad():
        for x, y, is_surf, mask in loader:
            x = x.to(device); y = y.to(device)
            is_surf = is_surf.to(device); mask = mask.to(device)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pn = model({"x": x_norm, "mask": mask})["preds"]
            pp = pn.float() * stats["y_std"] + stats["y_mean"]
            for i in range(x.shape[0]):
                n = mask[i].sum().item()
                preds.append(pp[i, :n].cpu())
                ys.append(y[i, :n].cpu())
                surfs.append(is_surf[i, :n].cpu())
                vols.append((~is_surf[i, :n] & mask[i, :n]).cpu())
    out[sn] = {"pred": preds, "y": ys, "surf": surfs, "vol": vols}

torch.save(out, CACHE / f"{tag}.pt")
print(f"Saved {tag}")
