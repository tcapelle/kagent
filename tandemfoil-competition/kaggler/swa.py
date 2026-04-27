"""Stochastic Weight Averaging across multiple Transolver checkpoints.

Averages model parameters in weight-space. Works best when the inputs are
on the same training trajectory (e.g. successive chain warm-starts).

Run:
  python swa.py --checkpoints A.pt,B.pt,C.pt --out swa_out.pt
  python eval_val.py --checkpoints swa_out.pt   # check val score
"""

import json
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml

from data import X_DIM
from model import Transolver


@dataclass
class Config:
    checkpoints: str  # comma-separated paths
    out: str = "checkpoints/swa.pt"
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"


cfg = sp.parse(Config)

ckpts = [c.strip() for c in cfg.checkpoints.split(",") if c.strip()]
print(f"Averaging {len(ckpts)} checkpoints:")
for c in ckpts:
    print(f"  - {c}")

# Read first checkpoint's config to infer architecture.
mc_path = Path(ckpts[0]).parent / "config.yaml"
if mc_path.exists():
    with open(mc_path) as f:
        mc = yaml.safe_load(f)
else:
    mc = dict(
        space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
        n_hidden=192, n_layers=6, n_head=6, slice_num=128, mlp_ratio=2,
        output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
    )

# Average state dicts.
sds = [torch.load(c, map_location="cpu", weights_only=True) for c in ckpts]
swa_sd: dict[str, torch.Tensor] = {}
for k in sds[0].keys():
    stacked = torch.stack([sd[k].float() for sd in sds], dim=0)
    swa_sd[k] = stacked.mean(dim=0).to(sds[0][k].dtype)

# Sanity: make sure it loads.
m = Transolver(**mc)
m.load_state_dict(swa_sd)
print("SWA model loaded successfully.")

out_path = Path(cfg.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(swa_sd, out_path)

cfg_path = out_path.parent / "config.yaml"
with open(cfg_path, "w") as f:
    yaml.dump(mc, f)

print(f"Saved SWA checkpoint to {out_path}")
print(f"Saved config to {cfg_path}")
