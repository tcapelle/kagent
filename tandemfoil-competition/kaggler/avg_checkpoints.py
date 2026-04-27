"""Average two checkpoint state dicts (SWA-lite) into a new checkpoint.

Useful for combining e.g. v3 and v7 (same architecture, related minima)
into a single model that averages their parameters.

Run:
  python avg_checkpoints.py \
      --ckpt_a models/model-A/checkpoint.pt \
      --ckpt_b models/model-B/checkpoint.pt \
      --weight_a 0.5 \
      --out models/model-AVG/checkpoint.pt
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch


@dataclass
class Config:
    ckpt_a: str
    ckpt_b: str
    out: str
    weight_a: float = 0.5  # weight on A (B gets 1-weight_a)


cfg = sp.parse(Config)
sd_a = torch.load(cfg.ckpt_a, map_location="cpu", weights_only=True)
sd_b = torch.load(cfg.ckpt_b, map_location="cpu", weights_only=True)

assert set(sd_a.keys()) == set(sd_b.keys()), "checkpoint keys must match"

w = cfg.weight_a
sd_avg = {k: w * sd_a[k].float() + (1 - w) * sd_b[k].float() for k in sd_a}

out_path = Path(cfg.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(sd_avg, out_path)

# Copy config.yaml (architectures must match)
config_a = Path(cfg.ckpt_a).parent / "config.yaml"
shutil.copy2(config_a, out_path.parent / "config.yaml")
print(f"Averaged ({w}*A + {1-w}*B) → {out_path}")
