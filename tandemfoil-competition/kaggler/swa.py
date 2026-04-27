"""Average several Transolver checkpoints (SWA) and save the averaged model.

Usage:
  python swa.py --ckpts ckpt1.pt ckpt2.pt ckpt3.pt --out swa.pt
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import simple_parsing as sp
import torch


@dataclass
class Config:
    ckpts: List[str]
    out: str = "checkpoints/swa.pt"


cfg = sp.parse(Config)

states = [torch.load(p, map_location="cpu", weights_only=True) for p in cfg.ckpts]
print(f"Averaging {len(states)} checkpoints")

avg = {}
for k in states[0]:
    if states[0][k].dtype.is_floating_point:
        avg[k] = sum(s[k] for s in states) / len(states)
    else:
        avg[k] = states[0][k]  # keep buffers (e.g. int) from first ckpt

Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(avg, cfg.out)
print(f"Saved SWA → {cfg.out}")
