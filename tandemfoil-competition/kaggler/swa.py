"""Stochastic Weight Averaging: average weights of multiple checkpoints.

Usage:
  python swa.py --checkpoints v14/checkpoint.pt v15/checkpoint.pt v16/checkpoint.pt --out checkpoints/swa.pt
"""

from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    out: str = "checkpoints/swa.pt"


cfg = sp.parse(Config)
assert len(cfg.checkpoints) >= 2

weights = cfg.weights or [1.0] * len(cfg.checkpoints)
assert len(weights) == len(cfg.checkpoints)
w_sum = sum(weights)

states = []
for p, w in zip(cfg.checkpoints, weights):
    sd = torch.load(p, map_location="cpu", weights_only=True)
    states.append(sd)
    print(f"Loaded {p}  weight={w}")

avg = {}
for key in states[0].keys():
    tensors = [s[key].float() for s in states]
    acc = sum(t * w for t, w in zip(tensors, weights)) / w_sum
    avg[key] = acc.to(states[0][key].dtype)

out_path = Path(cfg.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(avg, out_path)
print(f"\nSaved SWA checkpoint to {out_path}")
