"""Average predictions across multiple commits/checkpoints to form an ensemble.

Usage:
  python ensemble.py --commits b4d6f23 dd27e19 --agent edward
The averaged predictions are saved to /mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<HEAD>/.
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    commits: list[str] = field(default_factory=list)
    agent: str = "unknown"
    weights: list[float] | None = None  # optional per-commit weights; defaults to uniform


cfg = sp.parse(Config)
assert cfg.commits, "Pass --commits <hash1> <hash2> ..."

if cfg.weights is None:
    cfg.weights = [1.0] * len(cfg.commits)
assert len(cfg.weights) == len(cfg.commits)
total_w = sum(cfg.weights)

head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out_dir = PREDICTIONS_DIR / cfg.agent / head
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Ensemble of {len(cfg.commits)} commits → {out_dir}")
for c, w in zip(cfg.commits, cfg.weights):
    print(f"  {c}  w={w}")

for split in TEST_SPLITS:
    paths = [PREDICTIONS_DIR / cfg.agent / c / f"{split}.pt" for c in cfg.commits]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
    pred_lists = [torch.load(p, weights_only=True) for p in paths]
    n_samples = len(pred_lists[0])
    avg = []
    for i in range(n_samples):
        stacked = torch.stack([w * pl[i] for pl, w in zip(pred_lists, cfg.weights)], dim=0)
        avg.append(stacked.sum(0) / total_w)
    out = out_dir / f"{split}.pt"
    torch.save(avg, out)
    print(f"  {split}: averaged {n_samples} samples → {out}")

print(f"\nDone. Saved to {out_dir}")
