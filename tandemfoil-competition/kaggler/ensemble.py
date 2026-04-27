"""Ensemble predictions from multiple commits by weighted average.

Reads test_*.pt prediction files from
  /mnt/new-pvc/predictions/<RESEARCH_TAG>/<agent>/<source-commit>/
for each given source commit, then averages them with given weights and writes
to the current commit's output dir.

Run:
  python ensemble.py --sources def6b08 6d4b236 d6baae4 --weights 0.2 0.4 0.4
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
    sources: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    agent: str = "alphonse"


cfg = sp.parse(Config)
assert len(cfg.sources) == len(cfg.weights), "sources and weights must match"
weights = [w / sum(cfg.weights) for w in cfg.weights]
print(f"Ensembling {len(cfg.sources)} sources with normalized weights {weights}")

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
output_dir = PREDICTIONS_DIR / cfg.agent / commit
output_dir.mkdir(parents=True, exist_ok=True)

for split in TEST_SPLITS:
    averaged = None
    for src, w in zip(cfg.sources, weights):
        path = PREDICTIONS_DIR / cfg.agent / src / f"{split}.pt"
        preds = torch.load(path, weights_only=False)
        if averaged is None:
            averaged = [w * p for p in preds]
        else:
            for i, p in enumerate(preds):
                averaged[i] = averaged[i] + w * p
    out_path = output_dir / f"{split}.pt"
    torch.save(averaged, out_path)
    print(f"{split}: averaged {len(averaged)} samples → {out_path}")

print(f"\nEnsemble saved to {output_dir}")
