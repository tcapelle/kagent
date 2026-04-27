"""Average predictions across multiple commits and submit under HEAD commit.

Usage:
  python ensemble.py --sources 79894e7 0fa22ab 5d05ebb --weights 0.2 0.3 0.5
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm

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
    agent: str = "edward"


cfg = sp.parse(Config)
assert cfg.sources, "--sources required"
weights = cfg.weights or [1.0 / len(cfg.sources)] * len(cfg.sources)
assert len(weights) == len(cfg.sources), "weights len must match sources"
total = sum(weights)
weights = [w / total for w in weights]

commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
out_dir = PREDICTIONS_DIR / cfg.agent / commit
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Sources: {list(zip(cfg.sources, weights))}")
print(f"Output:  {out_dir}")

for split in TEST_SPLITS:
    parts = []
    for src in cfg.sources:
        p = PREDICTIONS_DIR / cfg.agent / src / f"{split}.pt"
        parts.append(torch.load(p, weights_only=True))
    n_samples = len(parts[0])
    blended = []
    for i in tqdm(range(n_samples), desc=split, leave=False):
        acc = sum(w * parts[k][i] for k, w in enumerate(weights))
        blended.append(acc)
    out_path = out_dir / f"{split}.pt"
    torch.save(blended, out_path)
    print(f"  → {out_path} ({n_samples} samples)")

print(f"\nEnsemble saved to {out_dir}")
