"""Average predictions from multiple commits (or any .pt prediction dirs).

Usage:
  python ensemble.py --sources "path/to/commit_a,path/to/commit_b" --agent tanjiro [--weights "0.5,0.5"]

Each source directory must contain the 4 test_*.pt files, each a list of tensors [N_i, 3].
Output goes to /mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/ensemble-<hash>/.
"""

import hashlib
import os
from dataclasses import dataclass
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
    sources: str  # comma-separated list of prediction directories
    agent: str = "tanjiro"
    weights: str = ""  # comma-separated weights, defaults to uniform


cfg = sp.parse(Config)

sources = [Path(s.strip()) for s in cfg.sources.split(",") if s.strip()]
assert len(sources) >= 2, f"need >=2 sources, got {sources}"

if cfg.weights:
    weights = [float(w) for w in cfg.weights.split(",")]
else:
    weights = [1.0 / len(sources)] * len(sources)
assert len(weights) == len(sources), "weights/sources length mismatch"
assert abs(sum(weights) - 1.0) < 1e-6, f"weights must sum to 1: {weights}"

# Hash of source commits + weights for a stable output directory name.
sig = "|".join(f"{s.name}:{w:.3f}" for s, w in zip(sources, weights))
tag = hashlib.sha1(sig.encode()).hexdigest()[:8]
out_dir = PREDICTIONS_DIR / cfg.agent / f"ensemble-{tag}"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Ensemble of {len(sources)} sources -> {out_dir}")
for s, w in zip(sources, weights):
    print(f"  {w:.3f}  {s}")

for split in TEST_SPLITS:
    ys_per_source = []
    for s in sources:
        preds = torch.load(s / f"{split}.pt", weights_only=True)
        ys_per_source.append(preds)
    # Sanity check lengths
    n = len(ys_per_source[0])
    for ys in ys_per_source:
        assert len(ys) == n
    merged = []
    for i in range(n):
        # Weighted average in physical space
        stacked = torch.stack([ys[i] for ys in ys_per_source], dim=0)  # [S, N, 3]
        w = torch.tensor(weights, dtype=stacked.dtype).view(-1, 1, 1)
        merged.append((stacked * w).sum(dim=0))  # [N, 3]
    torch.save(merged, out_dir / f"{split}.pt")
    print(f"  {split}: merged {n} samples -> {out_dir / f'{split}.pt'}")

print(f"\nAll predictions saved to {out_dir}")
