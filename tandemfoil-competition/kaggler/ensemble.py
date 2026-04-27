"""Average per-sample test predictions from multiple submission commits.

Output goes to a new commit-tagged directory under apr27-5/<agent>/<commit>/.

Run:
  python ensemble.py --sources <c1> <c2> <c3> --weights 0.2 0.3 0.5 --agent tanjiro
"""

import os
import subprocess
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
    sources: list[str]  # short commit hashes whose predictions to average
    weights: list[float] | None = None  # equal weights if None
    agent: str = "tanjiro"


cfg = sp.parse(Config)
n = len(cfg.sources)
weights = cfg.weights if cfg.weights else [1.0 / n] * n
assert len(weights) == n, f"weights ({len(weights)}) must match sources ({n})"
total = sum(weights)
weights = [w / total for w in weights]

commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "ensemble"
out_dir = PREDICTIONS_DIR / cfg.agent / commit
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Ensemble of {n} sources -> {out_dir}")
for src, w in zip(cfg.sources, weights):
    print(f"  {src}: weight={w:.3f}")

for split in TEST_SPLITS:
    print(f"  {split}...")
    accum = None
    for src, w in zip(cfg.sources, weights):
        path = PREDICTIONS_DIR / cfg.agent / src / f"{split}.pt"
        preds = torch.load(path, weights_only=True)
        if accum is None:
            accum = [w * p for p in preds]
        else:
            for i, p in enumerate(preds):
                accum[i] = accum[i] + w * p
    out_path = out_dir / f"{split}.pt"
    torch.save(accum, out_path)
    print(f"    -> {out_path}")

print(f"Done. Predictions at {out_dir}")
