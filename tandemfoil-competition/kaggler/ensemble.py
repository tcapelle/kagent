"""Average predictions from multiple already-saved test runs.

Run:
  python ensemble.py --sources <commit1> <commit2> ... --weights w1 w2 ... --agent <name>

Reads from /mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<commit>/test_*.pt for each source,
weighted-averages, and writes to /mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<HEAD>/test_*.pt.
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
    sources: list[str] = field(default_factory=list)  # list of commit hashes
    weights: list[float] = field(default_factory=list)  # parallel to sources
    agent: str = "alphonse"


cfg = sp.parse(Config)
assert len(cfg.sources) > 0, "need at least one source"
if not cfg.weights:
    cfg.weights = [1.0 / len(cfg.sources)] * len(cfg.sources)
assert len(cfg.weights) == len(cfg.sources), "weights must match sources"

w_sum = sum(cfg.weights)
weights = [w / w_sum for w in cfg.weights]
print(f"Ensembling {len(cfg.sources)} sources with weights {weights}")

commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
out_dir = PREDICTIONS_DIR / cfg.agent / commit
out_dir.mkdir(parents=True, exist_ok=True)

for split in TEST_SPLITS:
    accum: list[torch.Tensor] | None = None
    for src, w in zip(cfg.sources, weights):
        src_path = PREDICTIONS_DIR / cfg.agent / src / f"{split}.pt"
        preds = torch.load(src_path, weights_only=True)
        if accum is None:
            accum = [w * p for p in preds]
        else:
            for i, p in enumerate(preds):
                accum[i] = accum[i] + w * p
    out_path = out_dir / f"{split}.pt"
    torch.save(accum, out_path)
    print(f"  → {out_path} ({len(accum)} samples)")

print(f"Ensemble saved to {out_dir}")
