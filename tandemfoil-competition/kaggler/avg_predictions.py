"""Average prediction files from two existing submissions.

Produces a meta-ensemble: average of two model ensembles' predictions, saved
under the current git commit hash.
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
    sources: list[str] = field(default_factory=list)  # commit hashes to avg
    weights: list[float] = field(default_factory=list)
    agent: str = "edward"


cfg = sp.parse(Config)
if not cfg.weights:
    cfg.weights = [1.0 / len(cfg.sources)] * len(cfg.sources)
weights = [w / sum(cfg.weights) for w in cfg.weights]
print(f"Averaging {len(cfg.sources)} sources with weights {weights}")

commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / cfg.agent / commit
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output: {output_dir}")

for split in TEST_SPLITS:
    src_lists = []
    for src in cfg.sources:
        path = PREDICTIONS_DIR / cfg.agent / src / f"{split}.pt"
        preds = torch.load(path, weights_only=False)
        src_lists.append(preds)
        print(f"  {src}/{split}: {len(preds)} samples")

    # Average predictions per sample
    n_samples = len(src_lists[0])
    averaged = []
    for i in range(n_samples):
        avg = sum(src_lists[k][i] * weights[k] for k in range(len(weights)))
        averaged.append(avg)

    out_path = output_dir / f"{split}.pt"
    torch.save(averaged, out_path)
    print(f"  -> {out_path}")

print("Done.")
