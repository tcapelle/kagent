"""Average pre-computed per-split prediction files from multiple commits.

Writes the mean prediction to the current git commit's output dir.

Run:
  python ensemble_preds.py --agent askeladd --sources 00a6780 1d7feda
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
    sources: list[str] = field(default_factory=list)  # commit hashes to avg
    weights: list[float] = field(default_factory=list)  # optional weights per source
    agent: str | None = None


cfg = sp.parse(Config)
assert len(cfg.sources) >= 1

weights = cfg.weights or [1.0] * len(cfg.sources)
assert len(weights) == len(cfg.sources)
w_sum = sum(weights)

agent = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
final_dir = PREDICTIONS_DIR / agent / commit
staging_dir = PREDICTIONS_DIR / agent / f".{commit}.staging"
if staging_dir.exists():
    import shutil
    shutil.rmtree(staging_dir)
staging_dir.mkdir(parents=True, exist_ok=True)

src_dirs = [PREDICTIONS_DIR / agent / s for s in cfg.sources]
for d in src_dirs:
    assert d.exists(), f"source missing: {d}"

for split in TEST_SPLITS:
    all_preds = [torch.load(d / f"{split}.pt", weights_only=True) for d in src_dirs]
    n = len(all_preds[0])
    assert all(len(p) == n for p in all_preds), "sample count mismatch"

    merged = []
    for i in tqdm(range(n), desc=split, leave=False):
        acc = torch.zeros_like(all_preds[0][i], dtype=torch.float64)
        for preds, w in zip(all_preds, weights):
            acc += preds[i].double() * w
        merged.append((acc / w_sum).float())

    out = staging_dir / f"{split}.pt"
    torch.save(merged, out)
    print(f"  → {out} ({n} samples)")

import shutil
if final_dir.exists():
    shutil.rmtree(final_dir)
staging_dir.rename(final_dir)
print(f"\nMerged predictions saved to {final_dir}")
