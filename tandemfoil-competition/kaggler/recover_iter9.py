"""Recover iter9's predictions from the iter16 4-way equal ensemble.

iter16 (commit 1c1f0f7) = (iter3 + iter4 + iter9 + iter15) / 4
So: iter9 = 4 * iter16 - iter3 - iter4 - iter15
"""

import os
from pathlib import Path

import torch

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PRED_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}/frieren")

ITER16 = "1c1f0f7"
ITER3 = "2c929ae"
ITER4 = "1509e10"
ITER15 = "a2e0b1a"

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]

out = PRED_DIR / "iter9_recovered"
out.mkdir(parents=True, exist_ok=True)

for split in TEST_SPLITS:
    p16 = torch.load(PRED_DIR / ITER16 / f"{split}.pt", weights_only=False)
    p3 = torch.load(PRED_DIR / ITER3 / f"{split}.pt", weights_only=False)
    p4 = torch.load(PRED_DIR / ITER4 / f"{split}.pt", weights_only=False)
    p15 = torch.load(PRED_DIR / ITER15 / f"{split}.pt", weights_only=False)

    p9 = [4 * p16[i] - p3[i] - p4[i] - p15[i] for i in range(len(p16))]
    torch.save(p9, out / f"{split}.pt")
    print(f"{split}: recovered {len(p9)} samples")

print(f"\nSaved recovered iter9 predictions to {out}")
