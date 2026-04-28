"""Average per-sample predictions from multiple commits and submit ensemble."""

import os
import subprocess
from pathlib import Path
import torch

PRED_DIR = Path(f"/mnt/new-pvc/predictions/{os.environ['RESEARCH_TAG']}/frieren")
SPLITS = ["test_single_in_dist", "test_geom_camber_rc", "test_geom_camber_cruise", "test_re_rand"]

# Strong-only 11-way: 10-way + iter36 (val 39.01, deep finetune of iter6)
COMMITS = ["37a85cf", "f088509", "ffcecba", "f4f626e", "b423825", "e920838", "61e383a", "d152472", "b0f4b78", "b0f438e", "02ddc2c"]

# Save under current HEAD's commit
out_commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
).stdout.strip()
out_dir = PRED_DIR / out_commit
out_dir.mkdir(parents=True, exist_ok=True)

for split in SPLITS:
    preds_lists = [torch.load(PRED_DIR / c / f"{split}.pt", weights_only=True) for c in COMMITS]
    n = len(preds_lists[0])
    avg = []
    for i in range(n):
        stacked = torch.stack([pl[i] for pl in preds_lists], dim=0)
        avg.append(stacked.mean(dim=0))
    torch.save(avg, out_dir / f"{split}.pt")
    print(f"{split}: averaged {len(COMMITS)} commits → {out_dir}/{split}.pt ({n} samples)")

print(f"\nEnsemble saved to {out_dir}")
