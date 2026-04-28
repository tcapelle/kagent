"""Average per-sample predictions from multiple commits and submit ensemble."""

import os
import subprocess
from pathlib import Path
import torch

PRED_DIR = Path(f"/mnt/new-pvc/predictions/{os.environ['RESEARCH_TAG']}/frieren")
SPLITS = ["test_single_in_dist", "test_geom_camber_rc", "test_geom_camber_cruise", "test_re_rand"]

# Strong-only 7-way: drop iter20 (val 43.34) and iter26 (val 44.86)
# kept: iter4 (40.97), iter6 (39.54), iter9 (42.10), iter11 (41.40), iter14 (39.42),
#       iter17 (37.99), iter23 (39.93)
COMMITS = ["37a85cf", "f088509", "ffcecba", "f4f626e", "b423825", "e920838", "61e383a"]

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
