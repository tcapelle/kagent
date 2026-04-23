"""Average predictions from multiple commits into a new submission.

Runs AFTER train.py+predict.py. Picks up two existing predictions on the PVC,
averages them per-sample per-channel, and writes a new submission keyed on
the current git commit.

Usage: git commit the ensemble config first, then `python ensemble.py`.
"""

import argparse
import os
import subprocess
from pathlib import Path

import torch


RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PRED_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}/frieren")

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True, help="commit hashes whose predictions to average")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="optional per-source weights (default equal)")
    args = p.parse_args()

    w = args.weights or [1.0] * len(args.sources)
    assert len(w) == len(args.sources)
    tot = sum(w)
    w = [x / tot for x in w]

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "unknown"
    out = PRED_DIR / commit
    out.mkdir(parents=True, exist_ok=True)
    print(f"Ensembling {len(args.sources)} sources with weights {w}")
    print(f"Output: {out}")

    for split in TEST_SPLITS:
        combined = None
        for src, weight in zip(args.sources, w):
            src_path = PRED_DIR / src / f"{split}.pt"
            preds = torch.load(src_path, weights_only=False)
            if combined is None:
                combined = [weight * p for p in preds]
            else:
                for i, p in enumerate(preds):
                    combined[i] = combined[i] + weight * p
        torch.save(combined, out / f"{split}.pt")
        print(f"  {split}: {len(combined)} tensors averaged")


if __name__ == "__main__":
    main()
