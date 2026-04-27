"""Quick streaming ensemble optimizer using cached val predictions.

Computes per-model val avg_surf_p, then evaluates uniform-on-top-k and
inverse-weighted ensembles, picks the best, submits to HEAD commit.
"""

import os
import subprocess
from pathlib import Path

import torch
from tqdm import tqdm

from data import VAL_SPLIT_NAMES

CACHE = Path("/tmp/edward_val_preds")
RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PRED_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}/edward")
TEST_SPLITS = [
    "test_single_in_dist", "test_geom_camber_rc",
    "test_geom_camber_cruise", "test_re_rand",
]

SOURCES = ["79894e7", "0fa22ab", "5d05ebb", "b04a915", "78e27f0",
           "b7080b6", "29827b6", "ce6d81a", "347116e", "39cb43e", "e8d2478"]


# Compute per-(model, sample) surf-pressure error tensor: errs[k][split][i] = |pred-y| on surface
print("Loading caches...")
preds = {}
ys = {}
surfs = {}
for src in SOURCES:
    print(f"  {src}...")
    d = torch.load(CACHE / f"{src}.pt", weights_only=False)
    preds[src] = {sn: [p[:, 2] for p in d[sn]["pred"]] for sn in VAL_SPLIT_NAMES}
    if not ys:
        ys = {sn: [t[:, 2] for t in d[sn]["y"]] for sn in VAL_SPLIT_NAMES}
        surfs = {sn: d[sn]["surf"] for sn in VAL_SPLIT_NAMES}
    del d


def metric(weights):
    total_p_mae = 0.0
    for sn in VAL_SPLIT_NAMES:
        ssum = 0.0
        n = 0
        for i in range(len(ys[sn])):
            blend = sum(w * preds[s][sn][i] for w, s in zip(weights, SOURCES) if w > 0)
            err = (blend - ys[sn][i]).abs()
            mask = surfs[sn][i]
            ssum += err[mask].sum().item()
            n += mask.sum().item()
        total_p_mae += ssum / max(n, 1)
    return total_p_mae / len(VAL_SPLIT_NAMES)


# Single-model val scores
print("\nSingle-model surface p MAE (val):")
single = {}
for s in SOURCES:
    w = [0.0] * len(SOURCES)
    w[SOURCES.index(s)] = 1.0
    score = metric(w)
    single[s] = score
    print(f"  {s}: {score:.4f}")

# Inverse-weighted full
inv_w = [1.0 / single[s] for s in SOURCES]
print(f"\nInverse-weighted (all 11): {metric(inv_w):.4f}")

# Drop worst, keep best
sorted_srcs = sorted(SOURCES, key=lambda s: single[s])
print(f"Sorted: {[(s, f'{single[s]:.2f}') for s in sorted_srcs]}")

# Try keeping top-K
results = []
for k in range(2, len(SOURCES) + 1):
    kept = set(sorted_srcs[:k])
    w = [1.0 / single[s] if s in kept else 0.0 for s in SOURCES]
    score = metric(w)
    results.append((k, score, w))
    print(f"top-{k}: {score:.4f}")

# Pick best
best_k, best_score, best_w = min(results, key=lambda r: r[1])
print(f"\nBest: top-{best_k} with avg_surf_p_val={best_score:.4f}")

# Greedy weight refinement
print("\nGreedy refinement...")
for round_i in range(2):
    improved = False
    for k in range(len(SOURCES)):
        if best_w[k] == 0.0 and round_i > 0:
            # also try adding back zero models
            for boost in [0.001, 0.01, 0.05, 0.1]:
                trial = best_w[:]
                trial[k] = boost
                score = metric(trial)
                if score < best_score:
                    best_score = score; best_w = trial; improved = True
                    print(f"  add {SOURCES[k]} @ {boost}: {score:.4f}")
            continue
        for delta in [-0.5, -0.25, 0.25, 0.5, 1.0]:
            trial = best_w[:]
            trial[k] = max(0.0, trial[k] * (1 + delta))
            score = metric(trial)
            if score < best_score:
                best_score = score; best_w = trial; improved = True
                print(f"  k={SOURCES[k]} delta={delta}: {score:.4f}")
    if not improved:
        break

total = sum(best_w)
norm_w = [w / total for w in best_w]
print(f"\nFinal: {best_score:.4f}")
for s, w in zip(SOURCES, norm_w):
    print(f"  {s}: {w:.3f}")

# Submit to HEAD
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()
out_dir = PRED_DIR / commit
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\nSubmitting to {out_dir}...")
for split in TEST_SPLITS:
    parts = [torch.load(PRED_DIR / s / f"{split}.pt", weights_only=True) for s in SOURCES]
    n = len(parts[0])
    blended = []
    for i in tqdm(range(n), desc=split, leave=False):
        acc = sum(w * parts[k][i] for k, w in enumerate(norm_w) if w > 0)
        blended.append(acc)
    torch.save(blended, out_dir / f"{split}.pt")
print(f"Done. Best val avg_surf_p = {best_score:.4f}")
