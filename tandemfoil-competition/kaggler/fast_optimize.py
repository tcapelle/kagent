"""Fast vectorized ensemble weight optimizer using cached val predictions.

Pre-stacks per-sample predictions into tensors, then evaluates the ensemble
metric on GPU.
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per (split, sample) flatten across surface points only into a 1D vector
# Stack model errors into a [K, total_surf_pts] matrix per split.
print("Building stacked surf-pressure error matrix...")
all_errs = {}  # split -> tensor [K, T_surf]
for sn in VAL_SPLIT_NAMES:
    print(f"  {sn}")
    # Load each model's prediction-on-surface-points, vectorized
    per_model_concat = []
    y_concat = []
    for k, s in enumerate(SOURCES):
        d = torch.load(CACHE / f"{s}.pt", weights_only=False)
        chunks = []
        ychunks = []
        for i in range(len(d[sn]["pred"])):
            surf = d[sn]["surf"][i]
            chunks.append(d[sn]["pred"][i][surf, 2])  # only pressure on surface
            if k == 0:
                ychunks.append(d[sn]["y"][i][surf, 2])
        per_model_concat.append(torch.cat(chunks))
        if k == 0:
            y_concat = torch.cat(ychunks)
    # Stack: [K, T_surf]
    pred_stack = torch.stack(per_model_concat).to(device)  # [K, T]
    y_t = y_concat.to(device)  # [T]
    all_errs[sn] = (pred_stack, y_t)
    print(f"    shape: pred={pred_stack.shape}, y={y_t.shape}")


def metric(weights):
    """Compute val avg surface p MAE with given normalized weights."""
    w = torch.tensor(weights, device=device, dtype=torch.float32)
    w = w / w.sum().clamp(min=1e-9)
    total = 0.0
    for sn in VAL_SPLIT_NAMES:
        pred_stack, y_t = all_errs[sn]
        blend = (pred_stack.t() @ w)  # [T] = sum_k w_k * pred_k
        err = (blend - y_t).abs().mean()
        total += err.item()
    return total / len(VAL_SPLIT_NAMES)


# Per-model singles
print("\nSingle-model val avg_surf_p:")
single = []
for k, s in enumerate(SOURCES):
    w = [0.0] * len(SOURCES); w[k] = 1.0
    single.append(metric(w))
    print(f"  {s}: {single[-1]:.4f}")

# Inverse-weighted full
inv_w = [1.0 / s for s in single]
print(f"\nAll 11 inv-weighted: {metric(inv_w):.4f}")

# Sort by score
order = sorted(range(len(SOURCES)), key=lambda k: single[k])
print(f"Order (best first): {[SOURCES[k] for k in order]}")

# Try top-K with inv weights
results = []
for k in range(2, len(SOURCES) + 1):
    keep_idx = set(order[:k])
    w = [1.0 / single[i] if i in keep_idx else 0.0 for i in range(len(SOURCES))]
    score = metric(w)
    results.append((k, score, w))
    print(f"top-{k}: {score:.4f}")

best_k, best_score, best_w = min(results, key=lambda r: r[1])
print(f"\nBest top-K: top-{best_k} with {best_score:.4f}")

# Greedy refinement
print("\nGreedy refinement...")
for round_i in range(3):
    improved = False
    for k in range(len(SOURCES)):
        for delta in [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0]:
            trial = best_w[:]
            if best_w[k] == 0:
                if delta < 0: continue
                trial[k] = delta * (1.0 / single[k])
            else:
                trial[k] = max(0.0, best_w[k] * (1 + delta))
            score = metric(trial)
            if score < best_score - 1e-5:
                best_score = score; best_w = trial; improved = True
                print(f"  k={SOURCES[k]} d={delta}: {score:.4f}")
    if not improved:
        print(f"  Round {round_i}: no improvement")
        break

total = sum(best_w)
norm_w = [w / total for w in best_w]
print(f"\nFinal val score: {best_score:.4f}")
for s, w in zip(SOURCES, norm_w):
    if w > 1e-4:
        print(f"  {s}: {w:.4f}")

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
        acc = sum(w * parts[k][i] for k, w in enumerate(norm_w) if w > 1e-6)
        blended.append(acc)
    torch.save(blended, out_dir / f"{split}.pt")
print(f"\nDone. Best val avg_surf_p = {best_score:.4f}")
print(f"Output: {out_dir}")
