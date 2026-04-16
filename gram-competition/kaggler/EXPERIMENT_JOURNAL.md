# Experiment Journal

Record every meaningful experiment here. Read this before starting a new
iteration so you don't repeat work or make the same mistake twice.

## Format

One entry per experiment, newest at the top:

```
### YYYY-MM-DD — short-title
- **Hypothesis:** what you expected to improve and why
- **Change:** what you actually modified (file + 1-line summary)
- **Result:** val/l2_error (best), train loss, epoch count, VRAM peak
- **Verdict:** kept / discarded — one sentence on why
- **Notes:** surprises, failure modes, ideas to try next
```

Keep entries short. Link W&B run URLs when useful.

---

## Entries

### 2026-04-16 — v3: multi-scale voxel stats (mean/std/dev) + laplacian + bf16 AMP
- **Hypothesis:** v2's voxel *mean* alone is weak — turbulence needs local
  variance and self-deviation (a gradient-like proxy), plus a cheap
  laplacian (coarse mean − fine mean) to signal spatial curvature.
  Richer spatial stats should finally break the 1.3 plateau.
- **Change:** `model.py` → `voxel_stats` returning (mean, std, dev) per scale,
  plus `laplacian = mean_coarse − mean_fine`. Input dim 41 → 53.
  `train.py` adds `--amp` bf16 autocast (~1.8× speedup).
  Arch: hidden=512, 10 ResBlocks. 67s/epoch. Peak 8.1GB (AMP).
- **Result:** Best val/l2=**1.2615** at epoch 49 of 49 (55 min timeout).
  Train loss 0.044 → 0.019. Val trajectory bounces but monotone-best:
  1.70, 1.66, 1.51, 1.43, 1.48, ..., 1.30 (ep20), 1.29 (ep27),
  1.28 (ep35), 1.27 (ep39), 1.27 (ep42), 1.26 (ep48, ep49).
  WandB run `edward/v3-voxelstats+amp` (uqgrg7g7). MAE
  Ux=0.86, Uy=0.36, Uz=0.59.
- **Verdict:** Kept — first genuine break through the 1.3 plateau
  (v1=1.34, v2=1.39). ~9% gain vs v2, ~6% vs v1. Still far from
  winner territory (~0.85).
- **Notes:** Loss still decreasing at timeout; longer training
  would likely push further. Next: either (a) run v3 longer with
  deeper model / more scales, or (b) move to inducing-point
  attention so each point can attend to a learned set of cluster
  tokens — gets us true long-range spatial interaction the voxel
  approach can't provide.

### 2026-04-16 — v2: voxel-pooled neighbors + dist-to-airfoil + Δv
- **Hypothesis:** pointwise MLP cannot model turbulence — it has no
  spatial context. Adding (a) multi-scale voxel-pooled neighbor velocity
  (0.05, 0.20 m scales), (b) log-distance to the nearest airfoil point,
  and (c) temporal Δv features should substantially lower val/l2.
- **Change:** `model.py` — `voxel_pool_mean`, `min_distance_to`; input
  dim grows from 19 → ~41. Architecture: hidden=512, 10 ResBlocks.
  121s/epoch. Peak VRAM 13.3 GB.
- **Result:** Best val/l2=1.39 at epoch 16 (of 16 run, killed early).
  Train loss 0.044 → 0.027. Val trajectory: 1.57,1.81,1.60,1.52,1.61,
  1.48,1.59,1.56,1.51,1.58,1.53,1.42,1.44,1.41,1.64,1.39.
  WandB run `edward/v2-voxel+dist`.
- **Verdict:** Only marginal gain over v1 (~1.34). Mean-only voxel
  features give weak spatial signal.
- **Notes:** Killed early (epoch 16/60) because trajectory mirrored
  v1's and the expected ~1.30 wouldn't justify full run; swap to v3
  with richer spatial stats (mean, std, self-deviation, laplacian
  proxy) and bf16 AMP for speed.

### 2026-04-16 — v1: residual + normalization + no-slip BC + time FiLM
- **Hypothesis:** absolute-velocity MLP wastes capacity modelling the ~35 m/s
  freestream. Residuals `v_out − v_in[-1]` have mean magnitude 2.46 m/s,
  ~14× smaller. Add: normalization, residual head, hard zero at airfoil
  indices, global time embedding broadcast to points.
- **Change:** `train.py` → `ResidualMLP` (hidden=512, 8 blocks).
  Loss in normalized space.
- **Result:** epoch 20, val/l2=1.3388 (wandb run `fkf8bty4`).
  Train loss plateaued around 0.026 (normalized MSE).
- **Verdict:** kept checkpoint but only baseline-level performance — the
  pointwise MLP has no way to learn spatial interactions (turbulence).
- **Notes:** `predict.py` was broken because importing `train` triggered its
  CLI parse. Extracted model to `model.py`. Val is noisy (bounces 1.34–1.65).
  Winners on mar29 hit 0.85 — need spatial features.
