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

### 2026-04-17 — v7: two-scale voxel-mix (0.08 + 0.25)
- **Hypothesis:** v6 used a single mix scale 0.12. Adding a coarser
  second scale (0.25) should capture longer-range structure while
  keeping fine detail at 0.08 — covering wake + near-airfoil regions.
- **Change:** `model.py` → `VOXEL_MIX_SCALES = (0.08, 0.25)`. Same
  `MIX_EVERY=2`, 10 ResBlocks, hidden=512. 90 min budget.
- **Result:** Best val/l2=**1.1824** at epoch 36 of 36 (90 min timeout).
  Train 0.044 → 0.0155. Per-epoch 136s (vs v6 ~105s average). v7 was
  ahead of v6 at matched epoch count (ep29: v7=1.22 vs v6=1.26) but
  the extra mix scale slowed epochs so v6 got 52 vs v7's 36.
  WandB `edward/v7-multiscale-mix`.
- **Verdict:** Discarded — worse than v6 (1.1681). Architecture is
  per-epoch better but slower, and cosine schedule (epochs=100) didn't
  fully anneal at timeout.
- **Notes:** If retrying, either set `epochs` closer to the expected
  budget so LR actually decays, or drop the coarser scale. Better
  next step: real learned aggregation on voxel tokens, not gated
  mean-pooling.

### 2026-04-16 — v6: iterative voxel-mix + 90 min budget
- **Hypothesis:** v4's per-point MLP never mixes latent representations
  across space. Adding lightweight scatter-mean message passing between
  ResBlocks (VoxelMix) gives iterative spatial communication. v5.1
  showed this matches v4 at equal epochs; extra training time should
  push it well past v4.
- **Change:** Same model as v5.1 — `VoxelMix` with per-scale LN +
  tanh-bounded per-channel gate (zero-init), scatter-mean at scale 0.12
  applied every 2 blocks. Training: `torch.nn.utils.clip_grad_norm_`
  (max_norm=1.0) added to stabilize early iterations. 10 ResBlocks, 5
  VoxelMix, hidden=512. `MAX_TIMEOUT_MIN=90`.
- **Result:** Best val/l2=**1.1681** at epoch 52 of 52 (91 min).
  Train loss 0.044 → 0.014, still descending. Val trajectory: 1.78
  (ep1), 1.46 (ep5), 1.41 (ep10), 1.33 (ep20), 1.26 (ep28), 1.24
  (ep35), 1.20 (ep40), 1.17 (ep48, ep52). WandB `edward/v6-mix-90min`.
- **Verdict:** Kept — ~6% better than v4 (1.2409 → 1.1681). Iterative
  message passing is clearly the right direction. Val still descending
  at timeout.
- **Notes:** Per-epoch time was variable (97-190s) — likely background
  GPU use. Gap to winner (~0.85) is now ~37%. Next: pivot to
  voxel-token transformer (voxel pool at scale 0.10 → ~1-2k tokens,
  self-attention among them, scatter back to points). Much stronger
  spatial model than gated scatter-mean, still tractable.

### 2026-04-16 — v5.1: VoxelMix (LN + tanh gate) + grad-clip (55 min)
- **Hypothesis:** v5 diverged at ep19 because unconstrained gate
  parameter allowed positive feedback. Bound the gate magnitude and
  add grad clipping.
- **Change:** `model.py::VoxelMix` — added per-scale LN on pooled
  features, tanh on gate, zero-init kept. `train.py` — add
  `clip_grad_norm_(..., 1.0)`.
- **Result:** Best val/l2=1.2545 at epoch 34 of 34 (55 min budget
  gave fewer epochs due to mix compute overhead). No divergence.
  At ep34, v4 was 1.2806 — so v5.1 beats v4 at equal epoch count.
- **Verdict:** Kept as stepping stone — architecture validated,
  needed longer training.

### 2026-04-16 — v4: 4 voxel scales (0.03/0.08/0.20/0.50) + within-voxel offsets
- **Hypothesis:** v3's two scales under-resolve boundary-layer detail
  (0.05) and large-structure context (need > 0.20). Adding finer 0.03
  and coarser 0.50 scales plus an "offset within voxel" feature
  (lets two points in the same voxel differentiate sub-voxel position)
  should improve further.
- **Change:** `model.py` → `VOXEL_SCALES = (0.03, 0.08, 0.20, 0.50)`,
  `voxel_stats` now returns sub-voxel offset per scale, input dim
  53 → 83. Default blocks 10 → 12. 77s/epoch (vs v3's 67s).
  Peak VRAM 9.6GB.
- **Result:** Best val/l2=**1.2409** at epoch 36 of 43 (55 min).
  Train loss 0.044 → 0.018. Still improving at timeout.
  WandB `edward/v4-4scales+offsets`.
- **Verdict:** Kept — modest 1.6% gain over v3 (1.2615 → 1.2409),
  diminishing returns for more hand-crafted features alone.
- **Notes:** Gap to winner (~0.85) is still huge. Pure pointwise MLP
  + voxel features has a ceiling because voxels don't mix across the
  network — each point only sees static pooled neighbors *once*,
  at the input. Next: iterative voxel-scatter/gather interleaved
  with ResBlocks (graph-conv style message passing).

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
