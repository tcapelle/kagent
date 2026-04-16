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

### 2026-04-16 — v2: voxel-pooled neighbors + dist-to-airfoil + Δv
- **Hypothesis:** pointwise MLP cannot model turbulence — it has no
  spatial context. Adding (a) multi-scale voxel-pooled neighbor velocity
  (0.05, 0.20 m scales), (b) log-distance to the nearest airfoil point,
  and (c) temporal Δv features should substantially lower val/l2.
- **Change:** `model.py` — `voxel_pool_mean`, `min_distance_to`; input
  dim grows from 19 → ~41. Architecture: hidden=512, 10 ResBlocks.
- **Result:** pending
- **Verdict:** pending
- **Notes:** Debug run: peak VRAM 13.3 GB, ~1 sec/step on 4 samples.
  Voxel grids use `torch_geometric.utils.scatter` (no torch-cluster).

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
