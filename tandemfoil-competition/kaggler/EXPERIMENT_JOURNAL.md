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

### 2026-04-27 — iter3: surface MSE → L1 (FAILED)

- **Hypothesis:** the metric is MAE in physical units; switching surface loss
  from MSE to L1 (in normalized space) should align the optimization with the
  metric.
- **Change:** `train.py` — surface loss uses `abs_err = err.abs()` instead of
  `err**2`; volume stays MSE.
- **Result:** 40 epochs, best val avg_surf_p = 92.71 at epoch 39
  (single=65.51, geom_rc=145.52, geom_cruise=65.13, re_rand=94.68). Run
  `f6...`.
- **Verdict:** discarded — worse than iter2 (87.59). Code reset.
- **Notes:** L1 won on `single_in_dist` (65 vs 72) but lost on
  `geom_camber_rc` (146 vs 123) — so it overfit the in-distribution ranges
  and hurt the OOD-camber generalization. Average dragged down by geom_rc.
  Lesson: L1 in normalized space ≠ MAE in physical units (off by a per-channel
  y_std factor) and seems to hurt generalization. If trying L1 again, scale by
  `y_std` to actually match the physical metric and possibly clip
  high-pressure outliers.

### 2026-04-27 — iter2: surf_p_weight 2→4

- **Hypothesis:** scored metric is purely surface pressure MAE, so doubling the
  surface-pressure channel weight in the loss (from 2× to 4×) should pull more
  optimization budget toward the only thing that's measured. Architecture and
  every other hyper-param identical to thorfinn-apr23-iter4 recipe (Transolver
  256x6, slice_num=96, mlp_ratio=4, bf16, sub-20k point sampling, surf_weight=20).
- **Change:** `train.py:43` — `surf_p_weight: 2.0 → 4.0`.
- **Result:** 39 epochs in 30 min; best val avg_surf_p=87.59 at epoch 37
  (single=72.04, geom_rc=123.15, geom_cruise=65.21, re_rand=89.93). Model 4.59M
  params, 9.3GB peak. Run `bjq3mkuc`. Commit `dbdce82`.
- **Verdict:** kept — first complete submission. Note: previous in-flight
  iter1 process from a stale session was sharing the GPU for the first ~9 min
  (slowed epoch 1 to 95s); after killing it, epochs settled at 45s.
- **Notes:** Curve still descending at epoch 37 (87.59 vs 88.4 at 39), so the
  recipe likely hasn't converged inside 30 min. geom_camber_rc is the worst
  split (123) — unseen front-foil camber for raceCar tandem; that's where
  generalization hurts most. Compare to apr27 leaderboard: top frieren=42,
  top thorfinn=42.9, askeladd=79; current 87 is in the same ballpark as the
  past askeladd weak baseline. Big gap to the leaders. Need either (a) longer
  training, (b) bigger/better architecture, or (c) loss aligned with physical
  MAE (not normalized MSE). Likely next: combine MSE with direct surface-MAE
  in physical units, and/or push surf_p_weight higher (8?) to see if the
  geom_rc gap closes.
