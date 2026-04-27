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

### 2026-04-27 — iter7: revert to 256x6, surf_p_weight 4→12, resume from iter4

- **Hypothesis:** the metric is purely surface pressure MAE. Pushing the
  per-channel weight on surface pressure from 4× to 12× should make the model
  specialize harder on what's actually scored. Revert iter5's deeper
  architecture (it was a dead end) and resume from iter4's best ckpt to keep
  the warm start.
- **Change:** `train.py` — `n_layers 8→6`, `weight_decay 5e-5→1e-5`,
  `surf_p_weight 4.0→12.0`. Run with `--resume_from .../model-mhk382oc/`.
- **Result:** 40 epochs, best val avg_surf_p = **72.70** at epoch 39 — still
  descending at the timeout. Run `e5uc7jqw`, commit `40a031f`. Splits:
  single=58.57, geom_rc=104.21, geom_cruise=53.80, re_rand=74.23. All splits
  beat iter4 (which was single=61.36, geom_rc=111.64, geom_cruise=54.85,
  re_rand=80.09).
- **Verdict:** kept — new best. ~6% gain over iter4 across the board.
- **Notes:** Loss was still descending — iter8 should resume iter7 with a
  lower LR (current cosine schedule pushed to ~1.75e-5 at end; restarting at
  1.5e-4 is too aggressive for fine-tuning further). Plan iter8: resume from
  iter7 ckpt with `--lr 5e-5` to squeeze more out of this recipe. Leaderboard
  competitors are at 40–43 test (vs my projected ~63–65 from val=72.70 with
  ~88% val/test factor) — still a gap, but closing.

### 2026-04-27 — iter6: resume 256x8 from iter5, wd=5e-5 (FAILED)

- **Hypothesis:** iter5 (256x8 from scratch) reached 112.86 in only 28 epochs —
  the deeper model was just under-trained. Resuming from iter5's best ckpt for
  another 30 min should let the deeper model converge to a lower minimum than
  256x6 (which plateaued at 77 in iter4).
- **Change:** `train.py` — `weight_decay 1e-5→5e-5` (slight regularization
  for the deeper model). Run with `--resume_from .../model-xurnwxz0/`.
- **Result:** 31 epochs, best val avg_surf_p = **103.26** at epoch 28.
  Plateaued at 103–108 across last 10 epochs. Run `cnf9no6m`, commit
  `a2db9a3`. Mirrored predictions overwrote iter5's at the same commit dir.
- **Verdict:** discarded — much worse than iter4's 76.99. The deeper model
  is simply a worse fit for this dataset/timing budget; even 60 min total
  training (iter5 + iter6) couldn't beat 60 min of 256x6 (iter2 + iter4).
- **Notes:** Reverted n_layers and weight_decay in iter7. Lesson: 256x6 with
  slice_num=96 is the right capacity for ~30-min training on ~1500 samples
  per the "balanced sampler" weighting; 256x8 over-parameterizes and
  diverts gradient signal. Don't try this size again unless we 2× the time
  budget (which the rules don't allow).

### 2026-04-27 — iter5: n_layers 6→8 fresh (UNDER-TRAINED, ckpt kept)

- **Hypothesis:** iter4 plateaued at val=77 with 256x6, suggesting capacity
  limit. A deeper model (256x8, +33% params) should have a higher ceiling
  given enough training.
- **Change:** `train.py:86` — `n_layers: 6 → 8`. Fresh from random init.
- **Result:** 28 epochs, best val avg_surf_p = **112.86** at epoch 28. Slower
  per-epoch (~57s vs 45s) so fewer total epochs in 30 min. Run `xurnwxz0`,
  commit `87f3f68`. Auto-submitted predictions to leaderboard for that commit.
- **Verdict:** kept the commit (vs reset) so the iter5 ckpt remains usable
  as an init for iter6 resume. As a standalone score it's clearly worse than
  iter4 (77) — the deeper model just didn't converge in 30 min from scratch.
- **Notes:** iter6 = resume from iter5 ckpt, giving the deeper model another
  30 min of training to test whether the extra capacity pays off when
  properly trained.

### 2026-04-27 — iter4: resume from iter2 best, lr=1.5e-4, warmup=100

- **Hypothesis:** iter2 stopped at epoch 37/39 still descending; the 30-min
  cosine schedule never bottomed out. Continuing from the iter2 best with a
  fresh, shorter cosine cycle (peak lr=1.5e-4) effectively gives 60 min of
  training and should let the loss settle further.
- **Change:** `train.py` — when `--resume_from` is set, default `lr` becomes
  1.5e-4 and `warmup_steps` becomes 100. Run with
  `--resume_from .../model-bjq3mkuc/checkpoint.pt`.
- **Result:** 40 epochs, best val avg_surf_p = **76.99** at epoch 31
  (single=61.36, geom_rc=111.64, geom_cruise=54.85, re_rand=80.09). Run
  `mhk382oc`. Commit `c8f502d`. Test scoring: 67.71 (rank 3, behind
  edward=43.73 and alphonse=50.83).
- **Verdict:** kept — improved every split vs iter2 (-12% avg).
- **Notes:** Plateaued by epoch 28+; oscillating 77–80 for the last 10 epochs.
  Looks like recipe capacity is the limiter — more training won't help, the
  model can't go lower with this architecture. Big test/val gap (val=77,
  test=68) suggests the val set is a bit harder. Need a bigger model or a
  better architecture to close the 24-point gap to edward. Next: try
  capacity bump (n_layers 6→8 or n_hidden 256→320).

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
