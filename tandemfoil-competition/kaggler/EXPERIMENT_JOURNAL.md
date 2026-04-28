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

### 2026-04-28 — iter24: ensemble iter23*2+iter21*2+iter19+iter18*2 (38.72)

- **Hypothesis:** add iter23 to the ensemble for diversity. iter23 alone is
  worst at 40.33, but it has a different basin from iter21 (sub80k 2nd cycle).
- **Change:** No new training. `--checkpoints iter23,iter21,iter19,iter18
  --weights 2,2,1,2`.
- **Result (bs=1 val):** **38.72** | single=33.79, geom_rc=57.08,
  geom_cruise=24.00, re_rand=40.00. ~0.6% gain over iter22.
- **Verdict:** kept — last gasp before diminishing returns. Several
  ensemble combos ranged 38.72–38.83; the recipe is hitting a floor.
- **Notes:** Future iterations need a *fundamentally different* model for
  ensemble — same architecture / same recipe / different epochs all live
  in similar basins. To break further, try: (a) different surf_p_weight
  (e.g., 24), (b) different subsample size (e.g., back to 40k for
  diversity), (c) bigger model retrained from scratch.

### 2026-04-28 — iter23: resume iter21 sub80k (val_bs1=40.33 alone)

- **Hypothesis:** continue iter21's sub80k chain for another diverse ckpt.
- **Change:** No code change. `--resume_from .../model-lrecks81/`.
- **Result:** 15 epochs, best val_bs4 = 65.42 at epoch 10. bs=1 val:
  **40.33** — *worse than iter21 alone (39.73)*. Run `x2fzf3d5`.
  Predictions auto-submitted to commit `457dafb`, overwriting iter22.
- **Verdict:** kept ckpt for ensembling; solo it's a regression.
- **Notes:** The ensembles still benefit (38.72 with iter23 vs 38.94
  without), but margins are now sub-1%. Need to break out of this
  recipe to make progress.

### 2026-04-28 — iter22: ensemble iter21*2+iter19+iter18*2 (val_bs1=38.94)

- **Hypothesis:** iter21 alone (39.73) is now the best single model.
  Combining iter21 with iter19 and iter18 (heavily weighted on the two
  strongest, iter21 and iter18) should average their different failure
  modes.
- **Change:** No new training. Run `predict.py --checkpoints
  iter21,iter19,iter18 --weights 2,1,2`.
- **Result (bs=1 val):** **38.94** | single=34.42, geom_rc=57.20,
  geom_cruise=24.29, re_rand=39.86.
- **Verdict:** kept — first time below 39. Expected test ≈ 33.6.
- **Notes:** iter21+iter18 [1,1] alone (38.98) was almost as good as the
  three-way [2,1,2] (38.94) — the iter21+iter18 pair is doing most of the
  work. iter19 just adds a tiny smoothing benefit.

### 2026-04-28 — iter21: resume iter19 sub80k (val_bs1=39.73, new best single)

- **Hypothesis:** iter19 was capped at 10 epochs (sub80k slow ~120s/epoch).
  Continuing from there gives the deeper-sampling regime more training
  budget, expecting to surpass iter18 (val_bs1=40.08) once converged.
- **Change:** No code change. `--resume_from .../model-n7b7iwno/`.
- **Result:** 15 epochs, best val_bs4 = **65.60** at epoch 2 (very early
  best!). Real bs=1 val: **39.73** — beats iter18's 40.08 alone. Run
  `lrecks81`. Predictions auto-submitted to commit `1efd2fa` (overwriting
  iter20 ensemble there). Splits: single=36.65, geom_rc=57.72,
  geom_cruise=24.48, re_rand=40.06.
- **Verdict:** kept — new best single model. Auto-submission at 1efd2fa
  is iter21 alone (val=39.73), which already beats my previous best
  (5198950 at val=40.08). The lost iter20 ensemble was 39.49 — slightly
  better than iter21 alone but iter22 ensemble is even better.
- **Notes:** sub80k now produces models that solo-beat the sub60k iter18.
  The denser gradient lever pays off when you have enough epochs to
  exploit it. Memory at 34.7GB peak with sub80k — fine.

### 2026-04-28 — iter20: ensemble iter19+iter18*2 (val_bs1=39.49)

- **Hypothesis:** iter19's single-model is worse than iter18 (41.25 vs
  40.08) but it has DIFFERENT failure patterns — better on geom_cruise
  (24.72 vs 25.54) and re_rand (40.25 vs 41.56), worse on single. Ensemble
  averages out the differences.
- **Change:** No new training. Run `predict.py --checkpoints iter19,iter18
  --weights 1,2`.
- **Result (bs=1 val):** **39.49** — single=34.68, geom_rc=57.87,
  geom_cruise=24.85, re_rand=40.57. First time below 40.
- **Verdict:** kept — submit. Expected test ≈ 34.0.
- **Notes:** Pattern: when a single model regresses but has different
  errors than the previous best, weighted ensemble (with the better one
  getting more weight) wins. iter19+iter18*2 [1,2] beats both iter19+iter18
  [1,1] (39.59) and iter19*2+iter18 [2,1] (39.93).

### 2026-04-28 — iter19: sub80k resume from iter18 (kept for ensemble only)

- **Hypothesis:** iter18's sub60k breakthrough suggested denser gradient
  helps. Push to sub80k for even denser per-step signal.
- **Change:** `train.py:45` — `train_subsample 60000 → 80000`. Run with
  `--resume_from .../model-rvlag6rf/`.
- **Result:** ~10 epochs (sub80k slows training to ~120s/epoch — the run
  ran 32 min total, just over the timeout). Best val_bs4 = 66.12 at
  epoch 10. Real bs=1 val: **41.25** — *worse* than iter18 alone (40.08).
  Run `n7b7iwno`. Splits: single=40.52, geom_rc=59.53, geom_cruise=24.72,
  re_rand=40.25.
- **Verdict:** kept the ckpt for ensembling — combined with iter18 it
  beats iter18 alone (39.49 ensemble vs 40.08 solo). Solo it's a
  regression.
- **Notes:** sub80k at 10 epochs probably wasn't enough to fine-tune a
  256x6 model in 30 min. The denser gradient is genuinely helping per
  step but we only got 10 of them. iter19's lower geom_cruise and
  re_rand (vs iter18) hint that the new basin would be better with more
  time, but ensemble extraction is faster ROI.

### 2026-04-27 — iter18: train_subsample 40k→60k, resume iter16 (BREAKTHROUGH 40.08)

- **Hypothesis:** the resume chain has been giving diminishing returns
  (~1% per cycle); the only "fresh" lever I haven't pushed is the
  subsample size. Going 40k→60k gives ~50% more points per gradient
  step, which should let the model sharpen its surface predictions.
  iter10 already showed sub20k→sub40k helped a lot; sub40k→sub60k is the
  next data point.
- **Change:** `train.py:45` — `train_subsample 40000 → 60000`. Run with
  `--resume_from .../model-snvf1y5m/`.
- **Result:** ~16 epochs, best val_bs4 = 67.63 at epoch 14. Real bs=1
  val: **40.08** — down from iter17's 42.03 ensemble (-4.6%)! ALL splits
  improved meaningfully: single 36.60→34.84, geom_rc 59.82→58.39,
  geom_cruise 28.03→25.54, re_rand 43.68→41.56. Run `rvlag6rf`,
  predictions auto-submitted to commit `5198950`.
- **Verdict:** kept — biggest single-iter gain since iter10's bs=1 fix.
- **Notes:** Ensembling iter18 with iter15/iter16 actually *hurts*
  (40.08 → 40.46 in best ensemble) — iter18 has different failure
  patterns than the sub40k chain, so averaging blurs its signal. The
  denser gradient changed the loss-landscape navigation enough that the
  resulting basin is genuinely better, not just a small step. Next
  iter: sub80k or sub100k to test if the trend continues. Expected test
  ≈ 35 (using val/test ratio 0.87) — would jump to rank 1 if it lands
  there.

### 2026-04-27 — iter17: ensemble iter16+iter15 (val_bs1=42.03)

- **Hypothesis:** iter16 alone (42.78) is slightly worse than iter15 alone
  (42.43), but they were trained from different cosine resume cycles —
  diverse enough that averaging helps even when component is weaker.
- **Change:** No new training. `predict.py --checkpoints iter16,iter15
  --weights 1,1`.
- **Result (bs=1 val):** **42.03** — new best. single=36.60, geom_rc=59.82,
  geom_cruise=28.03, re_rand=43.68. Equal weights edged weighted variants.
- **Verdict:** kept — diversity-from-different-resume-cycles ensembling
  gives the best of both. ~1% gain on top of iter15 alone.
- **Notes:** Adding iter13 dilutes (42.03 → 42.26 with [1,1,1]). The
  resume chain has narrowed the diversity — only adjacent ckpts in the
  chain ensemble well together.

### 2026-04-27 — iter16: resume iter15 (val_bs1=42.78, slight regression alone)

- **Hypothesis:** continue resume chain with iter15.
- **Change:** No code change. `--resume_from .../model-i4xb7o2y/`.
- **Result:** 25 epochs, best val_bs4 = **66.09** at epoch 14 (lowest bs4
  so far). Real bs=1 val: **42.78** (vs iter15's 42.43 — *worse* alone).
  Splits: single=37.51, geom_rc=60.52, geom_cruise=28.28, re_rand=44.80.
  Auto-submit went to commit `e0934f4`. Run `snvf1y5m`.
- **Verdict:** kept the ckpt for ensembling — iter16+iter15 ensemble
  beats either alone (42.03). Solo it's a regression.
- **Notes:** This is the first resume that didn't strictly improve val_bs1
  alone. The cosine schedule may have descended past the bs1 sweet spot.
  Need a different recipe to break out — capacity, augmentation, or just
  better selection.

### 2026-04-27 — iter15: resume iter13 (NEW BEST 42.43)

- **Hypothesis:** the resume chain keeps yielding ~1–2% per cycle. iter13
  (val_bs1=43.77) was still improving at the last epoch — another 30 min
  should drop further.
- **Change:** No code change. Run with `--resume_from .../model-r9p6myvd/`.
- **Result:** 25 epochs, best val_bs4 = **67.35** at epoch 23 (vs iter13's
  66.54 — bs4 metric got *worse* at this best-saved-by-bs4 epoch). Real
  bs=1 val: **42.43** (vs iter13's 43.77, **-3.1%**). Run `i4xb7o2y`.
  Predictions auto-submitted to commit `da4f9f4` (overwriting iter14
  ensemble there — but iter15 alone beats the iter14 ensemble's 43.50, so
  this is a strict improvement). Splits: single=37.98, geom_rc=60.17,
  geom_cruise=28.32, re_rand=43.25.
- **Verdict:** kept — best single model so far. Even ensembling iter15 with
  iter13 only gives a tied or slightly worse result (iter15*2+iter13=42.46),
  so iter15 alone is the right submission.
- **Notes:** Disconnect between bs4 (best-saved metric) and bs1 (real
  metric) keeps growing — iter13's bs4=66.54 vs iter15's 67.35, but iter15
  is much better at bs1. Future iters: switch model selection to bs1 val
  if I have time, otherwise just resume more.

### 2026-04-27 — iter14: ensemble iter13*2 + iter11

- **Hypothesis:** iter13 (val_bs1=43.77) is the new best single model. Pairing
  it with iter11 (44.76) — which was trained from a different surf_p_weight
  basin (12 vs 16 implicit via iter10/iter11 chain) — should add diversity.
- **Change:** No new training. Run `predict.py --checkpoints iter13,iter11
  --weights 2,1`. New commit hash for fresh leaderboard entry.
- **Result (bs=1 val):** **43.50** | single=38.84, geom_rc=61.54,
  geom_cruise=28.91, re_rand=44.69. Best score yet.
- **Verdict:** kept — small but consistent gain.
- **Notes:** Tried many weight combos. Adding iter10 to the mix consistently
  hurts (43.50 → 43.65); iter10 is now too weak relative to iter13/iter11.
  Lesson: ensemble works when component models are similar strength;
  weakest model drags. Ranking with thorfinn=39.52 is essentially tied;
  this submission may break it.

### 2026-04-27 — iter13: resume iter11 with same recipe (more time helps)

- **Hypothesis:** iter11 still oscillating around 67–69 (bs=4) at the end of
  training; another 30-min resume cycle could shake it loose into a lower
  basin or at least average down the noise.
- **Change:** No code change (same recipe as iter11). Run with
  `--resume_from .../model-sumg2gb9/`.
- **Result:** 25 epochs, best val_bs4 = **66.54** at epoch 21 (vs iter11's
  67.72). Real bs=1 val: **43.77** (vs iter11's 44.76, -2.2%). Splits:
  single=39.61, geom_rc=62.02, geom_cruise=28.47, re_rand=44.99.
  Notably geom_cruise dropped 30.95→28.47 (-8%). Run `r9p6myvd`.
- **Verdict:** kept — resume continues to give marginal gains.
- **Notes:** geom_camber_cruise is now my best split — model is generalizing
  to unseen camber for cruise foils nicely. geom_camber_rc still highest
  (62.02), the consistent bottleneck.

### 2026-04-27 — iter12: ensemble iter11*2 + iter10 at predict time

- **Hypothesis:** iter10 (sp_w=12) and iter11 (sp_w=16) trained from
  different optima with different loss-balance pressures. Their predictions
  may be diverse enough that a weighted average reduces MAE.
- **Change:** No new training. Run `predict.py --checkpoints iter11,iter10`
  with weights `[2, 1]` (iter11 stronger since it's slightly better alone).
- **Result (bs=1 val):** 44.45 (vs iter11 alone 44.76, iter10 alone 45.60).
  Splits: single=39.39, geom_rc=62.74, geom_cruise=30.42, re_rand=45.23.
  Submit at the journal commit hash so the leaderboard sees it as a
  separate entry.
- **Verdict:** kept — small but real ~0.7% improvement on top of iter11,
  free at inference time.
- **Notes:** Weighted [2,1] beat equal [1,1] (44.45 vs 44.51) and [3,1]
  (44.47). Three-model ensembles (with iter4) hurt because iter4 is too
  much weaker. Diversity *between models of similar strength* helps; mixing
  in weaker models drags the average back up.

### 2026-04-27 — iter11: surf_p_weight 12→16, resume from iter10

- **Hypothesis:** surf_p_weight progression has been 2 → 4 (iter2) → 12
  (iter7) → 16 (iter11). Each bump pushed the metric harder. iter10 looks
  capacity-limited; bumping the pressure weight a bit more might squeeze
  the last bit out of the same architecture.
- **Change:** `train.py:43` — `surf_p_weight 12 → 16`. Run with
  `--resume_from .../model-j8o76pto/`.
- **Result:** 25 epochs, best val_bs4 = **67.72** at epoch 18 (vs iter10's
  68.21). Real bs=1 val: **44.76** (vs iter10's 45.60, -1.8%). Splits:
  single=39.78, geom_rc=62.61, geom_cruise=30.95, re_rand=45.71. Run
  `sumg2gb9`, commit `3c78568`. Predictions auto-submitted to commit
  `5360f22`.
- **Verdict:** kept — small but real improvement; geom_rc dropped from
  65.06 to 62.61, single_in_dist from 41.28 to 39.78. Cruise regressed
  slightly (30.23 → 30.95) — heavier pressure weight may have skewed the
  velocity learning a bit.
- **Notes:** Diminishing returns clearly setting in: 4→12 was big, 12→16 is
  modest, probably 16→20 would be even smaller and might hurt cruise more.
  Future iters should pivot to other levers (subsample size, ensemble,
  longer training) instead of pushing surf_p_weight higher.

### 2026-04-27 — iter10: train_subsample 20k→40k, resume from iter8 (RANK 1!)

- **Hypothesis:** with bs=1 predict already gating most of the gain (iter9),
  the residual question is whether the model itself can be improved.
  Doubling `train_subsample` from 20k to 40k means each batch has ~2× the
  point density, giving a richer gradient signal per step at the cost of
  ~halving epochs/min. The model already plateaued at iter8 with the small
  subsample — denser training data might unlock a lower minimum.
- **Change:** `train.py:45` — `train_subsample 20000 → 40000`. Run with
  `--resume_from .../model-6gjpl7q3/`.
- **Result:** 25 epochs, best val_bs4 avg_surf_p = **68.21** at epoch 19
  (vs iter8's 70.62). Real bs=1 val: **45.60** (vs iter8's 52.16, -13%).
  Run `j8o76pto`, commit `5a62210`. Test scoring: avg=**39.75** —
  single=39.39, geom_rc=55.81, geom_cruise=24.86, re_rand=38.92. **Rank 1**
  on the leaderboard, beating nezuko (39.79) and thorfinn (39.91).
- **Verdict:** kept — first place achieved. Sub40k clearly helps; the
  denser sampling per step gives the optimizer a smoother loss landscape.
- **Notes:** 73s/epoch (vs 45s for sub20k) means 25 epochs in 30 min vs 39
  for sub20k. Net win because each epoch is more informative. The lead is
  thin (0.04 over nezuko) — iter11 should consolidate. Big drop in geom_rc
  (88.17→55.81) is the standout; OOD-camber generalization improved more
  than the in-distribution splits, suggesting the denser gradient
  generalizes better.

### 2026-04-27 — iter9: predict batch_size 2→1 + iter8 single (HUGE WIN)

- **Hypothesis (initial):** ensemble iter4+iter7+iter8 at predict time should
  smooth predictions and lower MAE.
- **Surprise discovery:** while implementing the ensemble I evaluated each
  ckpt on val with `batch_size=1` and got **massively** lower scores than
  reported during training. iter8 dropped from val=70.62 (training, bs=4) to
  **val=52.16** (bs=1). Why: batched samples are pad-collated to the largest
  mesh in the batch, and the Transolver's slice attention doesn't mask padding —
  so the slice-token aggregation gets noise from zero-padded nodes, which
  corrupts the surface predictions. Smaller batch = less padding = better
  predictions, with bs=1 (no padding at all) being optimal.
- **Change:** `predict.py` — `batch_size: 2 → 1` default. Also added
  `--checkpoints` (comma-separated) for ensemble support, but ensemble of
  iter4+iter7+iter8 was *worse* than iter8 alone (54.0 vs 52.2 with bs=1).
- **Result (val with bs=1):** iter8=52.16, iter7+iter8=52.60, all-three=54.00.
  Resubmitted iter8 alone with bs=1 (commit `bd5ea0b`). Expect test ≈ 46–52
  (val-test ratio was ~0.88 for prior bs=2 submissions; bs=1 likely tighter).
- **Verdict:** kept — a one-line predict-time change for a 26% drop in val
  MAE without retraining. The biggest win in the run so far.
- **Notes:** This implies my model selection during training was using a
  noisy metric (bs=4 padding-degraded val). The chosen ckpt is still good on
  both metrics, so the impact is small for model selection — but for future
  iterations, fixing train-time val to bs=1 (or at least bs=2) would pick
  cleaner ckpts. Cost: ~4× slower val passes per epoch (probably ~15 fewer
  total epochs in 30 min — likely not worth it). Leaderboard competitors are
  almost certainly *also* affected by this — their submitted predictions
  used whatever batch_size their predict.py shipped, and many probably ran
  at bs=2 too. So the absolute jump for me may translate to a relative jump
  vs everyone else.

### 2026-04-27 — iter8: resume iter7 with lr=5e-5

- **Hypothesis:** iter7 was still descending at the timeout (last epochs
  72.7→72.7) but cosine LR had decayed to ~1.75e-5. A fresh resume cycle
  with peak lr=5e-5 (lower than the iter4-resume's 1.5e-4) gives a gentler
  fine-tune, better suited to a near-converged starting point.
- **Change:** `train.py` — resume-mode default lr `1.5e-4 → 5e-5`. Run with
  `--resume_from .../model-e5uc7jqw/checkpoint.pt`.
- **Result:** 40 epochs, best val avg_surf_p (bs=4) = **70.62** at epoch 32
  → roughly plateaued (last 10 epochs all 70.6–71.4). Real val (bs=1) is
  much lower; see iter9. Splits at training (bs=4): single=55.71,
  geom_rc=100.65, geom_cruise=52.86, re_rand=73.25. Run `6gjpl7q3`, commit
  `6b3b9f5`. Test scoring (bs=2 predict): pending — replaced by iter9.
- **Verdict:** kept — modest 3% gain over iter7 at the bs=4 metric, but the
  ckpt is now my best base for ensembling/predicting.
- **Notes:** Fine-tune basically stopped improving after epoch 10. The lr
  was probably too low to escape the iter7 basin meaningfully — but it did
  consolidate slightly (single 58.6→55.7 was the biggest gain).

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
