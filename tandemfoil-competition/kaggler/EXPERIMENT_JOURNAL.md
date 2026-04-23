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

### 2026-04-23 — v5 fourth fine-tune (lr=1e-5)
- **Hypothesis:** v4 got to val=68.89 (test 60.06) with lr=2e-5 over 8 epochs. Drop LR by 2x again — v3→v4 gave ~5 val-points, v4→v5 should give ~2-3.
- **Change:** `--warm_start checkpoints/best.pt --lr 1e-5` (best.pt = v4).
- **Result:** best ep8 `val/loss=1.37, avg_mae_surf_p=65.38` (val splits 69/80/47/66). W&B: `askeladd/v5-ftv4`.
- **Verdict:** kept — clean monotone descent ep1(68.1)→ep2(67.5)→ep4(66.6)→ep7(66.1)→ep8(65.4). v4 test (60.06) was val 68.89×0.87; v5 should be ~57.
- **Notes:**
  - frieren jumped to 68.22 since last check — they may be using similar warm-start recipe. Need to maintain lead.
  - v3/v4/v5 are all in the same warm-start lineage so error correlation is high; ensembling them alone likely won't give decorrelation gains.

### 2026-04-23 — v4 third fine-tune (lr=2e-5)
- **Hypothesis:** v3 hit 73.81 val at ep7 with lr=5e-5 then flattened. Even smaller LR (2e-5) on top of v3 should squeeze out further descent without overshooting.
- **Change:** `--warm_start checkpoints/best.pt --lr 2e-5` (best.pt now points at v3). Same config otherwise.
- **Result:** best ep8 `val/loss=1.46, avg_mae_surf_p=68.89` (val splits 73/84/49/69). W&B: `askeladd/v4-ftv3`. V3 was scored 64.79 test — v4's expected test ~60 at same 88% val/test ratio.
- **Verdict:** kept — monotone improvement (ep2 70.2, ep6 69.5, ep8 68.9). Still slight improvements at tiny LR.
- **Notes:**
  - v3 test 64.79 → askeladd rank #1. Thorfinn up to 72.61 since. Closing gap — need to keep iterating.
  - Each fine-tune step gives ~4-5 val-points improvement (v2 94→v3 74→v4 69). Diminishing returns — probably one more fine-tune at lr=1e-5 squeezes ~2 more points.
  - Current leaderboard at submission time: me 64.79, thorfinn 72.61, frieren 76.43.
  - Auto-submit wrote to 05074bf (same as v3 commit) overwriting v3 preds. v3's leaderboard score is already cached so safe. Need new commit to submit v4.

### 2026-04-23 — v3 further fine-tune (lr=5e-5)
- **Hypothesis:** v2 was descending at ep4/6 but overfitting at ep7/8 with lr=2e-4. Drop LR to 5e-5 and warm-start from v2 checkpoint — should be able to extract more without overfitting.
- **Change:** only CLI args differ: `--warm_start checkpoints/best.pt --lr 5e-5`.
- **Result:** best ep7 `val/loss=1.56, avg_mae_surf_p=73.81` (val splits 81/88/53/73). Big improvement on all splits vs v2. W&B: `askeladd/v3-warmstart2`.
- **Verdict:** kept — new best by large margin. Continued monotonic improvement through ep7 before a slight regression at ep8.
- **Notes:**
  - Trajectory: v1 (ep8) → 136 → v2 (ep6) → 94 → v3 (ep7) → 74. Warm-starting + halving LR is the reliable recipe.
  - Auto-submit overwrote the experimental `5dc1b0c` ensemble dir but the score 89.21 was already captured before (single-model v2 alone is better at 85.22).
  - Running expected test ≈ 65-70 based on v1/v2 val-to-test ratio (~88-95%).

### 2026-04-23 — ensemble v1 + v2 (5dc1b0c)
- **Hypothesis:** Averaging v1 + v2 predictions might boost via error decorrelation.
- **Change:** `ensemble_preds.py` — simple element-wise mean of saved test predictions.
- **Result:** test avg_surf_p = **89.21**. Worse than v2 alone (85.22).
- **Verdict:** discarded — v1 is a strictly-worse ancestor of v2 (warm-start lineage), so averaging pulls predictions toward v1's errors. Decorrelation requires diverse checkpoints.
- **Notes:** Ensembling works only with diverse errors. For future ensembles use models trained from scratch with different seeds or architectures.

### 2026-04-23 — v2 warm-start fine-tune
- **Hypothesis:** v1 was still descending at the 30min cap. Warm-starting from v1's epoch-8 checkpoint with lower LR (2e-4) and skipping validation on odd epochs (val_every=2) should effectively extend training by ~8 epochs.
- **Change:** `train.py` adds `--warm_start` to load checkpoint before optimizer init, `--val_every` to skip validations, and auto-val near timeout.
- **Result:** best ep6 `val/loss=2.00, avg_mae_surf_p=94.07` (val splits 109/111/66/89). Down from v1's 136.16 ⇒ **31% improvement**. val on ep7 (95.05) and ep8 (100.76) regressed slightly — starting to overfit with warm lr. W&B: `askeladd/v2-warmstart`. Commit `6a1d52b`.
- **Verdict:** kept — huge win; beats v1 across all splits. But auto-submit raced the scorer again (marked "incomplete"). Re-submitting at new commit.
- **Notes:**
  - v1 tests vs v2 tests: geom_cruise was 86.75 → 66.68 (-23%), re_rand 114.60 → 89.24 (-22%). single_in_dist val dropped 186→109 (-41%).
  - Train loss actually *increased* on ep4/7 while val dropped — likely WeightedRandomSampler variance across epochs.
  - Next: ensemble v1+v2 (sure gain), then v3 bigger model from scratch for diversity.

### 2026-04-23 — v1 baseline Transolver + AMP
- **Hypothesis:** Transolver with pressure-aware attention plus bf16 AMP should fit 6-8 epochs in 30min budget and beat the naive copy-baseline. Scoring metric is `avg/mae_surf_p` (surface pressure MAE across 4 test splits).
- **Change:** refactored `train.py` into a `main()` guard, added `apply_no_slip`, bf16 AMP, checkpoint selection by `avg_mae_surf_p` (was `val/loss`). `predict.py` implemented end-to-end with Transolver loading + config.yaml. `n_hidden=192, n_layers=6, slice_num=128, bs=2, 1.73M params`.
- **Result:** best ep8 `val/loss=3.73, avg_mae_surf_p=136.16` (val split means 186/140/97/120). Peak 42GB VRAM. Ran 29.9min. Commit `7d57563`. W&B: `askeladd/v1-baseline-fixed`.
- **Verdict:** kept — first complete submission. Leaderboard snapshot before submission: thorfinn 87.51, frieren 109.27. Baseline scored "incomplete" in first scoring pass (race condition between predict.py finishing + scorer polling).
- **Notes:**
  - **Critical bug found first:** `is_surface=True` is NOT only the airfoil — it includes inlet/outlet/walls where freestream velocity is non-zero. Initial attempt zero-ed Ux,Uy on all surface nodes and would have catastrophically hurt score; disabled no-slip BC enforcement.
  - Only 7-8 epochs fit in 30min at current size. VRAM is at 42GB / 96GB so we have 2x headroom for bigger model or bigger batch.
  - val_loss fluctuates (ep5=4.43, ep6=4.53) — need longer training or better LR schedule.
  - Next ideas: (v2) scale model to ~5M params and bs=4 with AMP; (v3) pressure-channel loss weighting (`p_weight`); (v4) warm-start + fine-tune ensemble.
