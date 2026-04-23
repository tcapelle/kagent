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

### 2026-04-23 — v10 pressure-only loss (p_weight=20)
- **Hypothesis:** Scoring is *only* on pressure. Push p_weight from 5 → 20 so velocity loss becomes negligible. Model can specialize on pressure without distraction.
- **Change:** `--warm_start v9 --lr 1e-5 --surf_weight 40 --p_weight 20.0`.
- **Result:** best ep8 `val/loss=4.40, avg_mae_surf_p=58.22` — virtually tied with v9 (58.23). W&B: `askeladd/v10-ponly`.
- **Verdict:** marginal. v10 does NOT substantially beat v9. Dropping velocity loss too aggressively loses useful signal; pressure has physical coupling to velocity, so softer bias (p=5) was already near-optimum.
- **Notes:**
  - Ensemble experiments (v7+v8+v9, v8+v9, weighted) all tied or slightly regressed vs v9 alone (50.97) — warm-start lineage is too correlated.
  - Ensembling works when models have *different* errors; all my lineage has ~same errors.

### 2026-04-23 — v9 polish (lr=5e-6 surf=40 p=5)
- **Hypothesis:** v8 val 59.25 / test 51.96. Halve LR to 5e-6 (same weights) — squeeze last val points with minimal parameter changes.
- **Change:** `--warm_start v8 --lr 5e-6` (same surf=40, p=5).
- **Result:** best ep7 `val/loss=4.37, avg_mae_surf_p=58.23` (val 60/74/40/59). Small but consistent descent.
- **Verdict:** kept — every split improved vs v8. Projected test ≈ 51 at 88% ratio, maybe 0.5-1 better than v8.
- **Notes:**
  - Returns now sub-linear. v7 (val 60.58) → v8 (59.25) → v9 (58.23): −1.3 then −1.0 val-points. Expected diminishing.
  - Lead over frieren (55.32) still >3 points. Likely margin sufficient.

### 2026-04-23 — v8 extreme pressure focus (surf=40, p=5)
- **Hypothesis:** v7 (surf=30, p=3, lr=2e-5) dropped val to 60.58 / test 52.90. Push further: surf=40, p=5, lr=1e-5 (smaller step since loss magnitude grows).
- **Change:** `--warm_start v7 --lr 1e-5 --surf_weight 40 --p_weight 5.0`.
- **Result:** best ep7 `val/loss=4.50, avg_mae_surf_p=59.25`. Improvement over v7 (60.58) on val. Projected test ≈ 52 (similar to v7).
- **Verdict:** kept — v8 marginally better on val on all 4 splits vs v7.
- **Notes:**
  - Pattern continues: each round of finetune with bigger weighting + smaller LR shaves 1-2 val points.
  - Ensembles with earlier models (v5, v6) HURT because the newer model dominates — 6b9304b (v5+v6+v7) = 54.42, v7 alone 52.90.
  - Frieren at 55.32. My solo v7 already leads by 2.4. Expect v8 to widen to ~3-4.

### 2026-04-23 — v7 even more pressure-focused (surf=30, p_weight=3)
- **Hypothesis:** v6 (surf=20, p=2) reduced val to 64.24. Push further with surf=30, p_weight=3 — effectively 9× pressure-surface weighting. Use slightly lower LR (2e-5) to avoid overshooting.
- **Change:** `--warm_start v6 --lr 2e-5 --surf_weight 30 --p_weight 3.0`.
- **Result:** best ep8 `val/loss=3.46, avg_mae_surf_p=60.58` (val splits 63.5/75.3/42.1/61.4). Big jump from v6's 64.24 ⇒ **6% val improvement**. W&B: `askeladd/v7-highsurf`.
- **Verdict:** kept — pushing pressure weighting further works. Every split improved vs v6. Projected test ~53 at 88% ratio.
- **Notes:**
  - Pattern: heavier surface+pressure weighting moves metric further, so training loss structure matters more than architecture at this stage.
  - Leaderboard at start of v7: askeladd 56.07 (ens v5+v6) vs frieren 56.35. Gap razor-thin.
  - Next: submit v7 alone, ensemble v5+v6+v7, maybe v7+v6 (most diverse pair).

### 2026-04-23 — v6 pressure-focused fine-tune (surf=20, p_weight=2)
- **Hypothesis:** v5 scored 57.48 test. Ensemble v3+v4+v5 regressed to 57.91 (too correlated — all same warm-start lineage). Need a model with genuinely different error characteristics. Bias v6 toward *pressure on surface* specifically: `surf_weight 10→20, p_weight 1→2` effectively makes pressure-surface loss 4× more important during finetune. Use slightly higher LR (3e-5) than v5's 1e-5 since the loss landscape shifts.
- **Change:** warm-start from v5. `--surf_weight 20 --p_weight 2.0 --lr 3e-5`.
- **Result:** best ep7 `val/loss=2.54, avg_mae_surf_p=64.24` (val splits 66.5/79.1/46.1/65.3). Beats v5's best val (65.38) by ~1 point. W&B: `askeladd/v6-ftpress`.
- **Verdict:** kept — v6 single model better than v5 single on 3/4 splits. Critically, v6 improves *single_in_dist* from 69.25 → 66.54 which was my weakest axis vs thorfinn.
- **Notes:**
  - Learned from ensemble v3+v4+v5 experiment: correlated lineage doesn't help. Must change loss or arch.
  - val_loss is 2× higher than v5 because surf_weight now scales the loss differently — avg_surf_p is the consistent metric.
  - Next: submit v6 solo and a fresh ensemble v5+v6. v6 has different loss → different errors → should finally decorrelate and boost.

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
