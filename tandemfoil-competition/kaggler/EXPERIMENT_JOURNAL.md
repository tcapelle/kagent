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

### 2026-04-28 — iter26: rebalance loss (sb 15→6, spw 14→10, volw 0.3→0.4)
- **Hypothesis:** iter23-25 with sb=15+spw=14 may have over-fit on racecar_single. Rebalance: drop sb to 6, surf_p_w to 10, raise vol_w to 0.4. Expect val_geom_camber_rc to recover (was 4.07).
- **Change:** `--single_boost 6.0 --surf_p_weight 10 --vol_p_weight 0.4 --vol_uv_weight 0.4 --lr 2e-6` (small LR perturb).
- **Result:** 7 epochs, best val/avg_surf_p=50.65 at epoch 7 (-0.50 vs iter25). Trajectory: 52.53 → 53.56 → 51.64 → 51.52 → 50.75 → 50.86 → 50.65. Predictions at `askeladd/9cc6c53`. W&B: askeladd/iter26-rebalance-sb6-spw10. **Below 51!**
- **Verdict:** kept (-0.50). val_geom_camber_rc loss 4.07 → 2.98 (back to where it was at iter22). val_single_in_dist 2.35 → 1.74 (also improved). All four splits got better.
- **Notes:** Big lesson: sb=15 was *over-emphasising* single_in_dist — pulling capacity away from the other splits in a way that hurt the avg metric even though val_single_in_dist was still nominally improving. The iter23 perturb was a local trick that backfired in the long run; rebalancing recovered. iter27: continue this recipe at lr=1e-6.

### 2026-04-28 — iter25: chain at sb=15 + lr=1e-6
- **Hypothesis:** Continue.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=51.15 at epoch 6 (-0.08 vs iter24). Trajectory: 51.43 → 51.87 → 51.47 → 51.28 → 51.39 → 51.15 → 51.19. Predictions at `askeladd/10e623c`. W&B: askeladd/iter25-chain-sb15.
- **Verdict:** kept (-0.08). Diminishing.
- **Notes:** iter26: try rebalancing — drop sb 15→6 and surf_p_weight 14→10 to reduce overfit on single, give some weight back to volume.

### 2026-04-28 — iter24: settle at sb=15 + lr=1e-6
- **Hypothesis:** Drop LR back to 1e-6, keep iter23's sb=15 + surf_p_w=14.
- **Change:** `--lr 1e-6`. sb=15 unchanged.
- **Result:** 7 epochs, best val/avg_surf_p=51.23 at epoch 7 (-0.22 vs iter23). Trajectory: 52.16 → 52.13 → 52.05 → 51.69 → 51.45 → 51.30 → 51.23. Predictions at `askeladd/9c6b189`. W&B: askeladd/iter24-chain-sb15.
- **Verdict:** kept (-0.22).
- **Notes:** test scores caught up: iter22 (18fb32d) at test 44.95, rank 8. Top still 33.50. Continue.

### 2026-04-28 — iter23: stronger perturb (sb=15, surf_p_w=14, lr=2e-6)
- **Hypothesis:** iter22 plateaued. Push harder: sb 12→15, surf_p_weight 12→14, double LR to 2e-6 (perturb). The surf_p_weight bump may give the model more headroom on pressure.
- **Change:** `--single_boost 15.0 --surf_p_weight 14 --lr 2e-6`.
- **Result:** 7 epochs, best val/avg_surf_p=51.45 at epoch 7 (-0.43 vs iter22). Trajectory: 53.22 → 53.02 → 52.53 → 52.07 → 52.33 → 51.75 → 51.45. Predictions at `askeladd/18fb32d`. W&B: askeladd/iter23-perturb-sb15-spw14.
- **Verdict:** kept (-0.43). Notable side effect: val_geom_camber_rc loss went UP (3.53 → 4.07) — losses are dominated by surface_p_weight=14 now, but the *avg_surf_p* metric still improved because single_in_dist absorbed more capacity.
- **Notes:** iter24: drop LR back to 1e-6 to consolidate, watch geom_camber_rc.

### 2026-04-28 — iter22: chain at sb=12 + lr=1e-6
- **Hypothesis:** Continue the iter21 recipe.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=51.88 at epoch 7 (-0.08 vs iter21). Trajectory: 52.71 → 52.38 → 52.41 → 52.45 → 52.22 → 52.19 → 51.88. Predictions at `askeladd/d71547a`. W&B: askeladd/iter22-chain-sb12.
- **Verdict:** kept (-0.08). Diminishing returns again.
- **Notes:** iter23: stronger perturb (sb=15, surf_p_weight 12→14, lr=2e-6) to push out of plateau.

### 2026-04-28 — iter21: chain at sb=12 + lr=1e-6 (still grinding)
- **Hypothesis:** Continue.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=51.96 at epoch 7 (-0.40 vs iter20). Trajectory: 52.63 → 52.64 → 52.73 → 52.37 → 52.33 → 52.36 → 51.96. Predictions at `askeladd/8981d50`. W&B: askeladd/iter21-chain-sb12.
- **Verdict:** kept (-0.40). Below 52! sb=12 chain still has more headroom.
- **Notes:** val_single_in_dist 2.13 → 2.11. Continue.

### 2026-04-28 — iter20: settle at sb=12 + lr=1e-6
- **Hypothesis:** iter19 succeeded at sb=12+lr=2e-6 but the extra LR pushes the model around. Drop LR to 1e-6 to consolidate.
- **Change:** `--lr 1e-6` (lr halved). sb=12 unchanged.
- **Result:** 7 epochs, best val/avg_surf_p=52.36 at epoch 7 (-0.27 vs iter19). Trajectory: 53.38 → 53.41 → 52.88 → 52.85 → 52.67 → 52.43 → 52.36. Predictions at `askeladd/b3a7883`. W&B: askeladd/iter20-chain-sb12-lr1e6.
- **Verdict:** kept (-0.27).
- **Notes:** val_single_in_dist 2.16 → 2.13. iter21: same recipe, see if more headroom.

### 2026-04-28 — iter19: perturb out of plateau (sb=12 + lr=2e-6)
- **Hypothesis:** iter18 plateaued at 53.10. Higher LR + more single_boost should push the model out of the local min toward a better one.
- **Change:** `--single_boost 12.0 --lr 2e-6`. Doubled LR (vs iter18's 1e-6), bumped sb 8 → 12.
- **Result:** 7 epochs, best val/avg_surf_p=52.63 at epoch 6 (-0.47 vs iter18). Trajectory: 55.87 → 54.21 → 54.16 → 53.85 → 53.41 → 52.63 → 52.82. Predictions at `askeladd/90d0891`. W&B: askeladd/iter19-sb12-lr2e6-perturb.
- **Verdict:** kept (-0.47). Perturbation worked — escaping the plateau cost epoch 1 (55.87) but ended better than iter18.
- **Notes:** val_single_in_dist 2.20 → 2.16 (single_boost still pushing). iter20: drop LR back to 1e-6 to consolidate at sb=12.

### 2026-04-28 — iter18: chain sb=8 (plateau)
- **Hypothesis:** Continue.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=53.10 at epoch 6 (-0.03 vs iter17). Trajectory: 53.61 → 53.81 → 53.72 → 53.84 → 53.52 → 53.10 → 53.16. Predictions at `askeladd/073b149`. W&B: askeladd/iter18-chain-sb8.
- **Verdict:** kept (-0.03). Plateau at sb=8/lr=1e-6.
- **Notes:** Test scores caught up: iter17 (b8419dd) is at test 46.40 currently rank 8. Top at 34.32 — about 12 surf_p gap. iter19: perturb with sb=12 + lr=2e-6 to escape plateau.

### 2026-04-28 — iter17: continue chain at sb=8 (steady grind)
- **Hypothesis:** Same recipe as iter15/16. Confirm chain still has slack.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=53.13 at epoch 7 (-0.35 vs iter16). Trajectory: 53.95 → 54.31 → 53.30 → 53.77 → 53.66 → 53.33 → 53.13. Predictions at `askeladd/b8419dd`. W&B: askeladd/iter17-chain-sb8.
- **Verdict:** kept (-0.35).
- **Notes:** val_single_in_dist 2.25 → 2.23. Continue.

### 2026-04-28 — iter16: continue chain at sb=8 + lr=1e-6
- **Hypothesis:** Same recipe as iter15. Confirm sb=8 isn't a one-shot win.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=53.48 at epoch 7 (-0.29 vs iter15). Trajectory: 54.27 → 54.35 → 54.51 → 53.87 → 53.85 → 53.58 → 53.48. Predictions at `askeladd/557e861`. W&B: askeladd/iter16-chain-sb8.
- **Verdict:** kept (-0.29). val_single_in_dist 2.29 → 2.25; tiny improvements across all splits.
- **Notes:** iter17 continues same recipe.

### 2026-04-27 — iter15: bump single_boost 5 → 8
- **Hypothesis:** sb=5 gain plateaued (-0.16 in iter14). Push to sb=8 to keep the racecar_single signal dominant — model may have absorbed sb=5 already.
- **Change:** `--single_boost 8.0`. Same lr=1e-6 otherwise.
- **Result:** 7 epochs, best val/avg_surf_p=53.77 at epoch 7 (-0.59 vs iter14). Trajectory: 55.10 → 54.56 → 54.55 → 54.28 → 54.09 → 53.83 → 53.77. Predictions at `askeladd/6343096`. W&B: askeladd/iter15-sb8-lr1e6.
- **Verdict:** kept (-0.59). Bigger gain than expected when bumping single_boost mid-chain — confirms racecar_single was still under-represented even at sb=5.
- **Notes:** val_single_in_dist 2.34 → 2.29. iter16: continue at sb=8 to consolidate, but watch for over-emphasis (other splits not regressing yet).

### 2026-04-27 — iter14: chain at sb=5 + lr=1e-6 (more grinding)
- **Hypothesis:** Same recipe; iter13 dropped 0.74. See if chain still has slack.
- **Change:** None.
- **Result:** 7 epochs, best val/avg_surf_p=54.36 at epoch 7 (-0.16 vs iter13). Trajectory: 54.93 → 54.76 → 55.00 → 54.67 → 54.39 → 54.39 → 54.36. Predictions at `askeladd/77458e5`. W&B: askeladd/iter14-chain-sb5.
- **Verdict:** kept (-0.16). Diminishing returns kicking in for sb=5 — gain shrunk from 0.74 → 0.16 in one iter. Time to perturb.
- **Notes:** iter15: try `single_boost=8.0` to push harder before plateauing.

### 2026-04-27 — iter13: chain at sb=5 + lr=1e-6 (consolidate)
- **Hypothesis:** iter12's `single_boost=5.0` win wasn't a fluke. Continue same recipe to consolidate.
- **Change:** None — same as iter12.
- **Result:** 7 epochs, best val/avg_surf_p=54.52 at epoch 6 (-0.74 vs iter12). Trajectory: 55.96 → 55.74 → 55.29 → 55.25 → 54.80 → 54.52 → 54.56. Predictions at `askeladd/ea6e94a`. W&B: askeladd/iter13-chain-sb5.
- **Verdict:** kept (-0.74). Bigger gain than iter12 — single_boost is still pushing improvements through. All 4 splits dropped.
- **Notes:** Per-split val: single_in_dist 2.34, geom_camber_rc 3.57, geom_camber_cruise 1.48, re_rand 2.67. Single is improving ~0.05/iter; the others tiny. iter14: same recipe, see if we keep grinding down.

### 2026-04-27 — iter12: bump single_boost 3.5 → 5.0
- **Hypothesis:** val_single_in_dist is still my worst-relative split. iter7's `single_boost=2.5` and iter8's 3.5 both gave small wins; try 5.0.
- **Change:** `--single_boost 5.0`. Keep lr=1e-6, single_boost was the only knob moved.
- **Result:** 7 epochs, best val/avg_surf_p=55.26 at epoch 7 (-0.45 vs iter11). val_single_in_dist 2.42 → 2.37; geom_camber_rc, geom_cruise, re_rand all improved a hair too. Trajectory: 56.51 → 56.40 → 56.03 → 55.65 → 55.40 → 55.49 → 55.26. Predictions at `askeladd/3aa93a7`. W&B: askeladd/iter12-singleboost5-lr1e6.
- **Verdict:** kept (-0.45). Higher single_boost wins without hurting tandem/cruise — racecar_single domain is just data-poor at the standard 1/3 weight.
- **Notes:** Each iter at this stage gives ~0.5 surf_p. iter13: continue same recipe (sb=5, lr=1e-6) to consolidate.

### 2026-04-27 — iter11: bump LR back from 5e-7 to 1e-6
- **Hypothesis:** iter10 at lr=5e-7 was almost flat (-0.23). LR may now be too low to make further progress; bump back to iter9's level (1e-6) and see if there's still room.
- **Change:** `--lr 1e-6`. Same recipe.
- **Result:** 7 epochs, best val/avg_surf_p=55.71 at epoch 6 (-0.5 vs iter10). Trajectory: 56.87 → 56.65 → 56.24 → 56.07 → 56.40 → 55.71 → 55.72. Predictions at `askeladd/26ebe1a`. W&B: askeladd/iter11-chain-back-lr1e6.
- **Verdict:** kept (-0.5). lr=5e-7 is too low — going back up to 1e-6 found a tiny bit more headroom. Best val so far.
- **Notes:** Slow grind. Each iter at 1e-6 / 5e-7 nets ~0.3-0.5 surf_p. iter12: try `single_boost=5.0` to push the racecar_single domain harder — current val_single_in_dist (2.42) is still the worst-relative split.

### 2026-04-27 — iter10: chain at lr=5e-7 (floor test) + ensemble experiment
- **Hypothesis:** iter9 was -1.13 vs iter8; if I drop LR another 2x to 5e-7, gains should shrink — confirming we're near the chain's floor and the plateau is real.
- **Change:** `--lr 5e-7`. Same recipe otherwise.
- **Result:** 7 epochs in 30 min, best `val/avg_surf_p=56.21` at epoch 7. Trajectory: 56.66 → 56.59 → 56.31 → 56.43 → 56.45 → 56.47 → 56.21. Predictions at `askeladd/881f29e`. W&B: askeladd/iter10-chain-lr5e7-final.
- **Ensemble experiments (eval_ensemble.py):**
  - iter9 + iter10 (closest two): val/surf_p=56.24 — no help (chain ckpts are too correlated).
  - iter6 + iter9 + iter10 (wider spread): val/surf_p=57.21 — worse (iter6's higher loss drags average up).
  - All chain checkpoints lie along the same gradient path — averaging within the chain doesn't add diversity.
- **Verdict:** kept (-0.23 vs iter9). Confirms the chain plateau at val/surf_p ≈ 56. Ensemble within the chain is not the right move; would need an independent training run with a different seed/architecture for diversity.
- **Notes:** Test scores from iter9 (b977533) put me at 48.83 (rank 8 currently — others passed me). Top is at 35.24. Chain alone won't close the gap; need a structural change.

### 2026-04-27 — iter9: chain at lr=1e-6 (no perturbation)
- **Hypothesis:** iter8 paid 1 epoch to recover from changing `surf_p_weight` mid-chain. Drop LR 2x to 1e-6, keep all other settings the same. No loss-shape change → no perturbation, just slower polish.
- **Change:** `--lr 1e-6`. Everything else same as iter8.
- **Result:** 7 epochs in 30 min, best `val/avg_surf_p=56.44` at epoch 7. Trajectory: 57.07 → 57.53 → 57.08 → 57.23 → 56.65 → 56.57 → 56.44. Predictions at `askeladd/b977533`. W&B: askeladd/iter9-chain-lr1e6.
- **Verdict:** kept (-1.13 vs iter8). No epoch wasted to perturbation.
- **SWA experiment:** Tried averaging weights of {iter7, iter8, iter9} → val/surf_p=57.19 (worse than iter9 alone 56.44). Tried {iter8, iter9} → 56.78 (still worse). The chain was monotonically improving so each successive checkpoint dominates the average. SWA only helps when checkpoints are sampled around a flat minimum. Skip SWA at this stage.
- **Notes:** Diminishing returns are real — 60 → 57 → 56. Tried bumping LR back up (frieren did this and got similar results). Next angle for iter10: tinier LR=5e-7 to confirm we've hit the floor, then consider architectural changes if no progress.

### 2026-04-27 — iter8: stronger single_boost + bigger surf_p_weight
- **Hypothesis:** iter7 showed single_boost works without hurting other splits. Push it harder: `single_boost 2.5→3.5`, `surf_p_weight 10→12`. Same lr=2e-6, nosub, bs=2.
- **Change:** Just CLI args. No code change.
- **Result:** 7 epochs in 30 min, best `val/avg_surf_p=57.57` at epoch 7. Trajectory: 63.36 → 59.05 → 58.51 → 58.35 → 57.82 → 57.96 → 57.57. Predictions at `askeladd/01851f9`. W&B: askeladd/iter8-singleboost3.5-spw12.
- **Verdict:** kept (-0.85 vs iter7). Smaller win; epoch 1 was 63.36 — bumping `surf_p_weight` from 10→12 perturbed warm-started weights and cost ~1 epoch of recovery.
- **Notes:** Trade-off showed up: `val_single_in_dist` actually regressed (iter7 final 2.17 → iter8 final 2.50) while overall surf_p improved — the model traded single-foil accuracy for the other splits. Lesson: bumping the `surf_p_weight` in the *middle* of a chain costs an epoch of warm-start advantage; if the loss is good keep it stable. iter9: keep iter8's recipe but drop LR another 2x to 1e-6 (no loss change → no perturbation, just polish).

### 2026-04-27 — iter7: single_boost=2.5 to push the racecar_single domain
- **Hypothesis:** Per-split test gap analysis from iter5 showed `test_single_in_dist` was my biggest weakness (69.6 vs top 50.0). The training sampler is balanced 1/3 per domain {racecar_single, racecar_tandem, cruise}; upweighting racecar_single in the WeightedRandomSampler should give the model more single-foil exposure.
- **Change:** `train.py`: added `single_boost: float` config; multiplies sample_weights for samples in the `racecar_single` domain group (read from `meta.json`). Run with `--single_boost 2.5` and the same chain recipe (lr=2e-6, nosub, bs=2, 4-weight loss).
- **Result:** 7 epochs in 30 min, best `val/avg_surf_p=58.42` at epoch 7. Trajectory: 61.33 → 60.33 → 59.10 → 59.03 → 58.84 → 58.46 → 58.42. `val_single_in_dist` improved from iter6's 2.32 → iter7's 2.17 (the targeted split). Predictions at `askeladd/170bb37`. W&B: askeladd/iter7-singleboost2.5-lr2e6.
- **Verdict:** kept (-1.79 vs iter6). Single boost works without hurting other splits — none of them regressed. Worth pushing further.
- **Notes:** iter8: bump `single_boost` to 3.5 and `surf_p_weight` 10→12 for more aggressive surface-pressure focus on the hard split.

### 2026-04-27 — iter6: chain at lr=2e-6 (frieren chain step)
- **Hypothesis:** With iter5 settled at val/surf_p=63.80, drop LR another 2.5x to 2e-6 (frieren's iter4 LR) and let the model polish for 6 more full-mesh epochs.
- **Change:** No code changes. `--lr 2e-6 --epochs 8` (rest unchanged).
- **Result:** 6 epochs in 30 min, best `val/avg_surf_p=60.21` at epoch 6. Trajectory monotone: 62.30 → 61.99 → 61.25 → 61.20 → 61.02 → 60.21. Predictions at `askeladd/5079a56`. W&B: askeladd/iter6-nosub-bs2-lr2e6-chain2.
- **Verdict:** kept (-3.6 vs iter5). Diminishing returns at this LR — each epoch nets <1 surf_p. Test scores from iter5 put me at rank 5 (53.32) — top tier (45-46) is fern, frieren, thorfinn.
- **Notes:** Per-split test gap analysis (iter5 53.32 vs top): single_in_dist 69.6 vs 50.0 (largest gap), geom_rc 67.6 vs 59.8, geom_cruise 28.6 vs 25.1, re_rand 47.5 vs 42.5. **single_in_dist is by far the biggest opportunity** — closing it could move me into top 3. iter7: add `--single_boost` to upweight racecar_single domain in the WeightedRandomSampler.

### 2026-04-27 — iter5: drop subsampling, bs=2, lr=5e-6 (frieren's chain trick)
- **Hypothesis:** frieren's W&B trace (now 2nd at 46.87) showed iter3+iter4 ditched `train_subsample=40000` and switched to `bs=2, lr=5e-6` then `lr=2e-6`. Subsampling during training and validating on full meshes is a distribution shift — once the model is well-trained, this shift hurts more than the speed helps. Switching to no-sub + tiny LR for the final fine-tune should close that gap.
- **Change:** `--lr 5e-6 --train_subsample 0 --batch_size 2 --epochs 8` (rest of 4-weight loss unchanged).
- **Result:** 6 epochs in 30 min (full-mesh epochs are ~5 min vs ~1.3 min subsampled). best `val/avg_surf_p=63.80` at epoch 5. Trajectory: 75.05 → 71.23 → 65.17 → 67.21 → 63.80 → 63.92. **Epoch 1 alone dropped from 87 (iter4 final) to 75** — the subsample distribution shift was real and immediately costly. Predictions at `askeladd/3dd4327`. W&B: askeladd/iter5-nosub-bs2-lr5e6.
- **Verdict:** kept (-24 surf_p vs iter4, the biggest jump in the chain). Should jump several places on the leaderboard.
- **Notes:** Big lesson: subsampling is a useful **early-stage** tool to amortize compute and grow batch size, but it should be turned off for the final fine-tune. iter6: chain again at lr=2e-6 to follow frieren's pattern, see how much further the no-sub regime can go.

### 2026-04-27 — iter4: chain warm-start lr=2e-5 + surf_p_weight=10
- **Hypothesis:** With the loss now in thorfinn's 4-weight form and a usable iter3 base (99.84), continue the chain: drop LR another 2.5x to 2e-5, push `surf_p_weight` 6→10 (and `vol_p_weight` 0.5→0.3 since it's not the metric) to focus capacity on the leaderboard objective.
- **Change:** No code changes. Just `--lr 2e-5 --surf_p_weight 10 --vol_p_weight 0.3 --vol_uv_weight 0.3 --epochs 20`.
- **Result:** 19 epochs in 26 min (timeout). best `val/avg_surf_p=87.47` at epoch 19 (just barely below 90). Trajectory: 104 → 107 → 105 → 102 → 104 → 99 → 96 → 98 → 99 → 94 → 95 → 94 → 92 → 90 → 89 → 89 → 88 → 88 → 87. Predictions at `askeladd/a2bed01`. W&B: askeladd/iter4-chain-lr2e5-spw10.
- **Verdict:** kept (-12 surf_p vs iter3). On the leaderboard at rank 7 with iter3's 92.25 test score; iter4 should bump me up.
- **Notes:** Chain is paying off: 137 (iter1) → 114 (iter2) → 100 (iter3) → 87 (iter4). frieren is at 53.32 with bs=2, lr=2e-6, NO subsampling (chain3); they finish each chain with full meshes. Iter5: try the same — `train_subsample=0`, bs=2, lr=5e-6, see if removing the subsample distribution shift gives a final boost.

### 2026-04-27 — iter3: 4-weight loss (thorfinn recipe) + warm-start + 20 epochs
- **Hypothesis:** Replace the simple `surf_weight × (uv+p)/3` loss with thorfinn's 4-region/channel weighting `surf_p_w=6, surf_uv_w=1, vol_p_w=0.5, vol_uv_w=0.5`. Their config (W&B: `thorfinn/iter1-warmstart-surfp` → 54.81) suggests this gives finer control over what the model optimises. Continue the chain warm-start + sub40k + bs4 + lr5e-5 setup that worked in iter2.
- **Change:** `train.py`: replaced the chan_w/surf_weight scheme with four explicit weights; total loss `= surf_p_weight*l_surf_p + surf_uv_weight*l_surf_uv + vol_p_weight*l_vol_p + vol_uv_weight*l_vol_uv`. Switched val split_loss to a similar weighted sum. epochs 8→20.
- **Result:** 19 epochs in ~26 min (timeout). best `val/avg_surf_p=99.84` at epoch 19. Trajectory was non-monotonic (loss reformulation perturbed warm-started weights → epoch 1 surf_p=210 vs iter2 final 114), but recovered: 210 → 193 → 169 → 152 → 164 → 142 → 145 → 123 → 123 → 127 → 115 → 112 → 124 → 105 → 105 → 105 → 102 → 100 → 100. Predictions at `askeladd/84fb943`. W&B: askeladd/iter3-4weight-warm-lr5e5-20e.
- **Verdict:** kept (-15 surf_p vs iter2). Crossed below 100 — should land top-3 once scored. Improvement still happening at the end → iter4 should chain further.
- **Notes:** Surprise: iter2's effective surface-pressure weight (`surf_weight=15 × p_chan_w=4 / 3 ≈ 20`) was actually higher than iter3's (`surf_p_weight=6`), yet iter3 ended better. The relative `surf_p / vol_p` ratio matters less than I thought — what mattered was giving the model dedicated coefficients per region/channel so the optimiser can reweight cleanly. Mistakes: I ran `python -c "import train"` to "validate" the syntax — that triggered a real training run and crashed mid-epoch on a stale `cfg.surf_weight` reference; now removed. iter4: keep chain, drop LR to 2e-5, push surf_p_weight to 10 to see if more pressure emphasis helps now that the model is well-trained.

### 2026-04-27 — iter2: warm-start + point subsampling (40k) + low LR
- **Hypothesis:** W&B inspection of competitor configs showed the leaders (thorfinn, edward, frieren) all use `train_subsample=40000` with `batch_size=4`. Subsampling cuts per-epoch wall-clock ~3x → fits more epochs in the 30 min budget. Warm-start from iter1 with much lower LR (5e-5, matching thorfinn) lets the model fine-tune without overshooting.
- **Change:** `train.py`: add `train_subsample` (drop random volume nodes per training sample, keep all surface nodes via top-k score trick). bs 2→4, val_batch_size kept at 2 since validation runs on full meshes. p_weight 1→4 (extra surface-pressure emphasis). lr 8e-4→5e-5 (after killing a first attempt at 2e-4 that diverged from warm-start: train surf loss exploded 0.35 → 0.78 in epoch 1). epochs 50→15 so cosine actually anneals.
- **Result:** Warm-start from iter1's `checkpoints/best.pt` succeeded (no missing keys). 14 epochs in 22 min (~82 s/epoch). best `val/avg_surf_p=114.43` at epoch 14. Trajectory monotonic-ish: 138.91 → 136.5 → 133.8 → 135.9 → 128.1 → 128.2 → 122.7 → 121.9 → 119.6 → 120.5 → 116.4 → 115.4 → 115.3 → 114.4. Predictions at `askeladd/3cd9ef3`. W&B: askeladd/iter2-sub40k-bs4-warm-pw4-lr5e5.
- **Verdict:** kept (-22 surf_p vs iter1). Still 2x worse than thorfinn's 54 — gap suggests they have a much better warm-start chain or a smarter loss.
- **Notes:** First attempt at lr=2e-4 destabilised the warm-started weights immediately (epoch 1 surf_p=204, train losses doubled). Lesson: when warm-starting from a converged checkpoint, drop LR by 10-100x. Inspecting thorfinn's W&B config revealed the gap — they use 4-weight loss `(surf_p_w=6, surf_uv_w=1, vol_p_w=0.5, vol_uv_w=0.5)` instead of my `surf_weight=15` × `chan_w=[1,1,4]`. iter3 will adopt that scheme.

### 2026-04-27 — iter1: bigger Transolver + L1 loss + bf16
- **Hypothesis:** A bigger Transolver (256/6/128 vs default 128/5/64) with L1 loss in normalised space (matches the per-channel MAE metric exactly) and surf_weight=15 should land in the top half of the leaderboard. Add bf16 autocast + grad clip for stability and speed.
- **Change:** `train.py`: model 128/5/64 → 256/6/128 (3M params), loss MSE → L1 in normalised space, surf_weight 10 → 15, lr 5e-4 → 8e-4, batch_size 4 → 2 (Cruise samples are big), bf16 autocast, grad_clip=1.0, save best by `val/avg_surf_p` (the leaderboard metric), mirror checkpoints to PVC + `checkpoints/best.pt`. Factored model to `model.py`. Fixed `predict.py` (removed `NotImplementedError`, load Transolver from `model.py`, bf16 inference).
- **Result:** 6 epochs in 30 min (timeout). best `val/avg_surf_p=136.76` at epoch 6. Trajectory: 214 → 181 → 167 → 150 → 142 → 137 → 144. Peak VRAM 56 GB. W&B: askeladd/iter1-256x6-L1-bf16-sw15.
- **Verdict:** kept. Predictions saved at `askeladd/634f51a`. Improvement still flat at end → more epochs would help; should warm-start.
- **Notes:** auto-submit subprocess crashed because `predict.py` did `from train import Transolver` (which executed train.py at import time and parsed conflicting CLI args). Fixed in this commit. Cosine T_max=50 is wrong since we only do 6 epochs — LR stays near peak. Iter2: lower LR and warm-start, set epochs to ~8 so cosine actually anneals. Add per-channel weight on surface pressure (the leaderboard metric) — `chan_weights = [1, 1, p_weight]`.
