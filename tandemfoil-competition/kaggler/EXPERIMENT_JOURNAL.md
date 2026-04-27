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

### 2026-04-27 — iter10: chain LR 3e-6 + p_weight 30, plus SWA helper for next iter
- **Hypothesis:** continue chain ramp; gains shrinking but still net positive.
- **Change:** args only for iter10. After completion, added `--save_last_k` arg to train.py so iter11 can snapshot the last K epochs in-training and SWA them offline.
- **Result:** 28 epochs in 30.9 min. Best epoch **12** → val/avg_surf_p=45.31 (single=39.39, geom_rc=62.74, geom_cruise=31.41, re_rand=47.71). Run 6w1xchzm. Predictions at apr27-5/askeladd/ea77f6b.
- **Verdict:** kept — improvement (45.57 → 45.31, -0.26).
- **Notes:** Best was at *epoch 12*, not the end — model briefly hit a sharp minimum and drifted afterwards. Last 10 epochs all bounced around 45.4–45.6. **In-training SWA over the last K epochs is the right next move.** Negative result: SWA(iter9, iter10) val=45.40 (worse than iter10 alone). Within-run SWA is more promising than across-run SWA because the trajectory near the bottom is sharper.

### 2026-04-27 — iter9: chain at LR 4e-6 + p_weight 26
- **Hypothesis:** keep ramping (p_weight 22→26, LR 5e-6→4e-6, same domain bias 1,3,2). Diminishing returns expected.
- **Change:** args only.
- **Result:** 28 epochs in 30.9 min. Best epoch 23 → val/avg_surf_p=45.57 (single=39.67, geom_rc=63.09, geom_cruise=31.53, re_rand=47.98). Run 880g4ow3. Predictions at apr27-5/askeladd/d7415da.
- **Verdict:** kept — improvement (45.74 → 45.57, -0.17). Per-iter gain shrinking but consistent.
- **Notes:** geom_rc still ~63. Single ~39.7. Plateau is real on this architecture; need a substantively different move (diverse-init model + true ensemble) or accept ~45 floor.

### 2026-04-27 — iter8: stronger domain bias (1,3,2) + p_weight 22 + LR 5e-6
- **Hypothesis:** Iter7 domain bias worked. Push tandem weight harder (1,2.5,1.5 → 1,3,2) and continue chain ramp.
- **Change:** args only.
- **Result:** 28 epochs in 30.9 min. Best epoch 27 → val/avg_surf_p=45.74 (single=39.55, geom_rc=63.20, geom_cruise=31.76, re_rand=48.47). Run vl7ppxcd. Predictions at apr27-5/askeladd/5543a18.
- **Verdict:** kept — improvement (45.95 → 45.74, -0.21).
- **Notes (negative):** Ensemble(iter7,iter8) val=45.81; SWA(iter7,iter8) val=45.81. Both worse than iter8 alone — chained models too correlated, averaging hurts. Plateau is real.

### 2026-04-27 — iter7: domain re-weighting + chain LR 6e-6 + p_weight 18
- **Hypothesis:** geom_camber_rc=64.36 dominated the avg in iter6 (next worst is re_rand=49). Oversample tandem domains to push the model harder on the bottleneck distribution. Add `--domain_weights "1,2.5,1.5"` (single 1×, racecar_tandem 2.5×, cruise 1.5×). Continue chain ramp.
- **Change:** add `--domain_weights` CLI to train.py — multiplies the existing balanced sampler weights by per-domain factors before constructing `WeightedRandomSampler`. No change to data.py.
- **Result:** 28 epochs in 30.9 min. Best epoch 25 → val/avg_surf_p=45.95 (single=39.66, geom_rc=63.59, geom_cruise=31.78, re_rand=48.75). Run z4e67k2u. Predictions at apr27-5/askeladd/8db4853.
- **Verdict:** kept — improvement (46.27 → 45.95, -0.32). Per-split: single +0.37 (worse), geom_rc -0.77, geom_cruise -0.28, re_rand -0.61. Tandem oversampling worked exactly as designed — single regressed slightly, all tandem splits won.
- **Notes:** geom_rc still dominates (63.59). val=45.95 is below thorfinn's apr27-bis test of 45.94 (their val was ~70.5, ratio ~1.5x → my test extrapolated ~30). Try one more chain with even more aggressive tandem weighting (3,2) and p_weight=22.

### 2026-04-27 — iter6: chain at LR 8e-6 + p_weight 15
- **Hypothesis:** keep walking the chain; p_weight 12→15, LR 1e-5→8e-6.
- **Change:** args only.
- **Result:** 28 epochs in 30.9 min. Best epoch 25 → val/avg_surf_p=46.27 (single=39.29, geom_rc=64.36, geom_cruise=32.06, re_rand=49.36). Run pw71sr2w. Predictions at apr27-5/askeladd/1ac7cb9.
- **Verdict:** kept — improvement (47.06 → 46.27, -0.79). Diminishing return per iter (-3.5 → -2.0 → -1.3 → -1.0 → -0.8) but consistent.
- **Notes:** geom_camber_rc=64.36 is now the dominant share of the average (next is re_rand=49.36). Need a way to attack that split specifically — likely oversample racecar-tandem domain or add geometry-augmenting strategy. One more chain (iter7) then change tactic.

### 2026-04-27 — iter5: chain at LR 1e-5 + p_weight 12, fp32 inference
- **Hypothesis:** Chain ramp keeps paying off (iter4: -1.31). p_weight 10→12 + LR 1.5e-5→1e-5 should add another small gain. Also default predict.py to fp32 (small free win — bf16 inference loses ~0.1pt on val/avg_surf_p).
- **Change:** args + `predict.py`/`predict_ensemble.py` default `bf16=False`. New helpers `swa.py` (parameter-space averaging) and `eval_val.py` (val score for any checkpoint or ensemble).
- **Result:** 28 epochs in 30.9 min. Best epoch 26 → val/avg_surf_p=47.06 (single=40.41, geom_rc=65.09, geom_cruise=32.76, re_rand=49.98). Run yd83krcc. Predictions at apr27-5/askeladd/969bc54.
- **Verdict:** kept — improvement (48.06 → 47.06, -1.00).
- **Notes (negative results):** SWA(iter4, iter5) → val 47.51 (worse than iter5 alone). Prediction ensemble (iter4, iter5) → val 47.45 (worse). Iter4+iter5 are too close on the same trajectory; averaging in either space hurts.

### 2026-04-27 — iter4: chain at LR 1.5e-5 + p_weight 10
- **Hypothesis:** chain ramp continues to pay off (3→5: -3.5, 5→8: -2). Push p_weight 8→10 with another small LR drop (2e-5 → 1.5e-5) to keep walking down.
- **Change:** args only. `--warm_start /mnt/new-pvc/kagent/apr27-5/askeladd/checkpoints/model-eck7oggk/checkpoint.pt --skip_warmup --lr 1.5e-5 --epochs 30 --p_weight 10.0`. Also added `predict_ensemble.py` and `eval_val.py` for upcoming ensemble work.
- **Result:** 28 epochs in 30.9 min. Best epoch 22 → val/avg_surf_p=48.06 (single=41.32, geom_rc=66.77, geom_cruise=33.08, re_rand=51.08). Run eo3xc5x1. Predictions at apr27-5/askeladd/f424fe5.
- **Verdict:** kept — improvement (49.37 → 48.06, -1.31). Diminishing returns visible (last 5 epochs all within 0.3 of best).
- **Notes:** Ensemble of all 4 chained checkpoints next; if val drops, submit. Otherwise iter5 = single chain at LR 1e-5 + p_weight 12 or augmentation.

### 2026-04-27 — iter3: chain again, raise p_weight 5→8, LR 3e-5→2e-5
- **Hypothesis:** iter2 trajectory was still descending at timeout, and p_weight 3→5 was a clear net win. Pushing p_weight further (5→8) and dropping LR a notch should keep moving down without destabilising.
- **Change:** args only. `--warm_start /mnt/new-pvc/kagent/apr27-5/askeladd/checkpoints/model-tska1pw8/checkpoint.pt --skip_warmup --lr 2e-5 --epochs 30 --p_weight 8.0`.
- **Result:** 28 epochs in 30.9 min. Best epoch 26 → val/avg_surf_p=49.37 (single=42.31, geom_rc=68.15, geom_cruise=34.30, re_rand=52.72). Run eck7oggk. Predictions at apr27-5/askeladd/cb450df.
- **Verdict:** kept — clear improvement (51.32 → 49.37, -1.95). p_weight ramp continues to help.
- **Notes:** Improvements per iter shrinking (3.5 → 2.0). geom_rc still the worst (68.2). Trajectory near plateau in last 5 epochs (49.7→49.4). Worth one more chain at lower LR plus consider an ensemble of the 3 chained checkpoints (rriy9vrf, tska1pw8, eck7oggk).

### 2026-04-27 — iter2: chain warm-start, raise p_weight 3→5
- **Hypothesis:** iter1 was still descending at timeout (cosine LR ≈0). Restarting with another cosine cycle at slightly lower peak (3e-5) and stronger surface-pressure weight (p_weight 3→5) should keep moving down without destabilising. p_weight is the leaderboard metric multiplier on surface pressure inside the L1 loss.
- **Change:** train.py unchanged, args only. `--warm_start /mnt/new-pvc/kagent/apr27-5/askeladd/checkpoints/model-rriy9vrf/checkpoint.pt --skip_warmup --lr 3e-5 --epochs 30 --p_weight 5.0`.
- **Result:** 28 epochs in 30.9 min. Best epoch 27 → val/avg_surf_p=51.32 (single=44.60, geom_rc=70.56, geom_cruise=35.66, re_rand=54.44). Run tska1pw8. Predictions at apr27-5/askeladd/27474de.
- **Verdict:** kept — clear improvement (54.79 → 51.32, -3.47). p_weight=5 was a net win even though epoch 1 was worse than iter1 final (model needed to readjust to the heavier p emphasis).
- **Notes:** geom_camber_rc still the worst split (70.5). Trajectory was still slowly descending at timeout — another chain at lower LR is worth a try. Could push p_weight further (e.g., 8) since 3→5 helped.

### 2026-04-27 — iter1: warm-start from thorfinn's apr27-bis checkpoint
- **Hypothesis:** thorfinn topped apr27-bis at test=45.94 (val~70.5) using L1 + surf-pressure focus, slice_num=128. Continuing to fine-tune at low LR (5e-5, no warmup) should push val below 60. The pipeline must be reconstructed from scratch since branches reset between runs.
- **Change:** new `model.py` (Transolver split out so predict.py can import); `train.py` rewritten with surf_weight=10, p_weight=3, train_subsample=40000, bf16, L1 loss in normalized space, optional `--warm_start` and `--skip_warmup`. Best by val/avg_surf_p (leaderboard metric). Auto-mirrors checkpoint to PVC and runs predict.py.
- **Result:** 28 epochs in 30.9 min. Best epoch 26 → val/avg_surf_p=54.79 (single=48.29, geom_rc=74.71, geom_cruise=38.11, re_rand=58.05). Run rriy9vrf. Predictions submitted at apr27-5/askeladd/9b41b49.
- **Verdict:** kept — val drops 70 → 54.79 vs apr27-bis baseline (val=70). Test score TBD when leaderboard updates.
- **Notes:** geom_camber_rc is the hardest split (75) and the bottleneck. Trajectory still descending at timeout — chain-train from this checkpoint with another 30 epochs at even lower LR for iter2. Consider adding p-only refinement head or extra slice tokens.

