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

