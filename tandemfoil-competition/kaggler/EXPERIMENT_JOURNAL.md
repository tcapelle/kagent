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

### 2026-04-23 — v3 node-subsampled baseline
- **Hypothesis:** cap nodes per training sample at 50k (keep all surface) to speed each step so we can run more epochs in the 30-min budget.
- **Change:** added `SubsampledDataset` wrapper; cfg `train_subsample_n=50000`; `epochs=18` so cosine LR finishes inside budget. Same 128h/5L architecture.
- **Result:** val/loss=3.99 at epoch 18, 12.3 min, 8.8GB peak. avg_surf_p=130.30 (vs v2 151.05, baseline incomplete). Commit 7818971.
- **Verdict:** kept as best-so-far on leaderboard metric. But val/loss regressed vs baseline (3.10) — subsampling shifts the volume-to-surface ratio seen during training, which likely overweights surface relative to what the loss expects.
- **Notes:**
  - Throughput jumped 4× (11.6 it/s vs 3 it/s baseline). Training finished in 12 min — 18 min of budget wasted.
  - Surface nodes in data (is_surface=True) do NOT actually have zero velocity (median mag=6 m/s, max=14 m/s) — the no-slip hard-projection idea would be wrong.
  - Leaderboard metric is **avg surface pressure MAE** (thorfinn leads at 77.98). Focus next iteration on pressure-channel surface accuracy specifically.
  - Inspected thorfinn's 0601ec5 commit: 192x6/mlp_ratio=4, bf16 AMP, warmup+cosine-by-steps, grad_clip=1, surf_weight=20, lr=7e-4, betas=(0.9,0.95), subsample 40k. That config is next.

### 2026-04-23 — v2 scale-up 192h/6L
- **Hypothesis:** bigger model should fit more capacity, beating the 128h baseline.
- **Change:** n_hidden 128→192, n_layers 5→6, slice_num 64→96, n_head 4→6, epochs 50→14.
- **Result:** val/loss=4.55 at epoch 5 (best); 32.4 min; 83.8 GB peak; only 7 epochs finished. avg_surf_p=151.05 (WORSE than baseline-era leaders). Commit cabd7f5.
- **Verdict:** discarded (but predictions still on leaderboard) — code already reverted to 128h/5L.
- **Notes:** 277s/epoch vs 130s/epoch baseline = 2.1× slower. Model was learning faster per epoch (epoch 5 already at 4.55) but didn't converge. Lesson: scaling up WITHOUT a speedup (bf16/subsample) busts the time budget. V4 combines both.

### 2026-04-23 — baseline Transolver 128h/5L
- **Hypothesis:** establish a floor with the template model.
- **Change:** wired predict.py, guarded train.py under `main()`. Default config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4, epochs=50, surf_weight=10.
- **Result:** val/loss=3.1000 at epoch 14 (30.7 min timeout). Splits: in_dist=5.10, camber_rc=3.54, camber_cruise=1.46, re_rand=2.30. Peak 42GB. Commit f65f434.
- **Verdict:** kept — first submission on leaderboard. Predictions at `/mnt/new-pvc/predictions/apr23/tanjiro/f65f434/`.
- **Notes:**
  - `val_single_in_dist` is the **worst** split (5.10) despite being the "easy" sanity check — likely because raceCar single has wide y ranges and surf_weight=10 amplifies surface errors.
  - Cosine LR scheduler was set for 50 epochs but training only reached 14 → LR never decayed near zero. Next run should set `epochs` to match the 30min budget so LR annealing finishes.
  - Predict auto-submit OOM'd because train process still held 93GB VRAM; fixed by freeing model + `empty_cache()` before subprocess (commit incoming).
