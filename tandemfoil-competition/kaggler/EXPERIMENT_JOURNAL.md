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

### 2026-04-23 — v5 add EMA + sub30k + warmup1000
- **Hypothesis:** EMA of weights is thorfinn iter2's big win (97→78 on val). Add it + thorfinn iter2 tweaks (sub30k, warmup 1000, lr 5e-4).
- **Change:** `EMA(decay=0.999)` class; swap to EMA weights before validation and save EMA state; restore live weights for next epoch. train_subsample_n 40k→30k, warmup 500→1000, lr 7e-4→5e-4. Everything else from v4.
- **Result:** best val avg_surf_p=96.13 at epoch 43, val/loss=4.95 (improvement from v4's 112.71 / 6.46). Reached 43 epochs in 30 min. Commit e2f2217.
- **Verdict:** kept — biggest single-run win so far, saves predictions automatically.
- **Notes:**
  - Early epochs (1-5) look terrible because EMA weights start from random init, so shadow ≈ random. By epoch 20+ EMA is well-calibrated and starts winning over live weights.
  - `val_geom_camber_rc` remains the hardest split (8.66 vs 2.4-5.8 others). Geometry-interpolation hard.
  - Next: ensemble v5 with a diverse-architecture run (128h/5L) — frieren did exactly this and won #1.

### 2026-04-23 — v4 thorfinn-style (192h/6L bf16 + p-weighted surf loss)
- **Hypothesis:** match thorfinn iter1 config (bf16 + warmup-cosine-by-steps + grad_clip + bigger model + subsample 40k) and add a 3× channel weight on surface pressure since the leaderboard metric is avg_surf_p.
- **Change:** n_hidden 128→192, n_layers 5→6, n_head 4→8, mlp_ratio 2→4, lr 5e-4→7e-4, wd 1e-4→1e-5, betas=(0.9,0.95), surf_weight 10→20, added `surf_p_weight=3`, warmup 500 + cosine by global step, grad_clip=1.0, bf16 autocast, subsample 40k, checkpoint selection by avg mae_surf_p across splits.
- **Result:** 2.59M params, 52s/epoch at 9.4 it/s, reached epoch 35 in 30.4 min. Best (EMA off) avg_surf_p=112.71 on val. val/loss=6.46 (worse than baseline 3.10 because scale of weighted loss changed — not directly comparable). Commit e4beb2d.
- **Verdict:** kept on leaderboard (supersedes v3, matches baseline at ~112). Not enough improvement to catch thorfinn (77.98).
- **Notes:**
  - val_geom_camber_rc is the hardest split (loss ~10 vs ~3-7 for others). Geometry-interpolation hardness.
  - Thorfinn iter1→iter2 improved 97→78 (on val) largely thanks to **EMA + smaller subsample (30k) + lower lr + longer warmup + surf_p_weight=2**. Adding EMA is the biggest known win; it's next.

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
