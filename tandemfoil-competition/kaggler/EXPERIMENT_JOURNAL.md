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

### 2026-04-23 — iter5: ensemble iter3+iter4 (equal weights)
- **Hypothesis:** Simple average of iter3 (82.24 surf_p) and iter4 (78.73) predictions. Typically ensembling two decorrelated strong models improves by 1-3%. iter3 is 128×5 and iter4 is 192×6 so architectural diversity is real.
- **Change:** Added `ensemble.py` that averages per-sample predictions from given commits, writes to a new dir keyed on current HEAD. Commit `25ebb0c`, no training.
- **Result:** scorer still marking "incomplete" at time of writing; will update.
- **Verdict:** pending.

### 2026-04-23 — iter4: 192x6 + 3-ep warmup + 35 epochs
- **Hypothesis:** iter3's 128x5 plateaued around 1.9 val/loss. A larger 192×6 model with proper warmup (iter2's failure was a cold start with aggressive cosine) should unlock more capacity without the convergence issues. Thorfinn uses 192x6 and is leading on test.
- **Change:** `train.py` n_hidden=192, n_layers=6, n_head=6, slice_num=64 (matches thorfinn's rumored config). Added `SequentialLR(LinearLR warmup 3 epochs + CosineAnnealingLR)`. `epochs=35` (46s/ep budget, ~27 min). Reverted iter2's Fourier features in `model.py`. Commit `1509e10`, run `29qm6q2b`.
- **Result:** best val/loss **1.9102** at epoch 31 (35/35 epochs, 26.7 min, 20.8 GB). Per-split at best: single_in_dist=2.05, geom_camber_rc=2.73, geom_camber_cruise=0.98, re_rand=1.88. Marginal ~0.6% improvement over iter3's 1.9212.
- **Verdict:** kept. Predictions at `/mnt/new-pvc/predictions/apr23/frieren/1509e10/`. Marginal val gain — will watch test scoring to confirm it beats iter3 on the leaderboard metric.
- **Notes:** Trained slower than iter3 per-epoch (46s vs 27s) so fewer epochs. Warmup worked as intended — epoch 1 val is high (20.7) because LR is still 0, then bounces back. Both iter3 and iter4 seem to hit ~1.9 wall; future gains likely need: (a) ensemble both checkpoints, (b) longer training (reduce val cadence), (c) different loss (L1/Huber), (d) residual prediction with AoA-derived prior.

### 2026-04-23 — iter3: iter1 arch + 50 epochs + grad_clip=1.0
- **Hypothesis:** iter1 was clearly still improving at epoch 25 (val/loss still decreasing under cosine schedule). Doubling epochs with the same 128×5 arch + adding grad_clip should let cosine tail squeeze out another 1.0+ val/loss. No architectural change so risk is low.
- **Change:** `train.py` `epochs=50`, added `cfg.grad_clip=1.0` + `clip_grad_norm_` call. Reverted iter2's Fourier+bigger-model changes (model.py back to original). Commit `2c929ae`, run `j6880sdl`.
- **Result:** best val/loss **1.9212** at epoch 40 (50/50 epochs, 22.3 min, 11.8 GB VRAM). Per-split at best: single_in_dist=1.87, geom_camber_rc=2.78, geom_camber_cruise=0.91, re_rand=2.13. **~34% improvement over iter1's 2.90.**
- **Verdict:** kept. Mirrors checkpoint to `checkpoints/best.pt`; predictions at `/mnt/new-pvc/predictions/apr23/frieren/2c929ae/`.
- **Notes:** Training was noisy epoch-to-epoch (spikes of ~0.5 val/loss) but trended steadily down until ~epoch 40, then plateaued. Cosine over 50 vs 25 epochs is much gentler — that's most of the win. Next targets: thorfinn at 77.98 avg_surf_p (ours pre-iter3 was 109.27). Could push with (a) even more epochs if we squeeze training, (b) longer model (warmup helps bigger nets), or (c) smarter loss (L1 on surface pressure directly).

### 2026-04-23 — iter2: Fourier position features + larger model (192x6, slice 64) — DISCARDED
- **Hypothesis:** Add Gaussian Fourier features on (x, z) position (32 freqs, sigma=2) + bump model to 192x6 with 6 heads and mlp_ratio=2. Should capture higher-frequency turbulent details and give more capacity.
- **Change:** `model.py` added `GaussianFourierFeatures` + wired into Transolver's preprocess (concat 2·N_freqs features onto input). `train.py` bumped n_hidden=192, n_layers=6, n_head=6, fourier_pos=32, epochs=20. Run `96rcbcl8`, commit `0f29c86`.
- **Result:** best val/loss 3.56 at epoch 20 (20/20 epochs, 15.3 min, 46s/ep, 21GB). Worse than iter1 (2.90). Test scores also worse: avg_surf_p 124.12 vs iter1's 109.27.
- **Verdict:** discarded — `git reset --hard HEAD~1`. The bigger model needed warmup it never got (cosine decayed too aggressively over 20 epochs); Fourier @ sigma=2.0 likely also added noise the model had to fight.
- **Notes:** Losses caught up around epoch 12 but never reached iter1's best. thorfinn's 192x6 (no Fourier, probably more epochs) achieved 87.51 surf_p and topped the board. Takeaway for iter3: either (a) stay with iter1 arch and train 2× longer, or (b) try 192x6 without Fourier. Also thorfinn submitted AFTER me so they have a stronger final config.

### 2026-04-23 — iter1: bf16 autocast + point subsampling + bs8
- **Hypothesis:** bf16 + subsampling 40k volume nodes per train sample gives ~4x speedup, unlocking more epochs in the 30-min budget without sacrificing quality (surface nodes are always kept so surf_loss is unaffected).
- **Change:** `train.py` bf16 autocast in forward/val, custom `subsample_collate` (keeps all surface + 40k random volume nodes), `batch_size=8`, `epochs=25`. Baseline Transolver (128×5, slice_num=64) unchanged. Also refactor: extracted model into `model.py` so `predict.py` can import without triggering training CLI.
- **Result:** best val/loss 2.90 at epoch 25 (25/25 epochs, 11.2 min train, 188 steps/epoch at ~9.5 it/s). VRAM peak 11.8 GB. Per-split val/loss at best: single_in_dist=3.01, geom_camber_rc=4.15, geom_camber_cruise=1.68, re_rand=2.76. Commit `7f63057`. Run `67zv1c0j`.
- **Verdict:** kept. First real submission (leaderboard was empty pre-submit).
- **Notes:** Loss noisy epoch-to-epoch due to stochastic subsampling, but cosine schedule pushed monotonic improvement over the last 5 epochs. `geom_camber_rc` (unseen raceCar camber) is by far the hardest split. Next ideas: Fourier position features, larger model (192×6), higher slice_num, possibly residual prediction from AoA/Re free-stream prior.
