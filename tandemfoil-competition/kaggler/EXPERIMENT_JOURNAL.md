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
