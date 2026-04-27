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

### 2026-04-27 — iter2: Fourier positional encoding + lr 5e-4
- **Hypothesis:** Adding sinusoidal positional encoding on (x, z) coordinates (10 log-spaced freqs up to 16 cycles/unit) gives the MLP preprocess access to high-frequency spatial information that helps with turbulence. Lower lr=5e-4 + drop p_channel_weight should remove iter1 instability.
- **Change:** models.py — added `n_pos_freqs`, `max_pos_freq` to Transolver; appends sin/cos features of (x,z) before preprocess MLP. train.py — n_pos_freqs=10, max_pos_freq=16, lr=1e-3→5e-4, p_channel_weight=2→1, epochs=12→15.
- **Result:** Trained 8 epochs, best val/loss=5.07 (down from iter1 6.02). Per-split val mae_surf_p: single=141.2, geom_rc=142.3, geom_cruise=83.8, re_rand=112.6 (avg=120.0). Test leaderboard: 105.58 avg surf_p (rank 4/5). W&B run k380h8ov.
- **Verdict:** kept (small but real improvement). Still far behind leaders (edward 43.75, alphonse 50.83 on test).
- **Notes:** train_surf still falling at epoch 8 (0.41→0.087) — model is undertrained. Surface mae_p plateaued ~120 across iters. Next: subsample volume points to allow 2× more epochs in 30-min budget while keeping all surface points.

### 2026-04-27 — iter1: bigger Transolver + bf16 + p-weighted surface loss
- **Hypothesis:** Scaling Transolver (192 hidden, 6 layers, 8 heads, slice_num=96), training in bf16 for speed, surf_weight=20, and weighting pressure 2× in surface MSE (since the leaderboard ranks by surface pressure MAE) will beat prior tanjiro (51.42 avg surf_p on apr27).
- **Change:** train.py — n_hidden 128→192, n_layers 5→6, n_head 4→8, slice_num 64→96, lr 5e-4→1e-3, surf_weight 10→20, added per-channel weighted surface MSE (pressure ×2), bf16 autocast in train+val, grad_clip=1, epochs=12. predict.py rewritten to load Transolver from new models.py and apply hard no-slip on surface velocity. predict.py auto-submit had imported train.py and triggered double CLI parsing → moved model classes into `models.py`.
- **Result:** Trained 8 epochs in 30 min (timed out). Best epoch 8: val/loss=6.02. Per-split val mae_surf_p: single=145.8, geom_rc=137.1, geom_cruise=90.2, re_rand=122.2 (avg=123.8). VRAM peak 84.6GB (within budget). W&B run zbie1byp.
- **Verdict:** kept (predictions submitted to apr27-4/tanjiro/55efe74). But val mae_surf_p (~124) is much worse than prev tanjiro's TEST mae_surf_p (~51) — the new run is undertrained / worse-tuned. Suspect bf16 inference precision or aggressive lr=1e-3 hurt convergence.
- **Notes:** Strong vol_loss decrease (0.73→0.20) but surf_loss plateaus around 0.13. Pattern suggests model is fitting field but not pressure peaks on surface. Next: try fp32, lr 5e-4, longer effective training.
