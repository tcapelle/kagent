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

### 2026-04-27 — v1 Transolver-256x8 + sub32K + p-weighted MSE + surf_w=20
- **Hypothesis:** Larger Transolver (256/8/8/64) than the 128/5/4/64 default + node subsampling for speed
  + extra weight on the pressure channel (y_std≈679 vs 22/10) + surf_weight=20 should beat baseline,
  matching the parameter regime of apr27 leaders (frieren 256/8 → 42.11 surf-p MAE).
- **Change:** train.py — `n_hidden=256, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2`,
  bf16 autocast, channel weights `[1, 1, 3]` on normalized MSE, `surf_weight=20`,
  cosine LR from `7e-4`, grad-clip 1.0, train_subsample 32 768 nodes/sample with 50 % surface oversampling
  (full mesh at validation). predict.py reads `model_config` from the checkpoint payload.
  Also wrapped the training entry-point in `main()`/`if __name__ == "__main__"` so predict can import Transolver
  cleanly. Whitelisted `tandemfoil-competition/kaggler/checkpoints/best.pt` in the root .gitignore.
- **Result:** wandb run `go992jul` (kagent-tandemfoil5). 29 epochs in 30.8 min, peak 13.8 GB.
  Best at epoch 25, val/loss = 5.84 (combined). Per-split val/loss at best:
  single_in_dist 5.21 · geom_rc 9.81 · geom_cruise 2.50 · re_rand 5.84.
  Test predictions saved to `/mnt/new-pvc/predictions/apr27-5/alphonse/5f51a09/` (4 splits × 200 samples).
  Scoring still pending at journal-write time.
- **Verdict:** kept — first credible submission; fast (≈63 s/epoch) and well below the 96 GB VRAM budget,
  so plenty of headroom for the next iteration.
- **Notes:** loss kept descending into epoch 29; cosine LR likely under-utilised (didn't fully decay).
  geom_camber_rc remained the hardest split (val/loss 9.8). Surf MSE was still ~0.34 (normalized) —
  big room for improvement on the leaderboard metric (surf p MAE in physical units).
  Next ideas: (a) longer effective training via bigger batch / higher slice_num; (b) Re-aware decoder
  conditioning to help the OOD-Re split; (c) Fourier features on (x,z) for high-frequency turbulence.
