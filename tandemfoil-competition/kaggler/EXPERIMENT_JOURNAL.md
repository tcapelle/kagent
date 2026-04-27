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

### 2026-04-27 — v3 warm-start from askeladd's apr27-5 leader + L1 + best-by-surf-p
- **Hypothesis:** the apr27-5 leader askeladd (test 51.22) reached val=54.79 by warm-starting
  from thorfinn's apr27-bis ckpt (192/6/6/128, fun_dim=24/space_dim=0). Continuing the same recipe
  *from askeladd's own checkpoint* should push val below 54 in 30 minutes — beating askeladd's test.
- **Change:** train.py — switch architecture to 192/6/6/128 fun_dim=24 space_dim=0 to match the warm-start
  shape; add `--warm_start <path>` (loads state_dict with `strict=False`); switch loss to L1
  (matches the leaderboard MAE metric); pick best checkpoint by `val/avg_surf_p` (the actual leaderboard
  metric) instead of combined val/loss; LR 5e-5, surf_weight=10, p_weight=3, train_subsample=40 000;
  print surf_p MAEs in the per-epoch summary so we can read the leaderboard signal directly.
- **Result:** wandb run, 29 epochs in 30.3 min from `model-rriy9vrf` warm-start.
  Best at epoch 26 → **val/avg_surf_p = 51.85** (single 44.70 / geom_rc 71.84 / cruise 36.00 / re_rand 54.87)
  vs askeladd's val=54.79 (single 48.29 / geom_rc 74.71 / cruise 38.11 / re_rand 58.05) — improved on
  every split. If the askeladd val→test ratio (0.935) holds, my test should land near ~48.5.
  Predictions saved to `/mnt/new-pvc/predictions/apr27-5/alphonse/ade83e9/`.
- **Verdict:** kept — clear improvement on every val split; lowest val/avg_surf_p I have measured.
- **Notes:** v2 (Fourier-8 features + p_weight=10 from-scratch) was started after v1 but killed at
  epoch 2 once I read askeladd's transcript and saw the warm-start recipe — I expected the warm-start
  to dominate any from-scratch architectural tweak in the 30 min budget. Trajectory was still descending
  at the timeout — chain-training from this checkpoint with another 30 min should help. The hardest
  split remains `geom_camber_rc` at 71.84; cruise is now small (36) and likely close to its floor.

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
