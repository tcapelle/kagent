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

### 2026-04-27 — iter2 d256/L8/s96 + bf16 (DISCARDED — OOM)
- **Hypothesis:** Bigger Transolver (frieren-like, n_hidden=256, n_layers=8, slice_num=96) + bf16 mixed precision should fit in 30 min and beat the 192/6/64 baseline.
- **Change:** train.py — model d256/L8/s96, autocast(bf16), MSE+MSE, surf_weight=10, select on `avg_mae_surf_p`, mirror best ckpt to PVC, free GPU before auto-submit subprocess.
- **Result:** OOM at epoch 1 step 0. Peak ~94.9 GB (94.97 GB total). Even with bf16 the activations from {B=4, N≈242K, slice_num=96, L=8} blow past the budget.
- **Verdict:** Discarded (`git reset --hard HEAD~1`).
- **Notes:** baseline d192/L6/s64 fp32 was 74 GB peak. Scaling factors that hurt most: slice_num 64→96 (≈1.5×) and L 6→8 (≈1.33×). For iter3, drop slice_num back to 64 and depth back to 6, only widen hidden to 256. Estimated peak: 74×(256/192)²/2 ≈ 66 GB (bf16). Comfortable.

### 2026-04-27 — iter1 SmoothL1 surface loss (DISCARDED)
- **Hypothesis:** Switching surface loss from MSE to SmoothL1 (β=1.0) would align training with the leaderboard's MAE metric and improve surface pressure accuracy.
- **Change:** train.py — Transolver d192/L6/s64 (n_head=6, mlp_ratio=2). Surface loss = SmoothL1(diff, β=1.0); volume loss = MSE. Selected best by `avg_mae_surf_p` (mean across 4 val splits) instead of `val/loss`. Refactored model classes into `model.py`. Added PVC checkpoint mirror.
- **Result:** 8 epochs in 32.1 min; train: vol=0.44 surf=0.09. Best epoch 7, **avg_mae_surf_p=137.7** — 3× worse than prior thorfinn baseline (42.90 with MSE+MSE+surf_weight=10). Auto-submit OOM'd because train.py kept the model on GPU when spawning predict.py subprocess.
- **Verdict:** Discarded (`git reset --hard HEAD~1`).
- **Notes:** SmoothL1 with β=1.0 has gradient |d| (≤1) inside the quadratic region — half of MSE's 2|d|. With surf_weight=10 unchanged, the *effective* surface gradient was halved, so training under-weighted the surface and the model learned a worse pressure field. To use SmoothL1 productively, double surf_weight or use β<1.0. Filed for later: also fix the auto-submit OOM by deleting the model and emptying CUDA cache before spawning predict.py.
