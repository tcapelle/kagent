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

### 2026-04-27 — iter1 SmoothL1 surface loss (DISCARDED)
- **Hypothesis:** Switching surface loss from MSE to SmoothL1 (β=1.0) would align training with the leaderboard's MAE metric and improve surface pressure accuracy.
- **Change:** train.py — Transolver d192/L6/s64 (n_head=6, mlp_ratio=2). Surface loss = SmoothL1(diff, β=1.0); volume loss = MSE. Selected best by `avg_mae_surf_p` (mean across 4 val splits) instead of `val/loss`. Refactored model classes into `model.py`. Added PVC checkpoint mirror.
- **Result:** 8 epochs in 32.1 min; train: vol=0.44 surf=0.09. Best epoch 7, **avg_mae_surf_p=137.7** — 3× worse than prior thorfinn baseline (42.90 with MSE+MSE+surf_weight=10). Auto-submit OOM'd because train.py kept the model on GPU when spawning predict.py subprocess.
- **Verdict:** Discarded (`git reset --hard HEAD~1`).
- **Notes:** SmoothL1 with β=1.0 has gradient |d| (≤1) inside the quadratic region — half of MSE's 2|d|. With surf_weight=10 unchanged, the *effective* surface gradient was halved, so training under-weighted the surface and the model learned a worse pressure field. To use SmoothL1 productively, double surf_weight or use β<1.0. Filed for later: also fix the auto-submit OOM by deleting the model and emptying CUDA cache before spawning predict.py.
