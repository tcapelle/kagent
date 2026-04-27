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

### 2026-04-27 — h128-l6 huber + AMP + channel-weighted loss (iter 2)
- **Hypothesis:** Smaller, faster-converging Transolver (n_hidden=128, n_layers=6, slice_num=32, mlp_ratio=4) with bf16 AMP, Huber loss, channel weight 2x on pressure, surf_weight=20 should outperform iter 1.
- **Change:** train.py — smaller model (1.16M params), Huber loss with channel weights, lr=1e-3 + 1-epoch warmup + cosine; predict.py wired up via shared model.py.
- **Result:** 15 epochs in 30 min, best val avg_surf_p=105.85 (epoch 14). Test avg_surf_p=94.66 — still worse than baseline 79.12. Trajectory still descending at end of budget.
- **Verdict:** Discarded — undertrained. Best path forward: subsample volume so we get more epochs.
- **Notes:** With AMP and bs=4, ran 15 epochs vs 8 in iter 1. Cruise track always best (val_geom_c=82); single_in_dist worst (val_single=119). Train loss still falling (0.166 → 0.035), so model isn't saturated. Next: surface-aware subsampling (keep all surface, subsample volume to 12K) for 5–10x speedup per batch.

### 2026-04-27 — h192-l8 channel-weighted + warmup (iter 1)
- **Hypothesis:** Bigger Transolver (n_hidden=192, n_layers=8) with channel-weighted MSE loss (3x on pressure), surf_weight=25, LR warmup, gradient clipping should beat baseline 79.12 by exploiting more capacity and better loss alignment.
- **Change:** train.py — bigger model (2.23M params), channel-weighted MSE, surf_weight=25, lr=1e-3 with 2-epoch warmup; predict.py wired up to load Transolver from yaml config; checkpoint selection by avg_surf_p instead of val/loss.
- **Result:** 8 epochs in 30 min (each ~3.8 min), best val avg_surf_p=132.18 (epoch 7). Test avg_surf_p=incomplete-but-likely-worse than baseline.
- **Verdict:** Discarded — too slow per epoch, undertrained.
- **Notes:** Bigger model needs more time per epoch (3.8 min) so only completes 8 epochs in budget. Validation trajectory still trending down. Trying smaller faster-converging model in iter 2 with same loss structure.
