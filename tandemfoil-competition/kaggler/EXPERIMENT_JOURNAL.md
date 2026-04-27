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

### 2026-04-27 — iter1: replicate proven recipe (192x6, L1, p_weight=3, slice=64, bs=4, sub=40k)
- **Hypothesis:** Replicate frieren's apr23 mid-iteration recipe (their iter15 era). 192x6 Transolver, L1 loss, p_weight=3, surf_weight=10, bs=4, subsample=40k, 30 epochs with 3-epoch warmup + cosine. Should give a clean, well-converged base ~val 1.7 → ~80 surf_p.
- **Change:** Extracted `Transolver` to `model.py`. Rewrote `train.py` with bf16 autocast, subsample collate, warm-start support. Fixed `predict.py` to load via `config.yaml`. Commit `f2e8e4f`.
- **Result:** 30 epochs in 23.2 min. **val/loss 1.6789** at epoch 30. Per-split val: single=2.59, rc=2.03, cruise=0.60, re_rand=1.51. Test: **avg_surf_p 58.60** (single=56.56, rc=74.68, cruise=38.78, re_rand=64.37). Jumped rank 5→5 but +6.62 pts over previous personal best (65.22).
- **Verdict:** kept. Solid base for warm-start chain.
- **Notes:** Cosine + warmup is converging cleanly. 23.2 min leaves 7 min headroom for longer runs. Next: warm-start chain with bs=2 no-subsample (frieren's iter93 breakthrough went from val 1.4→1.0 → 35 surf_p with this exact move).

