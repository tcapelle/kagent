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

### 2026-04-23 — iter1: bf16 autocast + point subsampling + bs8
- **Hypothesis:** bf16 + subsampling 40k volume nodes per train sample gives ~4x speedup, unlocking more epochs in the 30-min budget without sacrificing quality (surface nodes are always kept so surf_loss is unaffected).
- **Change:** `train.py` bf16 autocast in forward/val, custom `subsample_collate` (keeps all surface + 40k random volume nodes), `batch_size=8`, `epochs=25`. Baseline Transolver (128×5, slice_num=64) unchanged. Also refactor: extracted model into `model.py` so `predict.py` can import without triggering training CLI.
- **Result:** best val/loss 2.90 at epoch 25 (25/25 epochs, 11.2 min train, 188 steps/epoch at ~9.5 it/s). VRAM peak 11.8 GB. Per-split val/loss at best: single_in_dist=3.01, geom_camber_rc=4.15, geom_camber_cruise=1.68, re_rand=2.76. Commit `7f63057`. Run `67zv1c0j`.
- **Verdict:** kept. First real submission (leaderboard was empty pre-submit).
- **Notes:** Loss noisy epoch-to-epoch due to stochastic subsampling, but cosine schedule pushed monotonic improvement over the last 5 epochs. `geom_camber_rc` (unseen raceCar camber) is by far the hardest split. Next ideas: Fourier position features, larger model (192×6), higher slice_num, possibly residual prediction from AoA/Re free-stream prior.
