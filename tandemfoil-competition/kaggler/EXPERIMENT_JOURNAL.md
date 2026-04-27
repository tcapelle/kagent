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

### 2026-04-27 — iter2: bf16 + 256/6 model with warmup (discarded)
- **Hypothesis:** bigger model (256-d, 6 layers) + bf16 mixed precision should fit in 30 min and beat iter1.
- **Change:** `train.py` model_config `n_hidden=192→256, n_layers=5→6`; `surf_weight=20→30`; `lr=5e-4→8e-4`; added 200-step warmup + cosine; bf16 autocast; grad clip 1.0.
- **Result:** 8 epochs in 32 min, best at epoch 7 with val/loss=10.97 (vs iter1's 7.71). bf16 only saved ~10% time (240s vs 261s/epoch); peak GPU 89.5GB (vs iter1's 82.9). Auto-submit predict.py OOM'd because train.py held GPU memory.
- **Verdict:** discarded — bigger model converged slower; given the 30-min cap, iter1's smaller model finishes more useful epochs.
- **Notes:** warmup hurts when only 7-8 epochs fit total; surf_weight=30 didn't obviously help. **Lesson:** time-bounded runs reward fast convergence, not capacity. Need either (a) faster training (mesh subsampling) or (b) skip warmup entirely. Also need to free GPU memory between train + predict (or run predict separately).

### 2026-04-27 — iter1: bigger Transolver baseline (kept)
- **Hypothesis:** the template defaults (n_hidden=128, n_head=4, mlp_ratio=2) underfit; a larger Transolver with stronger surface weighting should make the leaderboard.
- **Change:** `train.py` model_config → `n_hidden=192, n_head=8, mlp_ratio=4` (slice_num=64, n_layers=5 unchanged); `surf_weight: 10→20`; `epochs: 50→12`. Refactored Transolver into `models.py` so `predict.py` can import without re-running train's argparse.
- **Result:** 7 epochs in 30 min (timeout). Best val/loss=7.71 at epoch 7. Peak GPU 82.9GB. Test scores: avg_surf_p=136.23 (single=163.65, geom_rc=147.93, cruise=94.92, re_rand=138.44). On the leaderboard at #1 (only other entry is nezuko at 350).
- **Verdict:** kept — first working submission, but well behind apr27's frieren@42.11. Need substantially more capacity/training to compete.
- **Notes:** training is FP32 → memory-bound at this size. Surface losses still decreasing at timeout, model under-trained. Cosine `T_max=epochs` never finishes because we time out early; could set `T_max≈actual_epochs`.
