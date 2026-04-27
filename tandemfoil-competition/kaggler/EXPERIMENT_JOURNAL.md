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

### 2026-04-27 — iter1: bigger Transolver baseline (kept)
- **Hypothesis:** the template defaults (n_hidden=128, n_head=4, mlp_ratio=2) underfit; a larger Transolver with stronger surface weighting should make the leaderboard.
- **Change:** `train.py` model_config → `n_hidden=192, n_head=8, mlp_ratio=4` (slice_num=64, n_layers=5 unchanged); `surf_weight: 10→20`; `epochs: 50→12`. Refactored Transolver into `models.py` so `predict.py` can import without re-running train's argparse.
- **Result:** 7 epochs in 30 min (timeout). Best val/loss=7.71 at epoch 7. Peak GPU 82.9GB. Test scores: avg_surf_p=136.23 (single=163.65, geom_rc=147.93, cruise=94.92, re_rand=138.44). On the leaderboard at #1 (only other entry is nezuko at 350).
- **Verdict:** kept — first working submission, but well behind apr27's frieren@42.11. Need substantially more capacity/training to compete.
- **Notes:** training is FP32 → memory-bound at this size. Surface losses still decreasing at timeout, model under-trained. Cosine `T_max=epochs` never finishes because we time out early; could set `T_max≈actual_epochs`.
