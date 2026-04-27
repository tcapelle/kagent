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

### 2026-04-27 — iter1: bigger Transolver + L1 loss + bf16
- **Hypothesis:** A bigger Transolver (256/6/128 vs default 128/5/64) with L1 loss in normalised space (matches the per-channel MAE metric exactly) and surf_weight=15 should land in the top half of the leaderboard. Add bf16 autocast + grad clip for stability and speed.
- **Change:** `train.py`: model 128/5/64 → 256/6/128 (3M params), loss MSE → L1 in normalised space, surf_weight 10 → 15, lr 5e-4 → 8e-4, batch_size 4 → 2 (Cruise samples are big), bf16 autocast, grad_clip=1.0, save best by `val/avg_surf_p` (the leaderboard metric), mirror checkpoints to PVC + `checkpoints/best.pt`. Factored model to `model.py`. Fixed `predict.py` (removed `NotImplementedError`, load Transolver from `model.py`, bf16 inference).
- **Result:** 6 epochs in 30 min (timeout). best `val/avg_surf_p=136.76` at epoch 6. Trajectory: 214 → 181 → 167 → 150 → 142 → 137 → 144. Peak VRAM 56 GB. W&B: askeladd/iter1-256x6-L1-bf16-sw15.
- **Verdict:** kept. Predictions saved at `askeladd/634f51a`. Improvement still flat at end → more epochs would help; should warm-start.
- **Notes:** auto-submit subprocess crashed because `predict.py` did `from train import Transolver` (which executed train.py at import time and parsed conflicting CLI args). Fixed in this commit. Cosine T_max=50 is wrong since we only do 6 epochs — LR stays near peak. Iter2: lower LR and warm-start, set epochs to ~8 so cosine actually anneals. Add per-channel weight on surface pressure (the leaderboard metric) — `chan_weights = [1, 1, p_weight]`.
