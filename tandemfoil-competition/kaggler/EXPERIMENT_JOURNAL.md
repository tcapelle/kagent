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

### 2026-04-27 — iter2: warm-start + point subsampling (40k) + low LR
- **Hypothesis:** W&B inspection of competitor configs showed the leaders (thorfinn, edward, frieren) all use `train_subsample=40000` with `batch_size=4`. Subsampling cuts per-epoch wall-clock ~3x → fits more epochs in the 30 min budget. Warm-start from iter1 with much lower LR (5e-5, matching thorfinn) lets the model fine-tune without overshooting.
- **Change:** `train.py`: add `train_subsample` (drop random volume nodes per training sample, keep all surface nodes via top-k score trick). bs 2→4, val_batch_size kept at 2 since validation runs on full meshes. p_weight 1→4 (extra surface-pressure emphasis). lr 8e-4→5e-5 (after killing a first attempt at 2e-4 that diverged from warm-start: train surf loss exploded 0.35 → 0.78 in epoch 1). epochs 50→15 so cosine actually anneals.
- **Result:** Warm-start from iter1's `checkpoints/best.pt` succeeded (no missing keys). 14 epochs in 22 min (~82 s/epoch). best `val/avg_surf_p=114.43` at epoch 14. Trajectory monotonic-ish: 138.91 → 136.5 → 133.8 → 135.9 → 128.1 → 128.2 → 122.7 → 121.9 → 119.6 → 120.5 → 116.4 → 115.4 → 115.3 → 114.4. Predictions at `askeladd/3cd9ef3`. W&B: askeladd/iter2-sub40k-bs4-warm-pw4-lr5e5.
- **Verdict:** kept (-22 surf_p vs iter1). Still 2x worse than thorfinn's 54 — gap suggests they have a much better warm-start chain or a smarter loss.
- **Notes:** First attempt at lr=2e-4 destabilised the warm-started weights immediately (epoch 1 surf_p=204, train losses doubled). Lesson: when warm-starting from a converged checkpoint, drop LR by 10-100x. Inspecting thorfinn's W&B config revealed the gap — they use 4-weight loss `(surf_p_w=6, surf_uv_w=1, vol_p_w=0.5, vol_uv_w=0.5)` instead of my `surf_weight=15` × `chan_w=[1,1,4]`. iter3 will adopt that scheme.

### 2026-04-27 — iter1: bigger Transolver + L1 loss + bf16
- **Hypothesis:** A bigger Transolver (256/6/128 vs default 128/5/64) with L1 loss in normalised space (matches the per-channel MAE metric exactly) and surf_weight=15 should land in the top half of the leaderboard. Add bf16 autocast + grad clip for stability and speed.
- **Change:** `train.py`: model 128/5/64 → 256/6/128 (3M params), loss MSE → L1 in normalised space, surf_weight 10 → 15, lr 5e-4 → 8e-4, batch_size 4 → 2 (Cruise samples are big), bf16 autocast, grad_clip=1.0, save best by `val/avg_surf_p` (the leaderboard metric), mirror checkpoints to PVC + `checkpoints/best.pt`. Factored model to `model.py`. Fixed `predict.py` (removed `NotImplementedError`, load Transolver from `model.py`, bf16 inference).
- **Result:** 6 epochs in 30 min (timeout). best `val/avg_surf_p=136.76` at epoch 6. Trajectory: 214 → 181 → 167 → 150 → 142 → 137 → 144. Peak VRAM 56 GB. W&B: askeladd/iter1-256x6-L1-bf16-sw15.
- **Verdict:** kept. Predictions saved at `askeladd/634f51a`. Improvement still flat at end → more epochs would help; should warm-start.
- **Notes:** auto-submit subprocess crashed because `predict.py` did `from train import Transolver` (which executed train.py at import time and parsed conflicting CLI args). Fixed in this commit. Cosine T_max=50 is wrong since we only do 6 epochs — LR stays near peak. Iter2: lower LR and warm-start, set epochs to ~8 so cosine actually anneals. Add per-channel weight on surface pressure (the leaderboard metric) — `chan_weights = [1, 1, p_weight]`.
