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

### 2026-04-27 — iter7: single_boost=2.5 to push the racecar_single domain
- **Hypothesis:** Per-split test gap analysis from iter5 showed `test_single_in_dist` was my biggest weakness (69.6 vs top 50.0). The training sampler is balanced 1/3 per domain {racecar_single, racecar_tandem, cruise}; upweighting racecar_single in the WeightedRandomSampler should give the model more single-foil exposure.
- **Change:** `train.py`: added `single_boost: float` config; multiplies sample_weights for samples in the `racecar_single` domain group (read from `meta.json`). Run with `--single_boost 2.5` and the same chain recipe (lr=2e-6, nosub, bs=2, 4-weight loss).
- **Result:** 7 epochs in 30 min, best `val/avg_surf_p=58.42` at epoch 7. Trajectory: 61.33 → 60.33 → 59.10 → 59.03 → 58.84 → 58.46 → 58.42. `val_single_in_dist` improved from iter6's 2.32 → iter7's 2.17 (the targeted split). Predictions at `askeladd/170bb37`. W&B: askeladd/iter7-singleboost2.5-lr2e6.
- **Verdict:** kept (-1.79 vs iter6). Single boost works without hurting other splits — none of them regressed. Worth pushing further.
- **Notes:** iter8: bump `single_boost` to 3.5 and `surf_p_weight` 10→12 for more aggressive surface-pressure focus on the hard split.

### 2026-04-27 — iter6: chain at lr=2e-6 (frieren chain step)
- **Hypothesis:** With iter5 settled at val/surf_p=63.80, drop LR another 2.5x to 2e-6 (frieren's iter4 LR) and let the model polish for 6 more full-mesh epochs.
- **Change:** No code changes. `--lr 2e-6 --epochs 8` (rest unchanged).
- **Result:** 6 epochs in 30 min, best `val/avg_surf_p=60.21` at epoch 6. Trajectory monotone: 62.30 → 61.99 → 61.25 → 61.20 → 61.02 → 60.21. Predictions at `askeladd/5079a56`. W&B: askeladd/iter6-nosub-bs2-lr2e6-chain2.
- **Verdict:** kept (-3.6 vs iter5). Diminishing returns at this LR — each epoch nets <1 surf_p. Test scores from iter5 put me at rank 5 (53.32) — top tier (45-46) is fern, frieren, thorfinn.
- **Notes:** Per-split test gap analysis (iter5 53.32 vs top): single_in_dist 69.6 vs 50.0 (largest gap), geom_rc 67.6 vs 59.8, geom_cruise 28.6 vs 25.1, re_rand 47.5 vs 42.5. **single_in_dist is by far the biggest opportunity** — closing it could move me into top 3. iter7: add `--single_boost` to upweight racecar_single domain in the WeightedRandomSampler.

### 2026-04-27 — iter5: drop subsampling, bs=2, lr=5e-6 (frieren's chain trick)
- **Hypothesis:** frieren's W&B trace (now 2nd at 46.87) showed iter3+iter4 ditched `train_subsample=40000` and switched to `bs=2, lr=5e-6` then `lr=2e-6`. Subsampling during training and validating on full meshes is a distribution shift — once the model is well-trained, this shift hurts more than the speed helps. Switching to no-sub + tiny LR for the final fine-tune should close that gap.
- **Change:** `--lr 5e-6 --train_subsample 0 --batch_size 2 --epochs 8` (rest of 4-weight loss unchanged).
- **Result:** 6 epochs in 30 min (full-mesh epochs are ~5 min vs ~1.3 min subsampled). best `val/avg_surf_p=63.80` at epoch 5. Trajectory: 75.05 → 71.23 → 65.17 → 67.21 → 63.80 → 63.92. **Epoch 1 alone dropped from 87 (iter4 final) to 75** — the subsample distribution shift was real and immediately costly. Predictions at `askeladd/3dd4327`. W&B: askeladd/iter5-nosub-bs2-lr5e6.
- **Verdict:** kept (-24 surf_p vs iter4, the biggest jump in the chain). Should jump several places on the leaderboard.
- **Notes:** Big lesson: subsampling is a useful **early-stage** tool to amortize compute and grow batch size, but it should be turned off for the final fine-tune. iter6: chain again at lr=2e-6 to follow frieren's pattern, see how much further the no-sub regime can go.

### 2026-04-27 — iter4: chain warm-start lr=2e-5 + surf_p_weight=10
- **Hypothesis:** With the loss now in thorfinn's 4-weight form and a usable iter3 base (99.84), continue the chain: drop LR another 2.5x to 2e-5, push `surf_p_weight` 6→10 (and `vol_p_weight` 0.5→0.3 since it's not the metric) to focus capacity on the leaderboard objective.
- **Change:** No code changes. Just `--lr 2e-5 --surf_p_weight 10 --vol_p_weight 0.3 --vol_uv_weight 0.3 --epochs 20`.
- **Result:** 19 epochs in 26 min (timeout). best `val/avg_surf_p=87.47` at epoch 19 (just barely below 90). Trajectory: 104 → 107 → 105 → 102 → 104 → 99 → 96 → 98 → 99 → 94 → 95 → 94 → 92 → 90 → 89 → 89 → 88 → 88 → 87. Predictions at `askeladd/a2bed01`. W&B: askeladd/iter4-chain-lr2e5-spw10.
- **Verdict:** kept (-12 surf_p vs iter3). On the leaderboard at rank 7 with iter3's 92.25 test score; iter4 should bump me up.
- **Notes:** Chain is paying off: 137 (iter1) → 114 (iter2) → 100 (iter3) → 87 (iter4). frieren is at 53.32 with bs=2, lr=2e-6, NO subsampling (chain3); they finish each chain with full meshes. Iter5: try the same — `train_subsample=0`, bs=2, lr=5e-6, see if removing the subsample distribution shift gives a final boost.

### 2026-04-27 — iter3: 4-weight loss (thorfinn recipe) + warm-start + 20 epochs
- **Hypothesis:** Replace the simple `surf_weight × (uv+p)/3` loss with thorfinn's 4-region/channel weighting `surf_p_w=6, surf_uv_w=1, vol_p_w=0.5, vol_uv_w=0.5`. Their config (W&B: `thorfinn/iter1-warmstart-surfp` → 54.81) suggests this gives finer control over what the model optimises. Continue the chain warm-start + sub40k + bs4 + lr5e-5 setup that worked in iter2.
- **Change:** `train.py`: replaced the chan_w/surf_weight scheme with four explicit weights; total loss `= surf_p_weight*l_surf_p + surf_uv_weight*l_surf_uv + vol_p_weight*l_vol_p + vol_uv_weight*l_vol_uv`. Switched val split_loss to a similar weighted sum. epochs 8→20.
- **Result:** 19 epochs in ~26 min (timeout). best `val/avg_surf_p=99.84` at epoch 19. Trajectory was non-monotonic (loss reformulation perturbed warm-started weights → epoch 1 surf_p=210 vs iter2 final 114), but recovered: 210 → 193 → 169 → 152 → 164 → 142 → 145 → 123 → 123 → 127 → 115 → 112 → 124 → 105 → 105 → 105 → 102 → 100 → 100. Predictions at `askeladd/84fb943`. W&B: askeladd/iter3-4weight-warm-lr5e5-20e.
- **Verdict:** kept (-15 surf_p vs iter2). Crossed below 100 — should land top-3 once scored. Improvement still happening at the end → iter4 should chain further.
- **Notes:** Surprise: iter2's effective surface-pressure weight (`surf_weight=15 × p_chan_w=4 / 3 ≈ 20`) was actually higher than iter3's (`surf_p_weight=6`), yet iter3 ended better. The relative `surf_p / vol_p` ratio matters less than I thought — what mattered was giving the model dedicated coefficients per region/channel so the optimiser can reweight cleanly. Mistakes: I ran `python -c "import train"` to "validate" the syntax — that triggered a real training run and crashed mid-epoch on a stale `cfg.surf_weight` reference; now removed. iter4: keep chain, drop LR to 2e-5, push surf_p_weight to 10 to see if more pressure emphasis helps now that the model is well-trained.

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
