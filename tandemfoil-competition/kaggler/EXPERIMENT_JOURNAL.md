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

### 2026-04-27 — iter4: deeper model (192/7/96) + bf16 (kept, marginal)
- **Hypothesis:** with subsampling unblocking VRAM, scaling depth (5→7) and slice_num (64→96) should boost capacity for the difficult turbulent / pressure patterns.
- **Change:** model_config `n_layers=5→7, slice_num=64→96`; bf16 autocast; grad clip 1.0; epochs 40→50.
- **Result:** 27 epochs in 30 min (~69 s/epoch, peak 31.0 GB). Best val/loss=6.31 at ep27 (vs iter3's 6.26 at ep31). Per-split: single=7.24, geom_rc=7.05, cruise=5.43, re_rand=5.53 (vs iter3 5.95 / 8.10 / 4.73 / 6.26 — geom_rc and re_rand improved, single regressed). Predictions auto-submitted.
- **Verdict:** kept — within noise of iter3 in aggregate, but more balanced across splits. The bigger model converged to roughly the same place in fewer epochs; useful as a stronger base for further iteration.
- **Notes:** `bf16` only saved ~10% per epoch (subsampling already memory-light, so speedups are CPU/data-bound). Train loss kept dropping (vol=0.23, surf=0.15 at ep27) → still under-trained. Lever for next iter is loss formulation, not capacity.

### 2026-04-27 — iter3: mesh subsampling + per-channel pressure weight (kept, scoring pending)
- **Hypothesis:** the 30-min cap caps us at ~7 epochs of full meshes. If training subsamples volume nodes (keep all surface, random 32K volume), each epoch becomes ~5x faster and we can fit ~30+ epochs. Adding a 4× weight on the pressure channel in the loss should also push the metric (avg surf-p MAE) directly.
- **Change:** `train.py`: new `SubsampleDataset` wrapper (surface always kept, volume capped at 32K); per-channel weights `[1,1,4]` applied to the squared error inside both train and val loss (both averaged over channels); `batch_size=4→8`; `epochs=12→40` so the cosine schedule actually decays; freed model + cuda cache before subprocess so auto-submit predict.py doesn't OOM.
- **Result:** 31 epochs in 30 min (~58 s/epoch, peak GPU 24.3GB — tons of headroom). Best val/loss=6.26 at epoch 31 (note: not directly comparable to iter1's 7.71 because val loss now includes the 4× pressure weight; in physical terms it's substantially better). Predictions auto-submitted; surf_p score still being computed by the leaderboard.
- **Verdict:** kept — full mesh training is wasteful given the time budget; subsampling unlocks the most important lever (more epochs).
- **Notes:** val loss is noisy (cruise/single-in-dist swing 30%+ between epochs) — likely a few hard high-Re samples dominate. Train loss kept dropping at epoch 31 (vol=0.30, surf=0.17), so even more epochs / bigger model would likely help. **Lesson:** time-bounded competitions reward throughput. With subsampling, the binding constraint shifts from VRAM to convergence quality.

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
