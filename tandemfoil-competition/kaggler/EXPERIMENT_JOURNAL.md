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

### 2026-04-27 — iter9: Fourier features for spatial coords (running)
- **Hypothesis:** Tancik's coordinate-MLP spectral-bias result + every NeurIPS 2024 ML4CFD top-4 method using positional encodings → vanilla Transolver feeding raw (x,z) into the preprocess MLP underrepresents the high-frequency Cp curvature near the LE stagnation point. Adding random Fourier features (n_freqs=32, sigma=2.0) before preprocess should let the model represent the LE pressure peak and reduce surface-p MAE.
- **Change:** `models.py`: new `FourierFeatures` class; `Transolver` now takes `ff_n_freqs/ff_sigma` and concatenates `[B, N, 2*n_freqs]` sin/cos features to the input of `preprocess`; `train.py` model_config sets `ff_n_freqs=32, ff_sigma=2.0`. Everything else identical to iter8 (pure-L1 Huber beta=0.05, surf_weight=20, p_weight=4).
- **Result:** RUNNING.
- **Verdict:** TBD.
- **Notes:** other research-recommended levers queued for follow-up: inlet-velocity canonicalization + flip aug, panel-method Cp prior, AB-UPT-style surface-volume cross-attention, log-pressure target.

### 2026-04-27 — iter8: pure-L1 loss (huber_beta=0.05) (kept, scoring pending)
- **Hypothesis:** Huber beta=1.0 is still mostly quadratic for typical normalized errors; the leaderboard metric is **L1**, so dropping beta to 0.05 makes the loss essentially pure L1 with only a tiny stable-gradient region near zero. Should reduce val MAE compared to iter5's beta=1.0.
- **Change:** `train.py` `huber_beta=1.0→0.05`; reverted iter7's surf_weight=30/p_weight=8 back to iter5's 20/4 (one variable change).
- **Result:** 27 epochs in 30 min. Best val/loss=4.32 at ep27, val surf_p=94.7. Per-split val: single=95.9 / geom_rc=116.4 / cruise=77.1 / re_rand=89.3. Predictions auto-submitted (commit 67f9572). Test scoring queued.
- **Verdict:** kept — first val surf_p clearly under 100 (vs iter5's val ~110-120 at similar epoch). Matches the metric directly.
- **Notes:** when val_loss is L1, train loss is also L1 — magnitudes are not comparable to Huber beta=1 numbers in earlier iters. Keep checkpoint selection on val surf_p (added in iter7).

### 2026-04-27 — iter7: surf_weight=30, p_weight=8 (discarded → reverted in iter8)
- **Hypothesis:** push surface and pressure even harder in the loss to drive surface-p MAE down faster.
- **Change:** `train.py` `surf_weight 20→30`, `p_weight 4→8`; switched checkpoint selection from val/loss to surf_p_avg.
- **Result:** 27 epochs. Best val surf_p=116.8 at ep27. Train surf=0.10 (vs iter5 0.05) — heavier surface weighting kept the volume part undertrained.
- **Verdict:** discarded — heavier weighting hurt. Kept the surf_p checkpoint selection (carried into iter8).
- **Notes:** with 99%+ of nodes being volume and surface squared error already 20× weighted, pushing further drowned out the volume signal and slowed convergence.

### 2026-04-27 — iter6: longer training + smaller subsample (24K) (discarded)
- **Hypothesis:** smaller subsample → faster epochs → fit more passes through the data.
- **Change:** `train_n_volume=32K→24K, epochs=50→60`. Same loss/architecture as iter5.
- **Result:** 32 epochs, val/loss=1.73 at ep30, val surf_p=112. Test surf_p=101.75 (vs iter5's 98.27).
- **Verdict:** discarded — slightly worse on test. The smaller subsample added noise without enough additional passes to compensate.
- **Notes:** there's a sweet spot for subsample size; 24K is too small. 32K is the right ballpark.

### 2026-04-27 — iter5: Huber/smooth-L1 loss for L1 metric alignment (kept, scoring pending)
- **Hypothesis:** the leaderboard ranks by surface-pressure **L1** MAE, but training uses MSE which over-weights outlier high-Re samples (single_in_dist has ~5x the variance of cruise). Switching to Huber (smooth L1, beta=1.0) should better match the metric and reduce noisy val swings on outliers.
- **Change:** `train.py`: replaced `(pred - y_norm)**2` with smooth-L1 `where(|err|<beta, 0.5*err^2/beta, |err|-0.5*beta)` in both train and val loss; everything else identical to iter4 (192/7/96, bf16, p_weight=4, surf_weight=20).
- **Result:** 27 epochs in 30 min. Best val/loss=1.73 at ep27 — note the absolute value isn't comparable to MSE-based iter4 (6.31), but per-split decline is dramatic (single 7.2→1.50, geom_rc 7.0→2.60, cruise 5.4→1.14, re_rand 5.5→1.69). Train surf=0.05 at ep27 — model was still improving at the timeout. Predictions auto-submitted.
- **Verdict:** kept — Huber dramatically tightened val loss across all splits; the fact that train was still falling at termination suggests even more epochs / slightly higher lr would help.
- **Notes:** Ranking by val/loss across iters now requires explicitly tracked MAE, not val/loss values. Plan iter6 to either (a) train longer with same recipe, (b) push p_weight up, or (c) add an L1-explicit pressure-only refinement head. Three submissions still showing `incomplete` on the leaderboard scorer; predictions look valid (no NaN/Inf, correct shapes), so it's a scorer queue lag.

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
