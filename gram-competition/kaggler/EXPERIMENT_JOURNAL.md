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

### 2026-04-16 — iter2: Perceiver-IO latent bottleneck (L=128)
- **Hypothesis:** pointwise ResMLP saturated because it cannot exchange information between neighbors; adding a Perceiver-IO latent bottleneck with L=128 learned queries will let the model aggregate global context for turbulent components.
- **Change:** `train.py` — replaced `ResMLP` with `Perceiver(point_dim=256, latent_dim=384, n_latents=128, n_process_blocks=6, heads=6, dim_head=64)`. Fourier(16 bands) pos features, time-mean velocity extra feature, time sinusoidal embed conditions the learned latent init. Encoder cross-attn + 6 self-attn processor + decoder cross-attn + residual/no-slip head.
- **Result:** val/l2 = **1.4033** at epoch 9; plateaued/oscillated 1.40–1.46 for epochs 10–11. 69 s/epoch, 3.1 GB. Run `y6e02zev`. Predictions saved to `/mnt/new-pvc/predictions/apr16/nezuko/b6d05db/val.pt`.
- **Verdict:** kept — strong improvement over iter1 (−0.43, −23%). Current #2 on leaderboard (thorfinn 1.30 leads with Transolver).
- **Notes:** Memory only 3.1/96 GB — huge room for capacity. LR schedule (cosine T_max=50) barely annealed by epoch 11 → LR too high at convergence. Oscillations likely batch_size=1 noise. Next: bigger model + grad accum + LR fit to actual epoch count.

### 2026-04-16 — iter1: physics-aware ResMLP (FiLM-t, residual, no-slip)
- **Hypothesis:** strong physics priors (residual to last input timestep, no-slip BC, per-sample pos+vel normalization, FiLM time conditioning) should close most of the easy laminar gap on top of the pointwise baseline.
- **Change:** `train.py` — replaced `BaselineMLP` with `ResMLP(hidden=384, n_blocks=8)`: FiLM-conditioned ResBlocks with sinusoidal time embedding of all 10 `t` values, residual delta prediction, airfoil-mask feature, hard no-slip BC.
- **Result:** val/l2 = **1.833** at epoch 1; train MSE plateaued at ~8.24 by epoch 2; val oscillated in 1.83–1.92 with no improvement through epoch 4. 9.4 GB peak, 62 s/epoch. Run `g5wlqt2e`.
- **Verdict:** kept as baseline (first checkpoint on disk). Killed early — pointwise model saturated without spatial interaction.
- **Notes:** Pointwise architecture can't model turbulence (no neighbor exchange). Need spatial operators. Try Perceiver-IO next (latent bottleneck, O(N·L) attention). Also consider: training-time point subsampling for more epochs, and per-component loss weighting (vel_std varies 7× between Uy and Ux, so MSE is Ux-dominated).

