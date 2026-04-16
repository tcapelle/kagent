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

### 2026-04-16 — iter1: physics-aware ResMLP (FiLM-t, residual, no-slip)
- **Hypothesis:** strong physics priors (residual to last input timestep, no-slip BC, per-sample pos+vel normalization, FiLM time conditioning) should close most of the easy laminar gap on top of the pointwise baseline.
- **Change:** `train.py` — replaced `BaselineMLP` with `ResMLP(hidden=384, n_blocks=8)`: FiLM-conditioned ResBlocks with sinusoidal time embedding of all 10 `t` values, residual delta prediction, airfoil-mask feature, hard no-slip BC.
- **Result:** val/l2 = **1.833** at epoch 1; train MSE plateaued at ~8.24 by epoch 2; val oscillated in 1.83–1.92 with no improvement through epoch 4. 9.4 GB peak, 62 s/epoch. Run `g5wlqt2e`.
- **Verdict:** kept as baseline (first checkpoint on disk). Killed early — pointwise model saturated without spatial interaction.
- **Notes:** Pointwise architecture can't model turbulence (no neighbor exchange). Need spatial operators. Try Perceiver-IO next (latent bottleneck, O(N·L) attention). Also consider: training-time point subsampling for more epochs, and per-component loss weighting (vel_std varies 7× between Uy and Ux, so MSE is Ux-dominated).

