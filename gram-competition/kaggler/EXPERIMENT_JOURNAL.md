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

### 2026-04-17 — residual + normalization + no-slip BC (point-wise ResMLP)
- **Hypothesis:** Predicting `delta = v_out - v_in[-1]` (residual) should be much easier than absolute velocity because the prediction horizon is only 5 ms (steps of 1 ms at freestream ~30 m/s). Normalize inputs with `stats.json` so Ux (std 20) doesn't dominate the MSE. Enforce no-slip at `idcs_airfoil` by zeroing predicted velocity.
- **Change:** `model.py/BaselineMLP` — point-wise ResMLP (hidden=512, 8 blocks) that outputs a normalized delta; add `v_in[-1]` back then denormalize. Loss computed in normalized space: `((pred - v_out) / vel_std).pow(2).mean()`.
- **Result:** val/l2_error = 1.3646 (epoch 20 of 23 @ 30.6 min, 10.8 GB peak). Training loss plateaued around 0.026.
- **Verdict:** Kept. Beats the copy-last-frame baseline (1.75) but nowhere near the top of the leaderboard (~0.75). Point-wise model has no spatial context — each point is treated independently, so it cannot actually learn the flow dynamics; it mostly learns a slightly smoothed version of copy-last-frame.
- **Notes:** Sanity-checked: `pred = v_in[-1]` (pure copy) also scores **1.7496** on val — my model only shaved 0.38 from copying. The leaders must be exploiting spatial context. Also fixed `predict.py: unrecognized arguments --checkpoint` by moving the model class into `model.py` so `from train import …` no longer fires the argparser. Next: voxel-grid + 3D CNN so each point can see its neighborhood.

