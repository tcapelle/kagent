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

### 2026-04-16 — exp1: residual + normalize + no-slip
- **Hypothesis:** Predicting delta from `velocity_in[-1]` is much easier than absolute velocity (|delta|=1.17 vs |v|=14 in raw units). Normalizing by vel_std balances the loss across Ux/Uy/Uz. Zero-init output layer → starts as identity. Hard no-slip on airfoil is a physical constraint the baseline ignores.
- **Change:** `train.py` BaselineMLP now: normalized v_in features, predicts delta_norm (zero-init head), denorms, adds to last frame, zeros out airfoil indices. hidden=384, n_blocks=8 (~4.7M params). Loss is MSE on normalized error.
- **Result:** pending
- **Verdict:** pending
