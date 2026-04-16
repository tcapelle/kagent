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

### 2026-04-16 — iter1: residual + no-slip + input norm
- **Hypothesis:** Three "free wins" from README: predict delta on top of velocity_in[-1] (strong prior), hard-zero at airfoil (physics), normalize velocity by dataset stats.
- **Change:** `model.py` — `ResidualMLP(hidden=384, n_blocks=8)`: normalize inputs, predict residual scaled by `vel_std` added to last-input velocity, zero out prediction at `idcs_airfoil`. Bumped hidden 256→384, blocks 6→8 (model still tiny at ~1.8M params, 8GB VRAM). 20 epochs, cosine LR.
- **Result:** val/l2_error = **1.7825** at epoch 20. train loss 8.15. 8.1GB peak. 18.1 min.
- **Verdict:** kept. Clear improvement over baseline (baseline is ~5+ in debug). But still worse than mar29 frieren (1.15) which had richer architecture. Room for big gains.
- **Notes:** Training still going down steadily at epoch 20 — the model is under-trained or under-parametrized. Need spatial context (neighbors); current model treats each point independently. Next: add per-point distance-to-airfoil feature and stronger spatial model.
  - Fix: predict.py was broken because importing `train.py` re-ran its top-level `sp.parse`. Moved `ResidualMLP` to `model.py`.
  - W&B run: p64qoxcf

