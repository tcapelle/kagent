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

### 2026-04-17 — residual-prediction-violet-e1
- **Hypothesis:** Residual prediction (output = v_last + delta) should beat absolute-velocity prediction since v_last is a strong prior; adding no-slip BC + richer features (temporal diffs, magnitudes, dt_out, per-sample pos normalization) + 8-block 512-hidden MLP with AMP bf16 should crush the baseline (0.80).
- **Change:** rewrote `train.py` with `PointResNet`: residual head, no-slip via mask, permuted reshapes (fixed the baseline's reshape that scrambles points vs pos), bf16 autocast, AdamW+cosine.
- **Result:** val/l2 = **1.2547** (worse than baseline 0.80). 49 epochs × 37 s, 6.6 GB VRAM. train_loss 7.93 → 3.33. Run `xw4tdege`.
- **Verdict:** discarded — regressed vs. baseline.
- **Notes:** The residual bias (`output = v_last` at init) gives val/l2 ≈ 1.63 at epoch 1, which is already worse than the baseline's "fit f(pos) → mean-flow" starting point (~0.80 after training). The point-wise model can't cancel the instantaneous turbulent fluctuations in v_last fast enough. The baseline's reshape "bug" is consistent across input/output, so it just learns `f(pos) → mean velocity field` which is a surprisingly strong signal given the mostly-laminar flow. **Next**: drop residual; predict absolute velocity with the baseline's output structure but add no-slip BC, richer features, and bigger capacity — keep closer to the baseline recipe.
