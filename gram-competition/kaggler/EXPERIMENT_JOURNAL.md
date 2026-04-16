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

### 2026-04-16 — residual + no-slip + input-norm (iter 1)
- **Hypothesis:** Baseline predicts full velocity from scratch and ignores the fact that flow is nearly stationary over 5 steps. Predicting `delta = v_out - v_in[-1]`, enforcing zero velocity at `idcs_airfoil`, and normalizing inputs with dataset stats should each help.
- **Change:** `train.py` — BaselineMLP now stores vel_mean/vel_std buffers, normalizes v_in, adds v_in[-1] to the MLP output, and zeros predictions at airfoil indices. Same hidden=256, 6 blocks.
- **Result:** best val/l2_error = **1.7778** (epoch 19/50, ~19 min). VRAM peak 4.2 GB. `train/epoch_loss` ~7.8 by the end.
- **Verdict:** kept. First checkpoint committed as `checkpoints/best.pt`. Run `fern/residual-noslip-norm` (W&B id `bw4wpqff`).
- **Notes:** `predict.py` failed to auto-submit because `from train import BaselineMLP` triggered `sp.parse(Config)` in train.py's argparse. Fixed by wrapping train.py's script logic in `main()`/`__main__`. Resubmitted val predictions manually. Next ideas (biggest): (1) spatial context via kNN EdgeConv — the MLP is still per-point; (2) Fourier-feature pos encoding; (3) time-delta conditioning per output step.

