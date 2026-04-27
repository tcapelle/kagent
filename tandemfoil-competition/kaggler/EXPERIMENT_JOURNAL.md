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

### 2026-04-27 — iter1: smooth-L1 + bf16 AMP + surf_weight=25
- **Hypothesis:** competition metric is `avg/mae_surf_p` (L1). Switching MSE→Smooth L1 (β=0.1) and raising `surf_weight` from 10→25 should align training with the metric. bf16 autocast lets us fit more epochs in 30 min.
- **Change:** `train.py` — Smooth L1 loss, surf_weight=25, bf16 autocast for fwd+loss, grad clip=1.0, n_hidden=192/n_layers=6/n_head=6 (matches apr27 frieren best). `model.py` extracted for clean `predict.py` import (refactor).
- **Result:** epoch 11/80 hit 30-min timeout. val/loss=5.998 (smooth-L1, surf_weight=25 weighted), avg_mae_surf_p=110.4. Per-split val_loss: single=7.73, rc=7.01, cruise=3.88, re=5.38. VRAM 58.1 GB. Run id `w2bmc9bd`, ckpt commit `84eae1a`.
- **Verdict:** kept (no prior baseline on this branch, first checkpoint). But avg_mae_surf_p=110 is far worse than apr27 frieren's 42.11 — likely because Smooth-L1 with β=0.1 has near-constant gradient for typical normalized errors, slowing convergence; we only ran 11 epochs.
- **Notes:** Likely fixes for iter2: drop Smooth L1 in favor of pure MSE (proven to train faster) OR keep Smooth L1 but with β=1.0 (Huber-like with larger quadratic region), increase epochs by speeding up (e.g. smaller batch padding via subsampling, or compile/SDPA). Also consider warmup LR + larger lr to converge faster.
