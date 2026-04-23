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

### 2026-04-23 — iter1-big256 (baseline)
- **Hypothesis:** larger Transolver (n_hidden=256, n_layers=8, slice_num=128) with bf16 autocast + grad checkpointing beats the 128/5 template.
- **Change:** model.py (added grad checkpoint), train.py (bs=2, bf16 autocast, grad clip=1.0, mirror ckpt to pvc)
- **Result:** val/loss=4.98 at epoch 4 (17.6GB VRAM, 7.5 min/epoch → only 4 epochs fit in 30 min)
- **Verdict:** kept — commit 9b456ae (preds already submitted)
- **Notes:** still monotonically improving each epoch; bottleneck is throughput, not capacity. Per-split val: single_in_dist=9.05 (hardest), rc=5.49, cruise=1.80, re_rand=3.59. Next: subsample volume points to fit more epochs.
