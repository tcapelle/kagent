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

### 2026-04-23 — iter2-subsample
- **Hypothesis:** subsampling volume points (keep all surface, 40K vol/sample) speeds up training 3x with minimal loss in quality. More epochs → lower val/loss.
- **Change:** train.py adds `subsample_volume` called per-batch; bs=4 (was 2); n_vol_train=40000
- **Result:** val/loss=3.0106 at epoch 14 (137s/epoch, 11.1GB VRAM). Down from iter1's 4.98 at epoch 4.
- **Verdict:** kept — commit 016070d
- **Notes:** Per-split @ epoch 14: single_in_dist=3.83, rc=4.05, cruise=1.43, re_rand=2.73. Still monotonically improving at epoch 14 — should try even more aggressive subsampling or fewer layers to allow more epochs.

### 2026-04-23 — iter1-big256 (baseline)
- **Hypothesis:** larger Transolver (n_hidden=256, n_layers=8, slice_num=128) with bf16 autocast + grad checkpointing beats the 128/5 template.
- **Change:** model.py (added grad checkpoint), train.py (bs=2, bf16 autocast, grad clip=1.0, mirror ckpt to pvc)
- **Result:** val/loss=4.98 at epoch 4 (17.6GB VRAM, 7.5 min/epoch → only 4 epochs fit in 30 min)
- **Verdict:** kept — commit 9b456ae (preds already submitted)
- **Notes:** still monotonically improving each epoch; bottleneck is throughput, not capacity. Per-split val: single_in_dist=9.05 (hardest), rc=5.49, cruise=1.80, re_rand=3.59. Next: subsample volume points to fit more epochs.
