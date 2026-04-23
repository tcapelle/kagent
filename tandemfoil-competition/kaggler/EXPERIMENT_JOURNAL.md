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

### 2026-04-23 — iter3-bs8-vol20k
- **Hypothesis:** bs=8 + 20K vol pts + 25 epochs + warmup LR beats iter2. More epochs should help since loss still decreasing.
- **Change:** bs=4→8, n_vol=40K→20K, epochs=50→25, added LinearLR warmup 2ep + cosine.
- **Result:** val/loss=2.1905 at epoch 21 (86s/epoch, 22.2GB VRAM). Down from iter2's 3.01.
- **Verdict:** kept — commit b600021
- **Notes:** Per-split @ epoch 21: single_in_dist=2.50, rc=3.07, cruise=1.08, re_rand=2.11. Still improving; LR near zero by end. rc is the biggest remaining loss.

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
