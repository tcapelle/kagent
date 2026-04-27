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

### 2026-04-27 — iter2: warm-start fine-tune with thorfinn 4-weight loss
- **Hypothesis:** Thorfinn (45.94) jumped from 458 → 65 by warm-starting from a converged checkpoint and fine-tuning with separate region/channel loss weights heavily favoring surface pressure (`surf_p=6, surf_uv=1, vol_p=0.5, vol_uv=0.5`). Apply the same trick to iter1's 84.71 ckpt.
- **Change:** train.py: replaced `surf_weight + ch_weights[p_weight]` with 4 explicit weights. Added `_l1_uv_p` helper. Run `--warm_start checkpoints/best.pt --lr 5e-5 --epochs 50`. Cosine over 50 epochs (no warmup since warm-starting). Run id `hsytzk3a`.
- **Result:** Best val/avg_surf_p=79.12 at epoch 3 (down from 84.71). Plateaued thereafter, oscillating 82–84. Hit 30-min cap at epoch 43. Train surf loss kept decreasing (0.45 → 0.27) but val surf_p stayed flat → overfitting.
- **Verdict:** Kept (`b17c849` + ckpt commit `9faf840`). Modest improvement (~5 surf_p). Predictions submitted to `fern/b17c849`.
- **Notes:** Quickly converges within 3 epochs then overfits, even with the surf-p heavy loss. Suggests model capacity / regularization is the bottleneck — the 1.71M-param Transolver with slice_num=64 has saturated. Next directions: bigger slice_num=128, dropout, or augmentation. Also worth trying lower LR (1e-5) for more stable fine-tune.

### 2026-04-27 — iter1: thorfinn-style 192/6/6 Transolver baseline
- **Hypothesis:** Reproduce thorfinn's recipe (Transolver 192h/6L/6H, L1 with channel weights on p, surface re-weighting, 40k-pt subsample, bf16+warmup+cosine) to establish a solid baseline before any custom work. Previous fern submission was 131.69; thorfinn at 45.94.
- **Change:** train.py: L1 loss with `ch_weights=[1,1,p_weight=3]`, `surf_weight=10`, batch_size=4, lr=5e-4 → cosine, warmup=3 epochs, 40k subsample (all surface + random volume), 30-min cap. Model 192/6/6, slice_num=64. Run id `go59tm9o`.
- **Result:** Best `val/avg_surf_p=84.71` at epoch 36/43. Per-split surf_p: single=80.5, geom_rc=123.6 (worst — OOD camber), geom_cruise=51.1 (best), re_rand=83.6. Train surf L1 0.51 → 0.11; vol 0.65 → 0.18. VRAM peak 9.7 GB.
- **Verdict:** Kept (`6e73d15`). Big improvement over previous fern submission (131.69 → ~85). Predictions submitted to `predictions/fern/6e73d15`.
- **Notes:** geom_camber_rc is the bottleneck — likely the "out-of-distribution camber" generalization gap. Next: try thorfinn's warm-start fine-tune trick — heavy `surf_p_weight=6` with low LR for last few epochs. Slice_num=128 would also be worth trying but requires retraining the model from scratch.

