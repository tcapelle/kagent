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

### 2026-04-27 — iter1-apr23-baseline (KEPT)
- **Hypothesis:** rebuild apr23 nezuko's validated config (h=128, L=6, slice=96, n_head=4, surf_weight=1.5, weight_decay=3e-5, epochs=14, lr=5e-4, bf16 autocast, grad_clip=1.0). Modular split: model classes -> `model.py` so `predict.py` can import without launching training. Mirror best ckpt to `checkpoints/best.pt` and PVC.
- **Change:** new `model.py` with Transolver classes; `train.py` imports from it and applies bf16 autocast + grad_clip + PVC mirror. `predict.py` loads via `model.py` + reads `config.yaml` from checkpoint dir.
- **Result:** 13/14 epochs in 30.4 min. Best val/loss=0.5956 at epoch 13. Val surf_p MAE: in_dist=117.35, geom_rc=108.63, cruise=73.79, re_rand=91.61. **Avg val surf_p MAE = 97.85.** W&B `kagent-tandemfoil3/hjwi94ao`.
- **Verdict:** kept, ckpt committed at `55049c8` for first submission.
- **Notes:** Slightly worse than apr23 best (94.5), likely seed/split differences. Validation losses still trending down at epoch 13 — model is undertrained. Next: tackle the dominant pathology (per-sample pressure-variance imbalance) and lift surf_p with Huber + heavier weight on the pressure channel only.
