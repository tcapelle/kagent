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

### 2026-04-27 — iter1: 192x6 L1 + bf16 + p_weight=3 + warmup + bs=4 sub60K (commit 9a14753)
- **Hypothesis:** Reproduce frieren's apr23 recipe: 192x6 Transolver, L1 loss for outlier robustness, bf16, p_weight=3 surface-pressure boost, warmup+cosine, bs=4 with subsample to 60K volume nodes. Pre-train for warm-start chain.
- **Change:** Refactored `train.py`/`predict.py` and added `model.py` with Transolver. New flags: `loss_type`, `p_weight`, `warmup_epochs`, `train_subsample`, `warm_start`, bf16 autocast, grad_clip=1.0.
- **Result:** val/loss=1.9973 at epoch 29 (30.4 min, 15.3GB peak). Per-split val: single=2.54, rc=2.87, cruise=0.70, re_rand=1.87. Cosine still descending at the end → likely undertrained for this config.
- **Verdict:** kept as warm-start base for iter2. Score TBD but expected similar to last apr27 iter1 (~57 surf_p).
- **Notes:** A bit worse than last session's iter1 (val 1.685) — random init variance and `warmup_epochs=3` (last time was different). Real win comes next: bs=2 + no-subsample warm-start (frieren's iter93 went 1.4→1.0 → score 35).
