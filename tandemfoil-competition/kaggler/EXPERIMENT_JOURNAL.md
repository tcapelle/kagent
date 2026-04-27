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

### 2026-04-27 — iter1: 192×6 slice=64 bs=4 sub40k L1 p_w=3 bf16 fresh-start
- **Hypothesis:** Adopt frieren's apr23 recipe (proven to score 35-42 on past leaderboards): 192×6 Transolver, slice_num=64, batch=4, subsample non-surface to 40k, L1 loss with p_weight=3, bf16, 3-epoch warmup + cosine. Quick-converging baseline before chaining.
- **Change:** Refactored — Transolver moved to model.py. Added bf16/L1/p_weight/warm_start/subsample_collate/grad_clip/warmup-cosine to train.py. predict.py loads config.yaml from checkpoint dir.
- **Result:** val/loss=**1.8264** at epoch 24/25 (19.3 min, 10.5 GB peak). Per-split: single=2.86, rc=2.29, cruise=0.64, re_rand=1.62. Run mscwi9ck.
- **Verdict:** kept — strong baseline matching frieren's iter4 (1.91). Predictions submitted to apr27-4/alphonse/def6b08.
- **Notes:** ~46s/epoch (375 batches). Ready for iter2 = warm-start bs=2 no-subsample full-mesh (breakthrough recipe).

