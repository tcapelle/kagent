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

### 2026-04-27 — iter2: warm-start bs=2 full-mesh lr=2e-5 (frieren breakthrough recipe)
- **Hypothesis:** Frieren's breakthrough was bs=2 + no subsample (full mesh) + warm-start. With bs=2 we get 4× more gradient updates per epoch (750 vs ~188); with full mesh the model sees every node so it can learn Re-dependent field structure that subsampling drops.
- **Change:** `python train.py --warm_start /tmp/iter1_best.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10`. Same model arch as iter1.
- **Result:** val/loss=**1.6100** at epoch 9/10 (25.1 min, 29.1 GB peak). Per-split (best epoch): single=2.75, rc=1.92, cruise=0.45, re_rand=1.33. Run ztzcdl7y.
- **Verdict:** kept — 12% reduction in val/loss vs iter1 (1.83 → 1.61). Predictions submitted to apr27-4/alphonse/77fdced.
- **Notes:** ~150s/epoch as expected. RAM used 29 GB (vs 10.5 GB at sub40k). Big drop on rc (-16%) and cruise (-30%) — full mesh helped most for tandem geometry. Continuing chain at lr=5e-6 for iter3.

### 2026-04-27 — iter1: 192×6 slice=64 bs=4 sub40k L1 p_w=3 bf16 fresh-start
- **Hypothesis:** Adopt frieren's apr23 recipe (proven to score 35-42 on past leaderboards): 192×6 Transolver, slice_num=64, batch=4, subsample non-surface to 40k, L1 loss with p_weight=3, bf16, 3-epoch warmup + cosine. Quick-converging baseline before chaining.
- **Change:** Refactored — Transolver moved to model.py. Added bf16/L1/p_weight/warm_start/subsample_collate/grad_clip/warmup-cosine to train.py. predict.py loads config.yaml from checkpoint dir.
- **Result:** val/loss=**1.8264** at epoch 24/25 (19.3 min, 10.5 GB peak). Per-split: single=2.86, rc=2.29, cruise=0.64, re_rand=1.62. Run mscwi9ck.
- **Verdict:** kept — strong baseline matching frieren's iter4 (1.91). Predictions submitted to apr27-4/alphonse/def6b08.
- **Notes:** ~46s/epoch (375 batches). Ready for iter2 = warm-start bs=2 no-subsample full-mesh (breakthrough recipe).

