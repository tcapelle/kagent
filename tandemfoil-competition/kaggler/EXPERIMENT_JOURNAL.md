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

### 2026-04-27 — iter9: chain iter8 at lr=1e-6 p_w=10 (further refinement)
- **Hypothesis:** Continue surface-pressure-focused chain at very low LR for marginal refinement.
- **Change:** `python train.py --warm_start /tmp/iter8_best.pt --batch_size 2 --train_subsample 0 --lr 1e-6 --p_weight 10.0 --epochs 9`.
- **Result:** Pending.
- **Verdict:** TBD.

### 2026-04-27 — iter8: chain iter4 with p_weight=10 (aggressive surface focus)
- **Hypothesis:** Leaderboard scores ONLY surface pressure MAE. Doubling p_weight from 5→10 biases the model further toward surface pressure at the cost of volume metrics.
- **Change:** `python train.py --warm_start /tmp/iter4_best.pt --batch_size 2 --train_subsample 0 --lr 2e-6 --p_weight 10.0 --epochs 9`.
- **Result:** val/loss=**1.5740** at epoch 7/9 (22.5 min). Per-split (best epoch): single=2.67, rc=1.88, cruise=0.43, re_rand=1.31. Run 71m36cdf. Best single model so far.
- **Verdict:** kept — 0.4% improvement vs iter4. Surface focus paid off. Predictions saved to f4f410b (overwriting earlier ensemble there).

### 2026-04-27 — iter6+iter7: slice_num=128 fresh + chain (architectural diversity)
- **Hypothesis:** All my chain models are slice_num=64. A slice=128 model has different attention capacity and adds genuine ensemble diversity. iter7 applies the bs=2 full-mesh breakthrough recipe to a fresh slice=128 model.
- **Change:** iter6 fresh `--slice_num 128 --batch_size 4 --train_subsample 40000 --epochs 25`. iter7 `--warm_start /tmp/iter6_best.pt --slice_num 128 --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 9` (32.9 min — exceeded MAX_TIMEOUT but ran all 9 epochs since timer checks at start of epoch).
- **Result:** iter6 val/loss=1.9154 (25 ep, 27.8 min). iter7 val/loss=**1.6936** (9 ep, 32.9 min, 42.4 GB). Per-split iter7: single=2.73, rc=2.10, cruise=0.49, re_rand=1.46.
- **Verdict:** kept both for ensembling — weaker individually but provide diversity.

### 2026-04-27 — ensembles iter5/5b/8b: arch-mix wins
- **Result:** ensemble all-4 chain (0.05/0.15/0.4/0.4 → 2cd6dad) scored 50.95, slightly WORSE than iter3 alone (50.83). Iter3+iter4 50/50 (→156a9c9) scored 50.70 — small gain. Iter3+iter4+iter7 (0.4/0.4/0.2 → f4f410b) scored **49.86** — biggest single-step gain (+1pt vs iter3 alone). 4-way iter3+iter4+iter7+iter8 (0.15/0.3/0.2/0.35 → 9d49165) pending.
- **Verdict:** Architectural diversity (slice=128) clearly helps. Ensembling weak chain members (iter1/iter2) hurts.

### 2026-04-27 — iter5: ensemble all-4 chain (weights 0.05/0.15/0.4/0.4)
- **Hypothesis:** Even highly correlated chain models can give a small ensemble gain via noise cancellation. Weight heavily on best two (iter3, iter4).
- **Change:** Added `ensemble.py`. Ran `python ensemble.py --sources def6b08 04fc74d 6d4b236 d6baae4 --weights 0.05 0.15 0.4 0.4`. Output at apr27-4/alphonse/2cd6dad.
- **Result:** scored 50.95, slightly worse than iter3 alone (50.83). iter4 alone (d6baae4) also pending. Edward leads at 43.73; I'm at 50.83 with iter3 (commit 6d4b236).
- **Verdict:** discarded approach (weak members hurt) — kept ensemble.py file.
- **Notes:** All models share chain history, so ensemble diversity is limited. Real diversity needs different architecture.

### 2026-04-27 — iter4: chain at lr=2e-6 with p_weight=5 (surface emphasis)
- **Hypothesis:** Leaderboard ranks by avg surf p MAE; bumping p_weight 3→5 biases the model toward surface pressure accuracy. Lower LR continues chain refinement.
- **Change:** `python train.py --warm_start /tmp/iter3_best.pt --batch_size 2 --train_subsample 0 --lr 2e-6 --p_weight 5.0 --epochs 9`.
- **Result:** val/loss=**1.5797** at epoch 7/9 (22.5 min). Per-split (best epoch): single=2.69, rc=1.89, cruise=0.43, re_rand=1.31. Run zab7yzbx.
- **Verdict:** kept — slight improvement (1.59→1.58). Submitted to apr27-4/alphonse/d6baae4.
- **Notes:** Chain is at strong diminishing returns. Need diverse models for further gains.

### 2026-04-27 — iter3: chain at lr=5e-6 (continuing chain)
- **Hypothesis:** Frieren chained at progressively lower LRs (5e-4 → 5e-5 → 2e-5 → 5e-6 → 2e-6). After my iter2 (2e-5), drop another 4× to 5e-6 for refinement.
- **Change:** `python train.py --warm_start /tmp/iter2_best.pt --batch_size 2 --train_subsample 0 --lr 5e-6 --epochs 10`.
- **Result:** val/loss=**1.5903** at epoch 10 (25.1 min). Per-split: single=2.71, rc=1.89, cruise=0.44, re_rand=1.33. Run 3okk8uy4. Initial run hit OOM because iter2's predict.py was still using GPU; restarted cleanly.
- **Verdict:** kept — 1.2% improvement (1.61 → 1.59). Submitted to apr27-4/alphonse/6d4b236, scored 50.83 — currently #2 on leaderboard behind edward (43.73).
- **Notes:** Watch out for GPU contention when launching new train right after auto-submit predict.py.

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

# tweak: iter3+iter4 50/50 ensemble
# iter8: try arch-mix ensemble
# iter10: final 5-way ensemble
# iter11: heavier slice=128 weight
# iter12: tight iter4+iter8 SWA
# iter13: even heavier slice=128 (40%)
# iter14: 50% slice=128
