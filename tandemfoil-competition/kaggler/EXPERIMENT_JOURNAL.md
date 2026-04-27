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

### 2026-04-27 — iter4: ensemble iter1 + iter2 weighted 0.3/0.7
- **Hypothesis:** iter1 (bs=4 sub=40k) and iter2 (bs=2 nosub) train on different mesh resolutions, so their errors should partially decorrelate. Weighted average favoring iter2 (the stronger one) should pull predictions toward iter2 while iter1 covers iter2's blind spots.
- **Change:** Added `ensemble.py` that averages saved test predictions per-sample. `python ensemble.py --sources f2e8e4f 89eb6fd --weights 0.3 0.7`. Commit `9c8be71`.
- **Result:** Pending scoring. (iter1 alone 58.60, iter2 alone 47.32.)
- **Verdict:** TBD — depends on scorer.
- **Notes:** If this clears 47.32, the diversity from sub=40k vs nosub paid off; otherwise ensemble dilutes iter2's edge and we should drop iter1 from blends.

### 2026-04-27 — iter3: chain warm-start iter2 lr=2e-5 bs=2 nosub — plateau
- **Hypothesis:** Continue chain: warm-start iter2 (val 1.4497) at lr=2e-5 with cosine, 10 ep, bs=2 no-sub. Frieren's chain saw 0.05/iter improvements; aim for ~0.05 → val 1.40.
- **Change:** No code change. `train.py --warm_start models/model-f2pq4i1f/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10 --warmup_epochs 0`. Commit `e9df9e1` (predictions landed there because the train.py infra-fix commit moved HEAD mid-run).
- **Result:** 10 epochs in 25.1 min. Best epoch 1, val/loss=1.4502 — basically identical to iter2's 1.4497. Per-split val: single=2.37, rc=1.78, cruise=0.37, re_rand=1.28. Test: pending.
- **Verdict:** discarded as a single-model improvement (no progress over iter2). Will retain for ensembles but expect minimal added diversity.
- **Notes:** lr=2e-5 was too low to escape iter2's basin. The cosine tail just hovers. Frieren's chain plateaued similarly around iter21-23 (val 1.44). Time better spent on architecturally diverse models for ensembles.

### 2026-04-27 — iter2: warm-start iter1 with bs=2 no-subsample (frieren's breakthrough config) 🚀
- **Hypothesis:** Apply frieren's iter93 trick: warm-start a converged base model with bs=2 + no subsampling. The 4x more gradient updates per epoch + full 240k-mesh inputs let the model learn Re-dependent field structure. Should jump val/loss meaningfully and unlock big surf_p gains.
- **Change:** `train.py --warm_start models/model-wrz7s30a/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 5e-5 --epochs 10 --warmup_epochs 1`. No code changes; commit `89eb6fd`.
- **Result:** 10 epochs in 25 min (149s/ep, 29 GB peak). **val/loss 1.4497** at epoch 10 (vs iter1's 1.6789). Per-split val: single=2.37, rc=1.78, cruise=0.37, re_rand=1.29 — all four splits improved, especially re_rand (-15%) and rc (-12%). **Test: avg_surf_p 47.32** (single=52.20, rc=63.89, cruise=27.87, re_rand=45.31). Jumped rank 5→3, only 5.21 behind #1 (frieren 42.11).
- **Verdict:** kept. Confirms the bs=2 + no-sub recipe transfers. Big single-step gain (-11.28 surf_p).
- **Notes:** Mesh memory peak 29 GB (vs 10 GB with sub=40k) — well under the 96 GB budget. Each step still ~5 it/s. Next: continue chain at lr=2e-5 to squeeze the cosine tail further.

### 2026-04-27 — iter1: replicate proven recipe (192x6, L1, p_weight=3, slice=64, bs=4, sub=40k)
- **Hypothesis:** Replicate frieren's apr23 mid-iteration recipe (their iter15 era). 192x6 Transolver, L1 loss, p_weight=3, surf_weight=10, bs=4, subsample=40k, 30 epochs with 3-epoch warmup + cosine. Should give a clean, well-converged base ~val 1.7 → ~80 surf_p.
- **Change:** Extracted `Transolver` to `model.py`. Rewrote `train.py` with bf16 autocast, subsample collate, warm-start support. Fixed `predict.py` to load via `config.yaml`. Commit `f2e8e4f`.
- **Result:** 30 epochs in 23.2 min. **val/loss 1.6789** at epoch 30. Per-split val: single=2.59, rc=2.03, cruise=0.60, re_rand=1.51. Test: **avg_surf_p 58.60** (single=56.56, rc=74.68, cruise=38.78, re_rand=64.37). Jumped rank 5→5 but +6.62 pts over previous personal best (65.22).
- **Verdict:** kept. Solid base for warm-start chain.
- **Notes:** Cosine + warmup is converging cleanly. 23.2 min leaves 7 min headroom for longer runs. Next: warm-start chain with bs=2 no-subsample (frieren's iter93 breakthrough went from val 1.4→1.0 → 35 surf_p with this exact move).

