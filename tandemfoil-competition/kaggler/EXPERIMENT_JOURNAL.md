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

### 2026-04-28 — iter26: extreme p_weight=15 chain on iter24 → val_surf_p=51.05, ensemble val=48.85
- **Hypothesis:** Push p_weight even higher (askeladd-territory) on the deepest chain.
- **Change:** Warm-start iter24 (`model-2hapqwcc`), `lr=1e-6 p_weight=15 epochs=11`. Run `7dmlgkos`.
- **Result:** epoch 11, val/loss=0.8143, surf_p=**51.05** — new best single (was iter24 51.13). Ensemble val=**48.85** (-0.003). Optimizer assigned 0.094 weight to iter26. Submitted at `8f3e0b9`.
- **Verdict:** kept; small gain. Ensemble appears fully saturated.

### 2026-04-28 — iter25: chain bigger model with p_weight=8 → val_surf_p=57.70 (still 0 ensemble weight)
- **Hypothesis:** Apply high-p chain to the bigger 256/6/8/96/mlp4 backbone too. Maybe combined with iter20's chain it'll add complementary errors.
- **Change:** Warm-start iter20 (`model-ww9506mn`), `lr=5e-6 p_weight=8 epochs=6`. Also fixed predict.py to use bf16 autocast (consistency with training). Run `5h2z94rc`.
- **Result:** epoch 6 best, val/loss=0.8878, surf_p=**57.70**. Optimizer assigned 0 weight. Bigger model with chain still doesn't break into the ensemble.
- **Verdict:** kept; ensemble val unchanged at 48.85.
- **Notes:** The bigger arch is fundamentally less competitive at the val level than my mature small chains. Ensemble has saturated around 48.85 for last 5+ iters. Path forward is unclear — chain is fully exhausted.

### 2026-04-28 — iter24: deepest chain on iter22 (lr=5e-7, p_weight=10) → val_surf_p=51.13, ensemble val=48.85
- **Hypothesis:** Squeeze a tiny gain from the best chain by going lr=5e-7 with p_weight=10.
- **Change:** Warm-start iter22 (`model-0t9gz1g7`), p_weight=10. Run `2hapqwcc`.
- **Result:** epoch 3 best, val/loss=0.8000, surf_p=**51.13** — new best single (was iter22 51.16). Ensemble val=**48.85** (-0.01). Submitted at `cb7ca23`.
- **Verdict:** kept; chain is fully saturated.

### 2026-04-28 — iter23: chain iter18 with p_weight=8 → val_surf_p=52.73, ensemble val=48.86
- **Hypothesis:** Same recipe as iter22 but on iter18 (the other strong chain). Adds another high-p chain checkpoint.
- **Change:** Warm-start iter18 (`model-cxsc8wnv`), `lr=1e-6 p_weight=8 epochs=11`. Run `4vknxeki`.
- **Result:** val/loss=0.8131, surf_p=52.73 (vs iter18's 52.80). Ensemble val=48.86 (-0.02). Top weights: iter22=0.24, iter9=0.22, iter15=0.17, iter11=0.12, iter23=0.09, iter18=0.09. Submitted under `ce98c93`.
- **Verdict:** kept; another -0.02. **Leaderboard rank #5 at 41.08** — tanjiro jumped ahead. Gap to rank 1 (askeladd 33.72) is 7.36 — too far for chain refinements alone.

### 2026-04-28 — iter22: chain iter15 with p_weight=8 (askeladd-style late-stage upweight) → val_surf_p=51.16, ensemble val=48.88
- **Hypothesis:** iter21 showed high p_weight from scratch is too aggressive, but on a mature chain checkpoint it should focus the late-stage refinement on pressure (the only scored metric). Apply p_weight=8 (medium between my 3 and askeladd's 12-16) to iter15 chain at lr=1e-6.
- **Change:** Warm-start iter15 (`model-n3rus7ko`), bs=2, full mesh, `lr=1e-6 p_weight=8 epochs=11`. Run `0t9gz1g7`.
- **Result:** epoch 11 best, val/loss=0.7899, surf_p=**51.16** — new best single (was iter15 at 51.23). Ensemble val=**48.88** (-0.02). Top weights now: iter9=0.25, iter15=0.21, iter22=0.18, iter18=0.17, iter11=0.10. Submitted at `bc68b6c`.
- **Verdict:** kept; high p_weight chain works on mature checkpoints. Plan iter23 = chain iter18 (the other strong chain) with p_weight=8 too.

### 2026-04-28 — iter21: p_weight=12 from scratch (askeladd insight) → val_surf_p=61.23
- **Hypothesis:** Askeladd's session log mentions p_weight=12-16. Heavy upweight on the only scored channel. Try p_weight=12 from scratch with otherwise-unchanged iter1 recipe.
- **Change:** `--p_weight 12.0` (4x my baseline). Run `cb77wl3c`.
- **Result:** epoch 30, val/loss=0.9729, surf_p=**61.23** — much worse than iter1's 53.46. Optimizer gave 0 weight. p_weight=12 from random init makes the model overshoot pressure and ignore Ux/Uy structure that helps generalize.
- **Verdict:** kept; lesson — high p_weight only works in late-stage chain, not from scratch. Submitted at `ceb4b1e`.

### 2026-04-28 — iter20: warm-chain askeladd-arch (bs=2, full mesh, 7 epochs) → val_surf_p=58.22
- **Hypothesis:** iter19 (256/6/8 mlp=4) was undertrained at 64.78. Apply the proven chain recipe.
- **Change:** Warm-start iter19, bs=2, train_subsample=0, lr=2e-5, p_weight=3, 7 epochs (limited by 4.5min/epoch with this big arch). Run `ww9506mn`.
- **Result:** val/loss=0.8201, surf_p=**58.22** (-10% from 64.78). Optimizer still gave 0 weight — even after chaining, the bigger arch is weaker than my matured Fourier+chain models. Submitted at `7eb86f7`.
- **Verdict:** kept; underperforms vs chain models. Likely needs more chain steps (askeladd has 13 checkpoints).
- **Notes:** 4.5 min/epoch is too slow to fit many chain steps in 30 min. Each iter only buys ~3 surf_p improvement at this size. Path forward might be lighter arch variants for ensemble or doubling down on chains of small models.

### 2026-04-28 — iter19: askeladd-arch (256/6/8 slice=96 mlp=4) → val_surf_p=64.78
- **Hypothesis:** Askeladd jumped to 34.56 on the leaderboard. Their checkpoints (apr27-4/askeladd/checkpoints/) all use n_hidden=256, n_layers=6 (mostly), n_head=8, slice_num=96, **mlp_ratio=4** (vs my mlp=2). Bigger MLPs in each transformer block. Try this architecture.
- **Change:** Added `--mlp_ratio` CLI flag (was hardcoded). Run from-scratch with `--n_hidden 256 --n_layers 6 --n_head 8 --slice_num 96 --mlp_ratio 4 --epochs 25`. 4.6M params (2.7x baseline). Run `d0ipjrs5`.
- **Result:** stopped at epoch 21 (timeout 29.1 min, 18.9GB peak). val/loss=0.8932, surf_p=**64.78** (worse than iter1 baseline). Optimizer added 0 weight to ensemble — bigger arch alone doesn't beat well-trained smaller chains. Submitted at `da0cdb3`.
- **Verdict:** kept; need to chain it next iter to actually realize the capacity advantage.
- **Notes:** Askeladd has 13 trained checkpoints — they're chaining and ensembling like me but with a bigger backbone. Plan iter20 = warm-chain iter19.

### 2026-04-28 — iter18: deeper chain on iter17 (lr=5e-6) → val_surf_p=52.80, ensemble val=48.90
- **Hypothesis:** Continue chain on iter17 (53.31). Adds another chain checkpoint to the optimizer pool. Pattern is robust now: every chain is worth ~-0.07 ensemble val.
- **Change:** `--warm_start model-q636386k --lr 5e-6 --epochs 11`. Run `cxsc8wnv`.
- **Result:** epoch 11, val/loss=**0.7478**, surf_p=**52.80** (-1% vs iter17's 53.31). Ensemble (16 sources) val avg_surf_p=**48.90** (-0.06 from 48.96). New top weights: iter15=0.25, iter9=0.24, iter3=0.20, iter18=0.16, iter11=0.12. Submitted under `83a1336`.
- **Verdict:** kept; another -0.06.

### 2026-04-27 — iter17: warm-chain iter16 → val_surf_p=53.31, ensemble val=48.96
- **Hypothesis:** Chain on the seed-diverse iter16 (55.57). Warm-chains have proven valuable in the optimizer (iter9, iter11, iter15 all weighted heavily). Adding a chain from a different seed should give a useful new perspective.
- **Change:** Warm-start iter16 (`model-wm65na34`), bs=2 + full mesh + lr=2e-5 + p_weight=3 + 11 epochs. Run `q636386k`.
- **Result:** epoch 11 best, val/loss=0.7548, surf_p=**53.31** (-4% vs iter16's 55.57). Optimizer (15 sources) val avg_surf_p **48.96** (-0.08 from 49.04). Top weights: iter15=0.28, iter9=0.25, iter3=0.19, iter17=0.13, iter11=0.12.
- **Verdict:** kept; submitted under `d096d3b`.
- **Notes:** Adding chains > adding fresh seeds for ensemble. Each chain run buys -0.07 to -0.10 val. The marginal cost is one full 30-min training session per chain step.

### 2026-04-27 — iter16: another seed (PYTHONHASHSEED=123) Fourier model + 9-way ensemble val=49.04
- **Hypothesis:** More seed-diverse Fourier models → larger effective ensemble. Same recipe as iter1 with PYTHONHASHSEED=123. Run `wm65na34`.
- **Result:** epoch 29 best, val/loss=0.7820, surf_p=**55.57**. Optimizer (now 14 sources) found val avg_surf_p **49.0383** (-0.07 from 49.12 → 49.04). New top weights: iter15=0.21, iter11=0.20, iter9=0.18, iter3=0.18, iter14=0.05, iter16=0.02, iter5/iter1/iter2/iter8 trace.
- **Verdict:** kept; submitted under `832307e`. Marginal improvement.
- **Notes:** Single-seed re-runs give diminishing returns (-0.04 each). Need bigger architectural moves to break through.

### 2026-04-27 — iter15: deeper chain on iter3 (lr=1e-6) → val_surf_p=51.23 + 8-way ensemble val=49.05
- **Hypothesis:** iter3 holds 38% of optimal ensemble weight. Refining iter3 directly should give the biggest single-model improvement. Chain it at lr=1e-6 for 11 epochs (the deepest yet on the Fourier branch).
- **Change:** Warm-start iter3 (`model-w2qvsfx1`), `lr=1e-6`, bs=2 full mesh. Run `n3rus7ko`.
- **Result:** epoch 11 best, val/loss=**0.7275**, surf_p=**51.23** (vs iter3's 51.27). Marginal solo gain. But ensemble optimizer with iter15 added improves: 49.12 → **49.05** with new weights {iter3=0.26, iter15=0.26, iter9=0.25, iter11=0.12, iter14=0.06, iter2=0.02, iter8=0.01, iter1=0.003}. Submitted under `8e5d14f`.
- **Verdict:** kept; -0.07 ensemble val improvement.
- **Notes:** iter15 is essentially a near-clone of iter3 (deeper chain) but contributes equal weight in the optimal ensemble — the small parameter delta still improves the blend. Best-single iter15 = 51.23, slightly better than iter3.

### 2026-04-27 — iter14: seed-diverse Fourier model + 6-way validated ensemble → val=49.12
- **Hypothesis:** Optimizer dropped non-Fourier models. Seed diversity within Fourier+chain family might help. Train another iter1-style model (192/6/6, fourier=8, p_weight=3, 30ep) with `PYTHONHASHSEED=42`.
- **Change:** Re-run iter1 recipe. Run `o4n26g12`. Submitted ensemble at HEAD via fast_optimize.py with 12 sources.
- **Result:** iter14 single val_surf_p ~ 56 (worse than iter3's 51), but adds 0.06 weight to optimal ensemble. Best val avg_surf_p **49.12** (from 49.16 with 5-way → 49.12 with 6-way). Final weights: iter3=0.38, iter9=0.23, iter11=0.17, iter2=0.16, iter14=0.06, iter1=0.005. Submitted under `8871171`.
- **Verdict:** kept; -0.04 marginal improvement from added diversity. Iter14 is a "ensemble tax" — single model is weak but helps the whole.
- **Notes:** Optimizer iterates fast (<10 sec for 11 sources). Path forward: more seeds? Try TTA (flip Re axis)? Continue best-chain (iter3 → iter11 chain)?

### 2026-04-27 — iter13: validation-grid ensemble optimization → val avg_surf_p=49.16
- **Hypothesis:** Hand-picked weights are likely suboptimal. Compute val predictions for every model, then greedy-search ensemble weights to minimize avg_surf_p directly. Frieren did this kind of search and squeezed 35.27 → 34.41 (3% gain).
- **Change:** New `fast_optimize.py` that pre-stacks per-(model,sample-on-surface) pressure predictions into a `[K, T]` GPU tensor and evaluates ensemble metric in O(KT). Ran with all 11 single-model checkpoints (iter1-3, iter5-12).
- **Result:** Single-model val surf_p (weighted L1):
  - iter3 51.27 (best), iter2 51.87, iter1 53.46, iter11 53.59, iter9 53.81, iter8 54.71, iter6 57.51, iter10 58.06, iter7 58.09, iter12 62.81, iter5 66.41
  - Top-5 (iter3+iter2+iter1+iter11+iter9) inv-weighted: 49.50
  - After greedy refinement: **49.16** with weights iter1=0.005, iter2=0.20, iter3=0.39, iter9=0.24, iter11=0.17
  - Bigger / vanilla branches (iter5,6,7,8,10,12) all dropped — too distinct from the dominant Fourier+chain family. Submitted under `d36d8cc`.
- **Verdict:** kept; -2.1 vs hand-tuned 9-way (51.27 single → 49.16 ensemble val, but the ensemble doesn't include the diversity branches).
- **Notes:** `optimize_ensemble.py` (greedy from Python lists) was too slow (no GPU). `fast_optimize.py` is the keeper. Best-single still iter3.

### 2026-04-27 — iter12: deeper 224d/8L/8H from scratch → val/loss=0.87, avg_surf_p=62.81
- **Hypothesis:** Capacity ceiling in 192/6/6. Try 224/8/8 (3M params, 1.7x iter1) for more representation power. Same proven recipe (subsample=40k, p_weight=3, 25 epochs).
- **Change:** `--n_hidden 224 --n_layers 8 --n_head 8 --slice_num 64 --fourier_scales 8 --epochs 25`. Run `62mgdajz`.
- **Result:** stopped at epoch 22 (timeout 29.8 min, 16.9GB). val/loss=**0.8739**, avg_surf_p=**62.81** — worse than iter1 baseline. Likely undertrained.
- **Verdict:** kept for ensemble; not a winning single. **Leaderboard at this point: edward rank #4 (42.27 with 9-way ensemble) vs top 3 bunched at 39.5–39.8.**
- **Notes:** Single best is still iter3 (51.27). Plan: lean down the ensemble (drop iter5, iter12) and weight iter3 heavier; consider warm-chaining iter12 to recover its capacity, or compute validation predictions and grid-search ensemble weights.

### 2026-04-27 — iter11: deepest vanilla chain (lr=1e-6) → val/loss=0.74, avg_surf_p=53.59
- **Hypothesis:** iter9 deeper chain (lr=5e-6) reached 53.81. Lower the lr further to 1e-6 for one more refinement step.
- **Change:** Warm-start from iter9 (`model-g6g19fjy`), `lr=1e-6`, 11 epochs, otherwise identical to iter9. Run `3xi0kuhu`.
- **Result:** epoch 10/11 best, val/loss=**0.7380**, avg_surf_p=**53.59**. 28.7 min, 29.1GB. Auto-submit succeeded.
- **Verdict:** kept; minor improvement (-0.4%). Vanilla chain has fully converged. 10-way ensemble (final weights 0.04/0.07/0.18/0.04/0.07/0.08/0.16/0.18/0.07/0.11) submitted under `505ec04`.
- **Notes:** Best single still iter3 (51.27). Vanilla chain plateaued around 53.5. Diversity for ensemble largely tapped.

### 2026-04-27 — iter10: more Fourier (16 scales, max_freq=32) → val/loss=0.81, avg_surf_p=58.06
- **Hypothesis:** Original iter1 used 8 Fourier scales with max_freq=16. Try doubling both (16 scales, max_freq=32) for finer-grained spatial frequencies — possibly captures small-scale turbulence near surfaces.
- **Change:** From-scratch run identical to iter1 except `--fourier_scales 16 --fourier_max_freq 32.0`. Run `djx8kgjc`.
- **Result:** epoch 30/30, 24.2 min, 10.6GB. val/loss=**0.8084**, avg_surf_p=**58.06**. Worse than iter1's 53.46 with 8 scales — more Fourier features hurt or just used capacity inefficiently.
- **Verdict:** kept for ensemble diversity. 9-way ensemble (weights 0.05/0.07/0.20/0.04/0.07/0.10/0.18/0.20/0.09) submitted under `b61ffd1`.
- **Notes:** Higher freqs may have over-fit positional encoding without enough data. Best single still iter3 (51.27). Diminishing returns on this branch.

### 2026-04-27 — iter9: deeper vanilla chain (lr=5e-6) → val/loss=0.74, avg_surf_p=53.81 (-2%)
- **Hypothesis:** iter8 (vanilla + bs=2 chain at lr=2e-5) reached 54.71. Continue chain at lr=5e-6 for one more 11-epoch round to squeeze more out of the vanilla branch and add another ensemble checkpoint.
- **Change:** Same as iter8 but warm-starting from iter8 (`model-5cqn03sv`) and `lr=5e-6`. Run `g6g19fjy`.
- **Result:** epoch 10/11 best, val/loss=**0.7405**, avg_surf_p=**53.81** (single=0.74, rc=0.99, cruise=0.45, re_rand=0.78). 28.7 min, 29.1GB. Auto-submit succeeded.
- **Verdict:** kept; mild gain (-2%). 8-way ensemble (weights 0.05/0.08/0.22/0.05/0.08/0.10/0.20/0.22 across iter1+2+3+5+6+7+8+9) submitted under `68be8bf`.
- **Notes:** Best individual model is still iter3 (Fourier+chain, 51.27). Next radical move could be a different architecture entirely (e.g., a small GNN-style local model) or larger Fourier scales (16 or 32 freqs) for finer turbulence detail.

### 2026-04-27 — iter8: warm-chain iter7 (vanilla, no Fourier) bs=2 + full mesh + p_weight=3 → val/loss=0.75, avg_surf_p=54.71
- **Hypothesis:** iter7 (vanilla, no Fourier, p_weight=1) had val/loss=0.6692 but high surf_p=58.09 because pressure wasn't weighted. Re-warm with the proven recipe to push pressure down. Adds another diverse base for the ensemble (vanilla architecture, p_weight=3 fine-tune).
- **Change:** `python train.py --warm_start /mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-gp7ce0vi/checkpoint.pt --fourier_scales 0 --batch_size 2 --train_subsample 0 --lr 2e-5 --p_weight 3.0 --epochs 11 --warmup_epochs 1`. Run `5cqn03sv`.
- **Result:** epoch 11/11, 28.6 min, 29.2GB peak. val/loss=**0.7501**, avg_surf_p=**54.71** (single=0.75, rc=1.00, cruise=0.46, re_rand=0.79). Auto-submit succeeded.
- **Verdict:** kept; surf_p improved 58→55 with chain (-6%), and the model adds vanilla-architecture diversity to the ensemble. Best single is still iter3 at 51.27 — the Fourier features remain a small but real win on the surf_p metric.
- **Notes:** Full ensemble now spans (Fourier+p_weight=3 chain) iter1-3, (bigger 256/8 + p_weight=1 chain) iter5-6, (vanilla no-Fourier + p_weight={1,3}) iter7-8. 7-way ensemble (weights 0.05/0.10/0.25/0.05/0.10/0.15/0.30 favoring iter3 and iter8) submitted under `576b3c5`. With 7 models spanning 3 architectural families, ensemble should give a real improvement over best single.

### 2026-04-27 — iter7: vanilla Transolver, no Fourier, p_weight=1, 30 epochs → val/loss=0.67, avg_surf_p=58.09
- **Hypothesis:** All five prior single models share Fourier features; ensemble diversity is lower than I'd like. Train a fully vanilla Transolver (`fourier_scales=0`) with p_weight=1 (balanced 3-channel L1) to get an architecturally distinct base model.
- **Change:** Made `FourierFeatures` opt-in (`model.py`: `fourier_scales=0` disables and the model takes raw 24 features). Run with `--fourier_scales 0 --p_weight 1.0 --epochs 30`. Run `gp7ce0vi`.
- **Result:** epoch 30/30 finished in 24.1 min (10.5GB peak). val/loss=**0.6692** (best across iter1-7 because no p_weight inflation), avg_surf_p=**58.09** (single=0.64, rc=0.91, cruise=0.43, re_rand=0.70). Higher surf_p than iter3 (51.27) because no p_weight focus, but lower overall val/loss → useful for ensembling. Auto-submit predict.py succeeded (GPU cleanup fix from prev iter held).
- **Verdict:** kept; higher diversity for the ensemble. Final 6-way ensemble (iter1+2+3+5+6+7 weighted 0.1/0.1/0.3/0.1/0.15/0.25) submitted under `886258e`.
- **Notes:** Surprised that vanilla model converged faster (24 min vs 28 min for iter1) — Fourier features add wallclock overhead. Best-single still iter3 at 51.27. The 6-way ensemble is the strongest submission. Plan iter8 = warm-chain iter7 with bs=2 + full-mesh + p_weight=3 to sharpen its pressure score.

### 2026-04-27 — iter6: warm-chain iter5 (256/6/8) on full mesh → val/loss=0.78, avg_surf_p=57.51 (-13%)
- **Hypothesis:** Apply the same bs=2 + train_subsample=0 + lr=2e-5 + p_weight=3 chain that took iter1 from 53→51 to the bigger iter5 model (66.41 surf_p). The bigger model has more capacity and may benefit more from the full-mesh fine-tune. Adds further ensemble diversity to iter1+iter2+iter3+iter5.
- **Change:** `python train.py --warm_start /mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-rh634jvn/checkpoint.pt --n_hidden 256 --slice_num 128 --n_head 8 --batch_size 2 --train_subsample 0 --lr 2e-5 --p_weight 3.0 --epochs 9 --warmup_epochs 1`. Run `bxq8ylix`.
- **Result:** stopped at epoch 6 (timeout 30.2 min, 56.4GB peak — 5min/epoch with the bigger model on full mesh). val/loss=**0.7834**, avg_surf_p=**57.51**. Per-split val/loss: single=0.77, rc=1.03, cruise=0.50, re_rand=0.83. Auto-submit predict.py succeeded this time (added `del model; torch.cuda.empty_cache()` before subprocess fork).
- **Verdict:** kept; -13% on iter5's surf_p, but still well above iter3 (51.27). Useful for the ensemble.
- **Notes:** Bigger model is much more memory-hungry (56GB vs 30GB). Final 5-way ensemble (iter1+iter2+iter3+iter5+iter6 weighted 0.1/0.15/0.4/0.15/0.2) submitted under `6b96924`. Best single still iter3 at 51.27; ensemble should clean up some single-model noise.

### 2026-04-27 — iter5: bigger model 256/6/8 slice=128, p_weight=1, from scratch → val/loss=0.76, avg_surf_p=66.41
- **Hypothesis:** The chain plateau (iter3=51.27) suggests a capacity/diversity ceiling for the 192/6/64 architecture. Train a bigger, distinct model from scratch (n_hidden=256, slice_num=128, n_head=8, p_weight=1 instead of 3) to add ensemble diversity.
- **Change:** Configurable model dims via CLI in `train.py`; predict.py reads `config.yaml` from the checkpoint dir. Run with `--n_hidden 256 --slice_num 128 --n_head 8 --p_weight 1.0 --epochs 28`.
- **Result:** stopped at epoch 19 (timeout 29.5 min, 20.2GB peak). val/loss=**0.7618**, avg_surf_p=**66.41** (single=0.75, rc=1.00, cruise=0.51, re_rand=0.79). Worse than iter1-3 single models, but trained with very different objective (no p_weight) → high ensemble diversity. 3.04M params (1.8x iter1).
- **Verdict:** kept for ensemble; surf_p alone wouldn't beat anything but diversity matters. Final ensemble (iter1+iter2+iter3+iter5 weighted 0.15/0.2/0.4/0.25) submitted under `b3f1560`.
- **Notes:** 28-epoch target was too ambitious — only got 19, schedule didn't finish cosine. Auto-submit predict.py died with OOM (something else briefly grabbed 88GB), ran predict manually after kill. Plan iter6: continue iter5 chain with bs=2 + full mesh + p_weight=3 to drive its solo score down, then 5-way ensemble.

### 2026-04-27 — iter4: 3-way ensemble of iter1+iter2+iter3 (weights 0.2/0.3/0.5)
- **Hypothesis:** Even chain-correlated models can give a small boost when blended (frieren saw 35.27→34.41 with 6-way ensembles). Worth a quick win while a fresh diverse model trains.
- **Change:** New `ensemble.py` reading per-split prediction tensors from N source commits and averaging. Submitted with weights 0.2/0.3/0.5 favoring iter3 (best single model).
- **Result:** Ensemble predictions written to `/mnt/new-pvc/predictions/apr27-4/edward/3a1cf7a/`. Best single model val/loss = 0.728; ensemble val score not measured directly (no val labels for test ensemble).
- **Verdict:** kept as a hedge; will compare scores once leaderboard refreshes.
- **Notes:** All three sources are warm-chained from iter1, so they're highly correlated — expect modest gain. Plan iter5 = train a fundamentally different model (n_hidden=256, slice_num=128, p_weight=1) for genuine ensemble diversity.

### 2026-04-27 — iter3: deeper warm-start chain (lr=5e-6) → val/loss=0.728, avg_surf_p=51.27 (-1%)
- **Hypothesis:** iter2's warm-start chain dropped surf_p only ~3% (53.46→51.87). Frieren's chain dropped ~5 surf_p per iter; mine is converging faster. Try one more chain step at `lr=5e-6` (4x lower) for 10 epochs to squeeze the last gains before pivoting.
- **Change:** `python train.py --warm_start /mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-yqqu10sg/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 5e-6 --p_weight 3.0 --epochs 10 --warmup_epochs 1`. Run `w2qvsfx1`.
- **Result:** epoch 10/10, val/loss=**0.7280**, **avg_surf_p=51.27** (single=0.70, rc=1.00, cruise=0.45, re_rand=0.77). 26.2 min wall, 29.3GB peak. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/5d05ebb/`. Auto-submit failed mid-run with OOM (something else briefly took 88GB on the GPU); ran predict.py manually after killing the orphan train workers.
- **Verdict:** kept — marginal 1% gain, chain has plateaued.
- **Notes:** Likely the Fourier features made iter1 already extract spatial info that frieren's chain unlocked between iters. To get to leaderboard-leader territory (~42 surf_p, possibly 35 with ensembles), need a fundamentally different angle. Plan iter4 = try a fresh, larger model (n_hidden=256, slice_num=128, n_layers=8) trained from scratch. If it lands near iter3, switch to ensembling iter1+iter2+iter3 predictions.

### 2026-04-27 — iter2: warm-start bs=2 full-mesh chain → val/loss=0.74, avg_surf_p=51.87 (-3%)
- **Hypothesis:** Frieren's apr27 iter2 found that warm-starting iter1 with `batch_size=2 + train_subsample=0` + `lr=2e-5` + `p_weight=3` for 10 epochs got 49.34 surf_p (from 54.26). The hypothesis is that the 40k subsample drops 60% of the volume mesh — refining on the full mesh at low LR teaches the model the spatial structure it missed.
- **Change:** `python train.py --warm_start /mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-77aydpcl/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --p_weight 3.0 --epochs 12 --warmup_epochs 1`. Run `yqqu10sg`.
- **Result:** stopped at epoch 11 (timeout 28.7 min, 29.3GB peak). Best val/loss=**0.7358**, **avg_surf_p=51.87** (single=0.71, rc=1.01, cruise=0.45, re_rand=0.77). All four splits improved a bit vs iter1. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/0fa22ab/`.
- **Verdict:** kept — modest 3% gain on surf_p, 2% on val/loss. Single-foil split still has the highest absolute val loss (0.71).
- **Notes:** The chain is converging slowly — frieren's chain dropped 4-5 surf_p per iter, mine dropped only 1.6. Might be that my Fourier features make iter1 already closer to convergence. Next: try a larger model (n_hidden=256, slice_num=128) or push lr down to 5e-6 and continue chaining for marginal gains.

### 2026-04-27 — iter1: proven recipe (Transolver 192/6/6 + Fourier features) → val/loss=0.75, avg_surf_p=53.46
- **Hypothesis:** Replicate frieren's apr27 proven recipe (Transolver n_hidden=192, n_layers=6, n_head=6, slice=64, L1 loss, p_weight=3, train_subsample=40k volume nodes, bf16, bs=4, lr=5e-4, surf_weight=10, warmup+cosine over 35 epochs). Add Fourier features for position (8 scales, max_freq=16) and attention masking on padded tokens — both unused by previous leaders. Ought to land near 50-55 surf_p with the Fourier features potentially helping high-frequency turbulence features.
- **Change:** New `train.py` + `model.py`. `model.py` exposes `Transolver` with `FourierFeatures` (sin/cos at 8 log-spaced freqs over normalized x,z) and `mask` plumbed through `PhysicsAttention` so padded tokens can't pollute slice softmax. `train.py` adds `SubsampledDataset` (keep all surface nodes, randomly subsample 40k volume), L1 channel-weighted loss with `p_weight=3` on the pressure channel, AdamW + warmup-cosine LR, grad-clip 1.0, bf16 autocast, automatic checkpoint mirror to `/mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-<id>/`, and to `checkpoints/best.pt`.
- **Result:** epoch 35/35, val/loss=0.7537, **avg_surf_p=53.46** (single=0.74, rc=1.02, cruise=0.47, re_rand=0.79). 28.2 min wall, 10.5GB peak. W&B run `77aydpcl`. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/79894e7/`.
- **Verdict:** kept — solid baseline, slightly better than frieren's apr27 iter1 (54.26).
- **Notes:** Loss still dropping at epoch 35 (cosine bottom). Plan iter2 = warm-start this checkpoint with bs=2, full mesh, lr=2e-5, p_weight=3, ~10 epochs. Frieren's apr27 iter2 dropped 54.26→49.34 with this recipe; expect similar gain. predict.py crashed first due to importing `train.py` at top-level (sp.parse fired); fixed by moving model into `model.py`.
