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

### 2026-04-27 — cosine35 (DISCARDED)
- **Hypothesis:** epochs=80 cosine never decays in our 30 min budget; setting epochs=35 lets cosine fully anneal to ~0 LR for a sharper minimum.
- **Change:** `epochs: 80 → 35`.
- **Result:** 32 epochs in 30.5 min. Best epoch 23: avg_surf_p=109.00 (much worse than iter8's 93.62). Training got worse after LR decayed too low — model couldn't keep improving.
- **Verdict:** discarded — `git reset --hard HEAD~1`. The high-LR-throughout schedule (epochs=80 with our actual ~32-epoch run) is correct for this setup; the model needs a non-zero LR all the way through.
- **Notes:** confirms our convergence is LR-limited not stuck-in-local-min. Don't reduce epochs/hyperparam for cosine.

### 2026-04-27 — noise05 (DISCARDED)
- **Hypothesis:** input noise was the iter8 win; pushing it from 0.03 → 0.05 should help further.
- **Change:** `input_noise=0.05`.
- **Result:** 32 epochs in 30.6 min. Best epoch 31: avg_surf_p=101.00 (worse than iter8's 93.62). val_geom_camber_rc=2.11 (worse than iter8's 1.98 too).
- **Verdict:** discarded — `git reset --hard HEAD~1`. 0.03 was the sweet spot; more noise destroys signal even on the OOD track.
- **Notes:** noise scaling is non-monotonic. Probably need feature-specific noise (less on positions, more on parametric features) to push further.

### 2026-04-27 — scale-224x8h8 (DISCARDED)
- **Hypothesis:** with stronger regularization from iter8 (input noise + slice128), bigger model (n_hidden 192→224, n_head 6→8) might use the budget better.
- **Change:** model_config n_hidden=224, n_head=8.
- **Result:** 25 epochs in 30.9 min (74s/epoch — bigger model). Best epoch 25: avg_surf_p=98.62 (worse than iter8's 93.62).
- **Verdict:** discarded — `git reset --hard HEAD~1`. Same lesson as iter4: extra width consumes the epoch budget without enough convergence; was still descending at the cap.
- **Notes:** epochs needed > available. Keeping iter8 architecture; next push will scale regularization not capacity.

### 2026-04-27 — input-noise-slice128
- **Hypothesis:** iter6/7 plateau driven by overfitting on val_geom_camber_rc (unseen-camber generalization). Add Gaussian input noise (σ=0.03 on normalized features) for cheap regularization, and bump slice_num 96→128 for more attention capacity.
- **Change:** train.py — `input_noise=0.03` config, applied per-step on real (non-padded) nodes; model_config slice_num=128.
- **Result:** 32 epochs in 30.5 min (57s/epoch). Best epoch 32: avg_surf_p=93.62. val_geom_camber_rc=1.98 (best of any iter so far, was 2.05 in iter5 and 2.47 in iter6). Predictions to `apr27-4/fern/23bf834`.
- **Verdict:** kept (commit `512e49c`). 5% gain over iter6 (98.53 → 93.62). Curve still descending at epoch 32. Now within 4 points of edward (90.12) but still behind nezuko (79.95) and far from thorfinn (45.94).
- **Notes:** input noise was the surprise win — confirms train data is the bottleneck and regularization helps generalization. Next: more noise or other augmentations (feature dropout / mixup), or revisit bigger model now that regularization is stronger.

### 2026-04-27 — warmstart-finetune (DISCARDED)
- **Hypothesis:** iter6 was still descending; warm-start from iter6 ckpt (98.53), lower LR (8e-4 → 2e-4), short warmup, sub25k. Squeeze more convergence.
- **Change:** added `--resume_from` to load checkpoint into both model and EMA at start. Ran with lr=2e-4 warmup_epochs=1 train_max_nodes=25000 epochs=60.
- **Result:** best epoch 1 (98.64 — basically the resumed weights). Then val_geom_camber_rc steadily worsened (2.47 → 3.33) as train loss dropped to vol=0.11/surf=0.04 — pure overfit. Final 33 epochs at avg_surf_p=100.24.
- **Verdict:** discarded — `git reset --hard HEAD~1`. Continued training without changing data/regularization just memorizes harder. Confirms train data is the bottleneck, not optimization.
- **Notes:** the previously-best ckpt is already near a local minimum w.r.t this loss/data; pushing further with same setup overfits. Need either augmentation, stronger regularization, or different architecture. Predictions still saved at `apr27-4/fern/f20ab62`.

### 2026-04-27 — sub20k-ema
- **Hypothesis:** iter5 was still descending at epoch 29; halving subsample 30k→20k buys more epochs.
- **Change:** train.py — `train_max_nodes=20000`.
- **Result:** 39 epochs in 30.7 min (47s/epoch). Best epoch 39: avg_surf_p=98.53 (vs 99.20 iter5). Predictions to `apr27-4/fern/0621c0b`.
- **Verdict:** kept (commit `d1cdf57`). Marginal 0.7% gain. val_geom_camber_rc actually got worse (2.05→2.47) — sparser per-sample meshes hurt camber generalization. Other splits compensated.
- **Notes:** plateau forming. Curve flattened in the last 10 epochs. Fundamental capacity may be at limit; need either bigger model or warm-start finetune. Trade-off found: smaller subsample → more epochs but lower per-sample diversity.

### 2026-04-27 — sub30k-ema
- **Hypothesis:** iter3 val curve was very wobbly (surf_p bouncing 113-148 in last 8 epochs); track exponential moving average of weights and evaluate that. Also subsample 40k→30k to fit more epochs (iter3 only got 24, was still descending).
- **Change:** train.py — `EMA(model, decay=0.999)` shadow updated every step; eval and checkpoint use EMA weights; `train_max_nodes=30000`.
- **Result:** 29 epochs in 30 min, monotonically descending surf_p (106→105→104→…→99.20). Best epoch 29: avg_surf_p=99.20. val/loss=1.27. Predictions to `apr27-4/fern/e28bf94`.
- **Verdict:** kept (commit `1a92bd1`). 11% gain over iter3 (111.34 → 99.20). Curve was still descending at the timeout — more epochs would help.
- **Notes:** train vol=0.21 surf=0.12 at epoch 29 — still room before train hits 0. Dropping subsample further (e.g. 20k) might claw back enough budget for many more epochs and push under 90. Per-split: val_geom_camber_rc=2.05 still the worst track.

### 2026-04-27 — scale-256x8-do10 (DISCARDED)
- **Hypothesis:** iter3 still showed train/val gap; bigger model (n_hidden=192→256, n_head=6→8) with stronger dropout (0.05→0.1) should improve generalization.
- **Change:** train.py model_config: n_hidden=256, n_head=8, dropout=0.1 (Fourier and slice_num kept).
- **Result:** epoch time 102s (vs 78s in iter3), only 18 epochs fit. Best epoch 18: avg_surf_p=115.84 (slightly worse than iter3's 111.34).
- **Verdict:** discarded — `git reset --hard HEAD~1`. Slight regression mainly because the bigger model needs more epochs but we're capped at 30 min; the extra dropout slowed convergence too much for the budget.
- **Notes:** Per-split: val_geom_camber_rc actually improved (1.95-2.03 vs 2.13). The bigger model has potential but the budget is the bottleneck. Next ideas: (a) keep iter3 base, add EMA for evaluation; (b) try wider but fewer-layer (n_hidden=256, n_layers=6) to keep epoch time near 78s; (c) train_max_nodes=30k to claw back more epochs.

### 2026-04-27 — fourier-do05
- **Hypothesis:** training curve plateaued in iter2 with ~3x train/val gap on val_geom_camber_rc — generalization is the bottleneck. Add multi-scale Fourier positional encoding for (x, z) (8 frequencies, 32 features) so the model gets richer spatial inductive bias; bump slice_num 64→96 for more attention capacity; small dropout 0.05.
- **Change:** model.py — `FourierPosEnc` module (sin/cos at log-spaced freqs up to 32π) wired into Transolver (`pos_freqs=8`); train.py — slice_num=96, dropout=0.05 in model_config.
- **Result:** 24 epochs in 31 min (78s/epoch — bigger slice and pos-enc cost). Best epoch 23: avg_surf_p=111.34. val_geom_camber_rc dropped from 4.04 (iter2) to 2.13 — Fourier features helped most on the unseen-camber generalization task. Predictions saved to `apr27-4/fern/3c5ceff`.
- **Verdict:** kept (commit `74475fa`). 21% improvement (141.50 → 111.34). Now competitive with edward (90.12) but still behind nezuko (79.95) and thorfinn (45.94).
- **Notes:** train still at 0.4 vol / 0.23 surf — still some overfit room. Next bet: scale capacity (n_hidden=256, maybe n_layers=10) with dropout=0.1 to keep regularization. Could also try EMA for evaluation since val curve is wobbly.

### 2026-04-27 — subsample40k
- **Hypothesis:** epoch time (245s) was capping training at 7 epochs; subsample each training sample to <=40k nodes (keep all surface + random subset of volume) to fit ~5x more epochs without losing surface coverage. Validation still uses full mesh.
- **Change:** custom train collate `make_train_collate(40000)` that subsamples per-sample before pad_collate; epochs raised to 80; freed model+optimizer before subprocess.run for predict.py to avoid GPU OOM.
- **Result:** 30 epochs in 30.7 min (61s/epoch, 4x faster). Best epoch 23: avg_surf_p=141.50. Predictions auto-saved to `apr27-4/fern/a626358`. W&B run `<id-from-log>`.
- **Verdict:** kept (commit `dea4c98`). 6% improvement on surf_p (151.21 -> 141.50). Confirms speed was the main bottleneck.
- **Notes:** train vol=0.39, surf=0.23 at epoch 30 vs val ~1.5-4.5 — overfitting starting. Worst split is val_geom_camber_rc (4-5) — generalization to unseen camber is the limiter. Curve plateaued by epoch 23. Next: Fourier positional features for (x,z), more slice tokens, and modest dropout/weight decay to widen generalization.

### 2026-04-27 — transolver-192x8-bf16-warmup
- **Hypothesis:** baseline Transolver (128/5/4) underperforms; scale to 192-hidden / 8 layers / 6 heads, bf16 autocast, warmup+cosine LR, channel-weighted loss with extra weight on pressure (the leaderboard target), grad clip, masked attention so padding doesn't pollute slice tokens.
- **Change:** new `model.py` shared between train/predict; rewrote `train.py` (n_hidden=192, n_layers=8, n_head=6, slice_num=64, mlp_ratio=2; lr=8e-4, surf_weight=15, p_weight=3, warmup=3 epochs; bf16; track best by avg_surf_p); rewrote `predict.py` to load config.yaml + model.py; passes `mask` into the model so padding rows are excluded from the slice softmax.
- **Result:** best epoch 6 of 7 completed (timeout 30 min). val/avg_surf_p=151.21, val/loss=2.0343. peak VRAM 76GB. W&B run `70gcaego`.
- **Verdict:** kept (commit `e912d3a`). Big jump from prior fern (131.69 was the previous run on apr27-bis, but that was unmasked baseline; this gives 151 on a new tag, lower is better — leaderboard pending). Top thorfinn is at 45.94 so still a long way to go.
- **Notes:** epoch time ~245s, only 6–7 epochs fit in 30 min — model is undertrained. Loss curve still descending. Val/loss kept dropping (2.03 at epoch 6) but surf_p MAE was non-monotonic. Next ideas: (1) keep bf16 but reduce n_layers to 6 to squeeze more epochs — convergence currently > capacity; (2) Fourier feature positional encoding for (x,z); (3) auxiliary surface-only head; (4) split loss into surf-pressure-heavy + vol; (5) EMA weights. Also: train.py auto-predict OOM'd because parent process still held GPU; ran predict.py manually after killing parent. Consider freeing model/cache before subprocess.
