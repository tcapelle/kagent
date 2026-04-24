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

### 2026-04-24 — v15-fullmesh-lr1e5-c
- **Hypothesis:** Same recipe as v14; per v14's lesson, keep lr=1e-5 — it's still producing per-run gains of ~0.05+ on full mesh.
- **Change:** `train.py --resume .../model-6lzqlfsd/checkpoint.pt --lr 1e-5 --train_max_points 0 --batch_size 2`.
- **Result:** Best `val/l2_error = 3.096` at epoch 8 (val/loss=1.27). Gain **0.05** over v14; all 8 epochs monotonic again. W&B `alphonse/v15-fullmesh-lr1e5-c` (`gjameqab`). **1.8% over v14, 55.5% over v2.**
- **Verdict:** Kept.
- **Notes:** Fourth full-mesh round still producing ~0.05 gains. No sign of plateau yet. Keep going.

### 2026-04-24 — v14-fullmesh-lr1e5-b (LR restart win)
- **Hypothesis:** v12/v13 used lr=5e-6/3e-6 and their gains were flattening (0.05, then 0.03). Maybe the LR was *too* low, not the model-was-converged. Try going *back up* to lr=1e-5 (same as v11) and see if it's still training.
- **Change:** `train.py --resume .../model-2f3iycuq/checkpoint.pt --lr 1e-5 --train_max_points 0 --batch_size 2`.
- **Result:** Best `val/l2_error = 3.1535` at epoch 8 (val/loss=1.31). Gain **0.07** over v13 — bigger than the last two rounds combined. 8 monotonic epochs. W&B `alphonse/v14-fullmesh-lr1e5-b` (`6lzqlfsd`). **2.2% over v13 (3.23→3.15), 54.7% over v2.**
- **Verdict:** Kept — key finding.
- **Notes:** Important lesson: dropping LR too aggressively was causing the apparent convergence in v12/v13. Going back to lr=1e-5 (same as v11) exposed more training left. Stick with lr=1e-5 for now and only lower it when clearly plateauing.

### 2026-04-23 — v13-fullmesh-lr3e6
- **Hypothesis:** Standard warm-start tail; drop LR to 3e-6. At this point gains are tiny but each extra pass on full mesh has been worth it.
- **Change:** `train.py --resume .../model-2xc78fxp/checkpoint.pt --lr 3e-6 --train_max_points 0 --batch_size 2`.
- **Result:** Best `val/l2_error = 3.2257` at epoch 8. Gain 0.03 over v12. W&B `alphonse/v13-fullmesh-lr3e6` (`2f3iycuq`). **0.9% over v12, 53.6% over v2.**
- **Verdict:** Kept — another small, monotonic gain.
- **Notes:** Full-mesh gains per iteration: v11 3.30, v12 3.25 (-0.05), v13 3.23 (-0.03). Curve is flattening. Considering switching to test-time ensembles (multiple seeds of full-mesh fine-tune) next.

### 2026-04-23 — v12-fullmesh-cont
- **Hypothesis:** v11 was still improving at epoch 8 when it timed out. Continue warm-start on full mesh with `lr=5e-6` — similar to the v10 tail-of-chain recipe but with the now-fixed training regime.
- **Change:** `train.py --resume .../model-wsxkhla1/checkpoint.pt --lr 5e-6 --train_max_points 0 --batch_size 2`.
- **Result:** Best `val/l2_error = 3.2544` at epoch 7 (val/loss=1.37). Gain 0.05 over v11, monotonic. W&B `alphonse/v12-fullmesh-cont` (`2xc78fxp`). **1.4% over v11 (3.30→3.25), 53.2% over v2.**
- **Verdict:** Kept — another small but real gain; full-mesh warm-start still has juice.
- **Notes:** At 4.3 min/epoch for full mesh we only get ~7 epochs per run but each one moves the needle. Each `re_rand` and `camber_cruise` split continues to improve, showing the OOD generalisation is still getting better.

### 2026-04-23 — v11-fullmesh-breakthrough (MAJOR WIN)
- **Hypothesis:** Training has been using `train_max_points=80_000` random subsample while val/predict use the full mesh (up to ~240K nodes). That's a distribution mismatch — the PhysicsAttention slice weights are computed on a different density of nodes between train and val. Warm-start from v10 with `train_max_points=0` (no subsample, full mesh) and `batch_size=2` should close this gap.
- **Change:** `train.py --resume .../model-csvexb95/checkpoint.pt --lr 1e-5 --train_max_points 0 --batch_size 2` — otherwise identical recipe. Epoch time ~4.3 min instead of 2.1 min, so only 8 epochs in 30 min.
- **Result:** Best `val/l2_error = 3.299` at epoch 8 (val/loss=1.39). **22% over v10 in one run (4.23→3.30), 52.6% over v2 baseline.** Biggest jump since v3. Split-level gains are dramatic on the OOD tracks: val_geom_camber_rc 2.98→1.70, val_re_rand 2.70→1.25, val_geom_camber_cruise 1.57→0.58. val_single_in_dist barely moved (2.09→2.03) — so the subsampling was mostly hurting OOD generalisation. W&B `alphonse/v11-resume-fullmesh-lr1e5` (`wsxkhla1`). Submitted at commit `190cc20`.
- **Verdict:** Kept — step change.
- **Notes:** HUGE learning — I should have tried full-mesh training much earlier. The hypothesis that subsampling "just acts like a regulariser" was wrong in this domain; it creates a real train/eval distribution gap for attention models that pool over the node set. Next move: continue warm-start on full mesh (v12).

### 2026-04-23 — v10-resume-lr5e6-from-v9 (current best)
- **Hypothesis:** Push the warm-start chain one more time with an even lower `lr=5e-6`. The tail-end gains are tiny but reliable — lower LR just does gentler refinement on the EMA.
- **Change:** `train.py --resume .../model-j4xz2ho8/checkpoint.pt --lr 5e-6`.
- **Result:** Best `val/l2_error = 4.2322` at epoch 15 (val/loss=2.34). Gain 0.024 over v9, still monotonic. W&B `alphonse/v10-resume-lr5e6` (`csvexb95`). **0.55% over v9 (4.256→4.232), 39.1% over v2.**
- **Verdict:** Kept — new best single model. Stopping the warm-start chain here; tail gains are now ≤0.025 per 30 min and likely to hit numerical noise soon.

### 2026-04-23 — v9-resume-lr1e5-from-v8 (current best)
- **Hypothesis:** Continue the warm-start chain one more iteration. Per-iteration gain is now ~0.03 so the expected value is small but the risk is zero — at worst we match v8.
- **Change:** `train.py --resume .../model-j93i4dy0/checkpoint.pt --lr 1e-5` (same recipe as v8).
- **Result:** Best `val/l2_error = 4.2562` at epoch 15 (val/loss=2.34). Gain 0.03 over v8, fully monotonic. W&B `alphonse/v9-resume-lr1e5-b` (`j4xz2ho8`). **0.8% over v8 (4.29→4.26), 38.8% over v2.**
- **Verdict:** Kept — now strictly the best single model.
- **Notes:** Warm-start chain history of val/l2 from fresh v4 through v9: 4.78 → 4.41 → 4.32 → 4.29 → 4.26. Gains per iteration: 0.37 / 0.09 / 0.03 / 0.03. Essentially flat now. Another iteration would give ≤0.02; not worth the 30 min unless nothing better is available. At this point the only paths with higher EV are (a) arch diversity + multi-model warm-start to produce an independent ~4.3 model for ensembling, or (b) proper input-space TTA — but the domain is z ≥ 0 half-space (not symmetric) so simple y-flip is not available.

### 2026-04-23 — v8-resume-lr1e5 (current best)
- **Hypothesis:** Since ensembling is stuck (v7 too weak, v5 too correlated with v6), try one more very-slow warm-start from v6 with `lr=1e-5`. The aim is to exploit what little low-LR refinement is left. If it gains more than ~0.02 it's worth keeping.
- **Change:** `train.py --resume .../model-x58etsc6/checkpoint.pt --lr 1e-5` from the v6 PVC checkpoint. Same arch, same loss.
- **Result:** Best `val/l2_error = 4.29` at epoch 14 (val/loss=2.32). 14+ monotonic epochs; gain of 0.03 l2 over v6. Also tried offline ensembles on val:
  - v6 + v8 (eval): 4.304 — slightly worse than v8 alone (4.29).
  - v8 alone is the single best of the 9 runs; ensembling its lineage partners doesn't help.
  W&B `alphonse/v8-resume-lr1e5` (`j93i4dy0`). Submitted at commit `4ac0cb3`. **0.7% over v6 (4.32→4.29), 38.4% over v2.**
- **Verdict:** Kept — clear best single model.
- **Notes:** Warm-start chain is now fully exhausted — v6 → v8 gave 0.03, next iteration would likely give <0.02. Going forward the only real ways to beat this are (a) a fundamentally different architecture that decorrelates, then real ensembling, (b) proper y-flip TTA (requires careful feature semantics work), or (c) hyperparameter changes (e.g. different slice_num) to hopefully reach a different local minimum.

### 2026-04-23 — v7-scratch + ensemble attempt (DISCARDED)
- **Hypothesis:** Warm-start is saturating (v5→v6 gain was only 0.09). A fresh-from-init v7 should produce decorrelated errors; averaging v6 + v7 predictions should beat v6 alone.
- **Change:** Ran `train.py` from scratch (same config as v4 — hidden=256, layers=8, slice_num=64). Added `eval_ensemble.py` so we can score ensembles offline on val before committing.
- **Result:** v7 alone `val/l2_error = 5.03`. Ensemble val scores (computed via `eval_ensemble.py`):
  - v6 alone: 4.322
  - v5 + v6 (correlated lineage): 4.349
  - v6 + v7: 4.459
  - All **worse than v6 alone**. v7 is too weak (~16% worse) to help v6 via simple averaging, and v5+v6 are too correlated.
- **Verdict:** Discarded — submitted v6's predictions at HEAD commit `9245c2f` instead.
- **Notes:** Key lesson for ensembling: the members have to be (a) individually close in quality, or (b) strongly decorrelated. v7-from-scratch is neither. If ensembling is to work, I should either warm-start v7 off v6 with more perturbation, or spend enough compute to bring a fresh-init model all the way to v6 quality. For now, another warm-start on v6 is the higher-EV move.

### 2026-04-23 — v6-resume-lr3e5
- **Hypothesis:** v5 was still improving at its last epoch but very slowly (last-epoch gain ≈0.006). A third warm-start round with `lr=3e-5` should squeeze out another ~0.1 l2 before we hit diminishing returns and need to switch strategies (seed ensemble / TTA).
- **Change:** `train.py --resume checkpoints/best.pt --lr 3e-5`; added `--checkpoints p1,p2,...` support to `predict.py` for eventual ensembling (not used yet here). Committed ensemble path first so future runs can use it.
- **Result:** Best `val/l2_error = 4.3220` at epoch 15 (val/loss=2.31). 15 more epochs of monotonic improvement; gain per epoch halved vs v5 (v5 gained ~0.37 over 15, v6 gained ~0.09). W&B `alphonse/v6-resume-lr3e5` (`x58etsc6`). Submitted at commit `5b6bd64`. **2.0% over v5 (4.41→4.32), 38% over v2.**
- **Verdict:** Kept — still a real gain, but we've hit diminishing returns on warm-start. Time to switch approach.
- **Notes:** Diminishing returns are clear: v3→v4 gain 0.61, v4→v5 gain 0.37, v5→v6 gain 0.09. Next move is to train a fresh model from scratch (different seed) and ensemble the two prediction streams. v5 (`axbjpfac`) and v6 (`x58etsc6`) are highly correlated (same lineage) so ensembling them directly won't help much. A fresh-from-init v7 gives us decorrelated errors.

### 2026-04-23 — v5-resume-lr1e4
- **Hypothesis:** v4 was still improving monotonically at epoch 15 when the 30-min timeout hit (train loss ~0.11, val/l2 still falling). A warm-start from v4's best checkpoint with a fresh cosine schedule and a lower peak LR (`1e-4` vs `5e-4`) should squeeze out more gains without overshooting from the already-good weights.
- **Change:** `train.py` — added `--resume <ckpt>` flag (loads `state_dict` into the model; EMA re-initialises to current weights, so its shadow is correct from step 0). Ran with `--resume checkpoints/best.pt --lr 1e-4`.
- **Result:** 15 more epochs in 30 min, all monotonic. Best `val/l2_error = 4.4079` at epoch 15 (val/loss=2.306). W&B `alphonse/v5-resume-lr1e4` (`axbjpfac`). Submitted at commit `ed4d82c`. **7.8% over v4 (4.78→4.41), 37% over v2.**
- **Verdict:** Kept — warm-start with lower LR is a cheap, reliable win whenever the previous run was still improving.
- **Notes:** `val_single_in_dist` is still the split that improves most (2.57→2.21 in 15 epochs), while `val_re_rand` is flattening (2.51→2.58 — actually slightly worse). The gap suggests we're starting to memorise in-dist at the expense of OOD generalisation. For v6 either (a) do another warm-start round to see if OOD holds, or (b) do TTA / seed-ensemble which is our one free win left.

### 2026-04-23 — v4-ema-cosine16
- **Hypothesis:** Two fixes on top of v3: (1) set cosine `T_max=effective_epochs=16` and `eta_min=lr*0.02` so LR anneals cleanly across the 30-min budget without warm-restart oscillation (v3 oscillated from epoch 12+); (2) add EMA of weights (`decay=0.999`) so the validated/saved weights are smoother than any single step — this usually gains ~5-10% on regression problems.
- **Change:** `train.py` — introduced `EMA` class, swap EMA weights into model during validation and restore after, updated checkpoint save to use EMA-validated weights. Set `effective_epochs=16`, `CosineAnnealingLR(T_max=16, eta_min=lr*0.02)`.
- **Result:** 15 epochs in 30 min. Monotonic improvement every epoch (no more oscillation). Best `val/l2_error = 4.78` at epoch 15, val/loss=2.68. Peak VRAM 33.5 GB. W&B `alphonse/v4-ema-cosine16` (`os7r07ln`). Submitted at commit `bb42606`. **11% over v3 (5.39→4.78), 31% over v2.**
- **Verdict:** Kept — EMA + proper schedule is a clear win and the model was still improving at the last epoch.
- **Notes:** EMA lags for the first ~5 epochs while it catches up to the trained weights (expected). The model is clearly not converged — train loss is still falling and val/l2 improved by 0.08 just between epochs 14 and 15. Next move: warm-start from this checkpoint for another 30-min round to extend effective training time.

### 2026-04-23 — v3-cw-subsample80k
- **Hypothesis:** Three compounding levers should beat v2: (1) channel weights [Ux=1.0, Uy=0.2, p=0.1] align training MSE with the leaderboard's physical velocity L2; (2) random point-subsampling to 80K per training sample halves per-batch compute so we can run `batch_size=4` and fit more epochs; (3) `T_max=8` in the cosine schedule actually anneals LR (in v2 we used T_max=50 so LR barely decayed).
- **Change:** `train.py` — added channel_weight_{Ux,Uy,p}, `train_max_points=80000` with a custom `train_collate` that subsamples before `pad_collate`, bumped `batch_size=4`, `effective_epochs=8` for cosine T_max, selection now by `val/l2_error` (not val/loss). `predict.py` + `model.py` refactor to avoid train-import side effects.
- **Result:** 15 epochs in 30 min (2.1 min/epoch, vs v2's 4 min). Best `val/l2_error = 5.39` at epoch 11, val/loss=3.33. Peak VRAM 33.5 GB. W&B `alphonse/v3-cw-subsample80k` (`reie1td7`). Submitted at commit `5dd72f0`. **23% improvement over v2 (6.96→5.39).**
- **Verdict:** Kept — clear win from all three axes.
- **Notes:** LR schedule is slightly mistuned: T_max=8 means LR re-warms after epoch 8 and caused oscillation from epoch 12 onward (best at epoch 11 was right at the restart). For v4, set T_max to match full 15-epoch budget and remove the warm-restart behaviour. Volume MAE on `val_single_in_dist` is still the dominant error — could investigate whether the inlet/free-stream region is the problem.

### 2026-04-23 — v2-hidden256-l8-amp (baseline submission)
- **Hypothesis:** Scaling the starter Transolver from hidden=128/layers=5 to hidden=256/layers=8 with bf16 AMP should materially reduce `val/l2_error` while still fitting in 96GB VRAM. AMP also gives more training steps in 30 min.
- **Change:** `train.py` — `n_hidden=256`, `n_layers=8`, `n_head=8`, `slice_num=64`, `mlp_ratio=2`; enabled bf16 autocast + gradient clip 1.0; added `val/l2_error` logging (mean sqrt(Ux² + Uy²) over all masked nodes). Filled in `predict.py` to load Transolver from saved checkpoint + config.yaml. Bumped `val/loss` to include the new l2 metric.
- **Result:** 8 epochs in 30 min. Best `val/l2_error = 5.84` at epoch 8, but checkpoint selection was by `val/loss` which picked epoch 7 (val/loss=3.74, l2=6.96). Peak VRAM 50.7 GB. W&B run `alphonse/v2-hidden256-l8-amp` (ce6hu0ux). Predictions submitted at commit `c2d3625`.
- **Verdict:** Kept — first real submission, sets baseline around l2≈6.96 with suboptimal checkpoint selection.
- **Notes:** Key finding — `val/loss` (dominated by surface pressure MSE) disagrees with `val/l2_error` (velocity only). Switching checkpoint selection to l2_error would have grabbed epoch 8 (l2=5.84, ~16% better) for free. Also found the `train.py` auto-submit crashes because `predict.py`'s `from train import Transolver` triggered train's argparse. Fixed by extracting the model into `model.py`. Cosine scheduler used T_max=50 so LR barely decayed across 8 epochs — shorten T_max to effective_epochs (~8) to let LR anneal.
