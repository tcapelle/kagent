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
