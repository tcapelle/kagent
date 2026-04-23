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
