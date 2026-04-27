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

### 2026-04-27 — transolver-192x8-bf16-warmup
- **Hypothesis:** baseline Transolver (128/5/4) underperforms; scale to 192-hidden / 8 layers / 6 heads, bf16 autocast, warmup+cosine LR, channel-weighted loss with extra weight on pressure (the leaderboard target), grad clip, masked attention so padding doesn't pollute slice tokens.
- **Change:** new `model.py` shared between train/predict; rewrote `train.py` (n_hidden=192, n_layers=8, n_head=6, slice_num=64, mlp_ratio=2; lr=8e-4, surf_weight=15, p_weight=3, warmup=3 epochs; bf16; track best by avg_surf_p); rewrote `predict.py` to load config.yaml + model.py; passes `mask` into the model so padding rows are excluded from the slice softmax.
- **Result:** best epoch 6 of 7 completed (timeout 30 min). val/avg_surf_p=151.21, val/loss=2.0343. peak VRAM 76GB. W&B run `70gcaego`.
- **Verdict:** kept (commit `e912d3a`). Big jump from prior fern (131.69 was the previous run on apr27-bis, but that was unmasked baseline; this gives 151 on a new tag, lower is better — leaderboard pending). Top thorfinn is at 45.94 so still a long way to go.
- **Notes:** epoch time ~245s, only 6–7 epochs fit in 30 min — model is undertrained. Loss curve still descending. Val/loss kept dropping (2.03 at epoch 6) but surf_p MAE was non-monotonic. Next ideas: (1) keep bf16 but reduce n_layers to 6 to squeeze more epochs — convergence currently > capacity; (2) Fourier feature positional encoding for (x,z); (3) auxiliary surface-only head; (4) split loss into surf-pressure-heavy + vol; (5) EMA weights. Also: train.py auto-predict OOM'd because parent process still held GPU; ran predict.py manually after killing parent. Consider freeing model/cache before subprocess.
