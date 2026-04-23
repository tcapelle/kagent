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

### 2026-04-23 — transolver-192x6-bf16-sub40k (iter 1)
- **Hypothesis:** A bigger Transolver (192 hidden x 6 layers x slice_num=64, ~2.6M params) trained in bf16 with point subsampling (40K nodes/sample, keeping all surface) and warmup+cosine schedule should beat the default 128x5 baseline and leave enough budget for >30 epochs in 30 min.
- **Change:** `train.py` — Transolver bumped to n_hidden=192/n_layers=6/mlp_ratio=4, added bf16 autocast, `subsample_batch` keeps all surface + random volume to 40K, AdamW betas=(0.9,0.95), warmup 500 steps then cosine, grad_clip=1.0, surf_weight=20, track best by `avg_surf_p`. `predict.py` — load model via `config.yaml`, bf16 autocast, mirror ckpt to PVC.
- **Result:** 34 epochs in 30 min (~54 s/epoch). Best epoch 32: `val/avg_surf_p=97.08` (single_in_dist=104.8, geom_rc=115.8, geom_cruise=74.6, re_rand=93.1). Peak VRAM 12.5 GB.
- **Verdict:** Kept — first iter lands a clean submission (`thorfinn/0601ec5`) on an otherwise-empty apr23 leaderboard.
- **Notes:**
  - Auto-submit fail: `predict.py`'s `from train import Transolver` re-ran `sp.parse(Config)` and died on `--checkpoint`. Fixed by moving model to `model.py`; manually ran `predict.py` after the code fix to upload predictions.
  - Steady decay but high variance between epochs (115 → 120 → 115 → 111) — cosine schedule still has LR too high near end. Next iter: lower peak LR or longer warmup.
  - Surface stats from the data show velocities are **not** zero on the airfoil (Ux mean ≈ 5, Uy range [-14, 5]). Don't enforce no-slip.
  - Hardest split is `geom_camber_rc` (116 on best epoch) — model is generalisation-bound, not capacity-bound.
  - Ideas for iter 2: longer training with smaller LR (3e-4) or more epochs (subsample 30K → ~45 epochs), add EMA, deeper/wider model (256x8, slice_num=128), separate attention for surface vs volume tokens, explicit radial/polar features around foils.
