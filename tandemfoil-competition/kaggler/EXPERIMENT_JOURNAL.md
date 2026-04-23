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

### 2026-04-23 — iter2-sub30k-lr5e4-surfp2x (iter 2)
- **Hypothesis:** Iter 1 was still improving at the timeout (e32) and had high epoch-to-epoch variance. Cut subsample 40K→30K for more epochs (~45), drop peak LR 7e-4→5e-4 with warmup 500→1000 to reduce variance, and upweight surface pressure 2× in the surface loss (the scoring metric).
- **Change:** `train.py` — `train_subsample=30000`, `lr=5e-4`, `warmup_steps=1000`, `surf_p_weight=2.0`, channel-weighted surface loss, `epochs=50` (so cosine schedule fully decays within the 30-min cap).
- **Result:** 42 epochs in 30.5 min (~44 s/epoch). Best epoch 42: `val/avg_surf_p=99.03`. **Test `avg_surf_p=77.98`** (split: single=57.5, geom_rc=103.5, geom_cruise=44.7, re_rand=106.2). Submission `thorfinn/4f0c60b`.
- **Verdict:** Kept — despite val being slightly worse (99 vs 97), test improved from 87.5 → 78.0. Likely because val tracks only 100 samples per split; test has 200.
- **Notes:**
  - `single_in_dist` and `geom_camber_cruise` dropped hard (92→58, 61→45). Those are the "similar-to-training" splits — longer training + lower LR extracts more signal there.
  - `geom_camber_rc` barely moved (102→103) — this is the capacity/generalisation frontier: unseen raceCar camber with large Re and AoA swings. Iter 3 must attack this specifically.
  - `re_rand` drifted up (95→106) — also tandem, also unseen Re mix. The iter 1 model was already close to its limit here; iter 2's lower LR may have under-trained tandem-specific behaviour. Candidate fix: upweight tandem samples in the balanced sampler, or extend training with tandem-focused epochs.
  - Train surf MSE bottomed out at 0.02–0.05 (overfitting start). Val plateau around 100 → bigger model or regularisation (dropout/EMA) may unlock another step.

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
