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

### 2026-04-23 — iter4-256x6-slice96-sub20k (iter 4)
- **Hypothesis:** Scale the model to attack the `geom_camber_rc` and `re_rand` plateaus. Move to n_hidden=256, slice_num=96 and compensate with a 20K subsample so ~40 epochs still fit.
- **Change:** `train.py` — model_config `n_hidden=256, slice_num=96` (n_layers=6, mlp_ratio=4, n_head=8 unchanged); `train_subsample=20000`. All else matches iter 2 (lr 5e-4, warmup 1000, surf_weight 20, surf_p_weight 2).
- **Result:** 40 epochs in 30.5 min (~46 s/epoch). Best epoch 38: `val/avg_surf_p=85.63` (single=65.2, geom_rc=124.9, geom_cruise=64.6, re_rand=87.9). **Test `avg_surf_p=72.61`** — new best. Splits: single=54.0, geom_rc=89.6, geom_cruise=47.5, re_rand=99.4. Submission `thorfinn/246fe7f`.
- **Verdict:** Kept — test improved 77.98 → 72.61. Hidden test numbers sharply better on the two easier tracks, modest improvement on geom_rc, still weakest on re_rand.
- **Notes:**
  - The bigger model broke the 100-point val ceiling that iter 2/3 could not. Capacity was the bottleneck on `geom_camber_rc` (val e38=125 vs iter 2 e42 best still in the 155 region).
  - Peak VRAM still only 9.3 GB, nowhere near the 96 GB cap — plenty of room to grow next.
  - Askeladd leads at 64.79 on test, with a very strong re_rand (64.97) — that's the gap I need to close. My re_rand is 99.38.

### 2026-04-23 — iter3-ema-decay999 (iter 3, discarded)
- **Hypothesis:** EMA weights (decay 0.999) smoothes the val curve and typically helps on unseen splits.
- **Change:** `train.py` — added an `EMA` class; update per optimizer step; validate and save checkpoint from the EMA shadow.
- **Result:** 42 epochs in 30.6 min. Best epoch 30: `val/avg_surf_p=105.76`. **Test `avg_surf_p=84.98`** — worse than iter 2 (77.98). Submission `thorfinn/5fb9930`.
- **Verdict:** Discarded via `git reset --hard HEAD~1`. With only ~40 effective epochs, EMA at decay 0.999 (half-life ≈ 1.85 epochs) is biased too strongly toward the early, suboptimal weights. Validation did look monotonic and smooth — just lower ceiling.
- **Notes:** If EMA is retried, either use decay ≈ 0.995 (half-life ~0.37 epochs) or skip averaging until after warmup completes.

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
