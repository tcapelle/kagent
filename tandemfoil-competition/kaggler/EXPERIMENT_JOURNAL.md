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

### 2026-04-27 — iter5-chain-lr5e-4-sw15 (iter 5)
- **Hypothesis:** Iter 4 was still improving when timeout hit. Warm-start a fresh 30-min run from iter 4's best checkpoint with halved LR (5e-4) and bumped surf_weight (15) to keep pushing surface error down.
- **Change:** `--warm_start /mnt/new-pvc/.../iter4_best_backup.pt --lr 5e-4 --surf_weight 15 --warmup_steps 100 --cosine_epochs 60`. EMA shadow weights are reinitialised from the loaded weights (so they're consistent at start).
- **Result:** Best epoch 58, mean val surf_p MAE = **47.59** (single_in_dist=37.51, geom_rc=74.81, geom_cruise=22.10, re_rand=55.95). val/loss=1.45. 58 epochs in 28.1 min. Run: `frieren/iter5-chain-lr5e-4-sw15` (`bf5ustqc`). Commit `eb54b3a`.
- **Verdict:** Kept — improved on iter 4 by 4.8% (50.01 → 47.59). All 4 split MAEs improved; biggest drops on `single_in_dist` (-4.8) and `geom_cruise` (-2.0). geom_rc and re_rand barely moved (-1.6 and -1.3). The metric ceiling on these OOD tracks is set by something other than training time — likely architecture / capacity for global reasoning.
- **Notes / next ideas:**
  - **Diminishing chains.** Each chain is ~5% — more chains won't catch thorfinn (40.68).
  - **OOD tracks dominate loss budget.** geom_rc 74.81 + re_rand 55.95 = 130.8 of total 210.4 (62%). Have to attack those directly.
  - Iter 6 plan: chain again with **larger subsample (80k vol)** to give each gradient step more signal — 250k samples vs 160k now. ~44s/step bench (vs 33s). Should still fit ~30 epochs in 28 min.

### 2026-04-27 — resmlp-h384-l1-pwt3-ema-msf (iter 4)
- **Hypothesis:** Apply Edward+alphonse's proven recipe — L1 loss with channel weight `[1, 1, 3]` (boosts pressure), EMA shadow weights with decay 0.9995, multi-scale Fourier features (8 scales, max_freq 16). Roll back to iter 2's fast ResMLP (h=384 b=6 bf16) since Transolver was too slow. Aim: beat alphonse (50.70).
- **Change:** New `MultiScaleFourier` class on positions; new `channel_loss(...)` with L1+p_weight=3; EMA shadow `state_dict` updated post-step (decay 0.9995, start 500 steps); validation and final ckpt saved using EMA weights; `cfg.warm_start` arg for chaining; `arch="resmlp"` default.
- **Result:** Best epoch 58, mean val surf_p MAE = **50.01** (single_in_dist=42.29, geom_rc=76.42, geom_cruise=24.05, re_rand=57.27). val/loss=1.16. 58 epochs in 28.1 min, ~33s/val-epoch + 25s/no-val. 10.1GB peak. Run: `frieren/resmlp-h384-l1-pwt3-ema-msf` (`w0hstbct`). Auto-predict OK. Commit `3b1371c`.
- **Verdict:** Kept — **biggest single jump yet** (68.58 → 50.01, 27% drop). Leaderboard: thorfinn 40.68, edward 43.73, alphonse 50.70 — projected rank ~3-4 once scored.
- **Notes:**
  - **L1 loss + p_weight=3 alone deliver most of the gain.** Aligns optimisation directly with leaderboard metric.
  - **EMA val curve is very clean** — every val epoch was a new best, monotone decrease 91.68 → 50.01. No noise. EMA's noise-cancelling effect.
  - Track gap analysis vs thorfinn: we beat thorfinn on `geom_cruise` (24.05 vs 22.69 — close) and rival on `single_in_dist` (42.29 vs 46.33). We lose badly on **`geom_rc` (76.42 vs 56.54)** and **`re_rand` (57.27 vs 37.18)** — OOD generalisation is our weak spot.
  - Model was still improving at epoch 58 (cosine_epochs=60). Chain iter 5 from this checkpoint.
  - geom_rc and re_rand suggest the network memorises the training distribution well but extrapolates poorly to new camber / new Re. Possible fixes: data aug (Re jitter, mirror-flip), more global communication (Transolver with **masking fix** + smaller LR), or a Re-conditioned FiLM layer.

### 2026-04-27 — transolver-h320-n8-bf16 (iter 3) — REGRESSION
- **Hypothesis:** Slice attention adds global communication that should help geom_rc (the worst track in iter 2 at 92.63). Bring in TransolverNet class (h=320, n_layers=8, heads=8, dim_head=40, slice_num=64, mlp_ratio=2). Bump cosine_epochs to 80 so LR doesn't decay to 0 mid-run.
- **Change:** Added `SliceAttention`, `TransolverBlock`, `TransolverNet` to `train.py`; new `arch="transolver"` config. Predict.py dispatches arch from config.yaml. Same loss/optimiser as iter 2.
- **Result:** Best epoch 20, mean val surf_p MAE = **108.15** (single_in_dist=125.25, geom_rc=119.77, geom_cruise=85.97, re_rand=101.61). Run: `frieren/transolver-h320-n8-bf16` (`<see W&B>`). Only 21 epochs in 28.9 min — attention blocks ~2.4× slower per epoch (92s vs 38s for iter 2).
- **Verdict:** Worse than iter 2 (108 vs 68). Compute budget didn't allow attention to amortise its cost. Not committing this checkpoint; keeping iter 2's. Code change kept (TransolverNet stays as a class for future warm-starts).
- **Notes / failure modes:**
  - **Bug confirmed in slice attention:** padded tokens still receive non-zero `slice_weights` and inflate `slice_norm`, diluting slice tokens. Edward's code masks slice_weights against `pad_mask`. Need to add this for any future Transolver run.
  - **Loss/metric mismatch:** training with MSE while leaderboard scores MAE. Edward+alphonse use **L1 loss with p_channel_weight=3** — directly aligns optimisation with the metric.
  - **No EMA.** Both alphonse and edward note val noise; EMA decay ≈ 0.9995 should give ~3-5% surf_p drop nearly free.
  - Cosine_epochs=80 over 21 epochs means LR mult was still ~0.93 at end — schedule fine, but compute cap binding.

### 2026-04-27 — resmlp-h384-b6-bf16-skipval (iter 2)
- **Hypothesis:** Iter 1 was undertrained (only 8 epochs in 30 min). Reduce model to h=384 b=6, enable bf16 autocast, validate every other epoch, increase LR slightly. Should yield ~3-4× more gradient steps.
- **Change:** Reduce hidden=384 / n_blocks=6 (~7.3M params), lr=1.5e-3, `torch.autocast(bfloat16)` in train + val, `val_every=2`, `cosine_epochs=30` to anneal cosine LR by epoch 30. Free GPU mem before auto-predict subprocess to fix iter-1 OOM.
- **Result:** Best epoch 30, mean val surf_p MAE = **68.58** (single_in_dist=67.26, geom_rc=92.63, geom_cruise=42.68, re_rand=71.74). val/loss=2.45. 50 epochs in 28.3 min, ~38s/val-epoch + 30s/no-val-epoch. 12.4GB peak VRAM. Run: `frieren/resmlp-h384-b6-bf16-skipval` (`bfh13z32`). Auto-predict succeeded this time. Commit `8e0978c`.
- **Verdict:** Kept — major improvement (1.81×), and would have ranked between alphonse (50.83) and askeladd (77.66) on apr27-4 leaderboard. Edward leads at 43.75. Still well off the front.
- **Notes / failure modes:**
  - **LR collapse at epoch 30** — `cosine_epochs=30` made the LR multiplier hit 0 exactly there, so epochs 32-50 had LR=0 and weights frozen. Validation outputs were *bit-identical* across epochs 32, 34, ..., 50 (proving the freeze). Wasted ~12 min of compute. Fix in iter 3: `cosine_epochs=80` so LR never bottoms.
  - **geom_rc still the worst track at 92.63**, almost 2× geom_cruise (42.68) — point-wise ResMLP can't model unseen camber geometry well. Need global communication (slice attention or kNN aggregation).
  - val curve was monotonically improving through epoch 30 — model was still learning when LR died. Strongly suggests both more epochs and more capacity will help.

### 2026-04-27 — resmlp-h512-b8-sw20-fourier (iter 1)
- **Hypothesis:** Replace baseline Transolver with proven frieren mar27 recipe — ResMLP with hidden=512, 8 blocks, lr=1e-3, surf_weight=20 (bumped from 10 since leaderboard ranks by surface pressure MAE), Fourier features on 2D position for high-freq capacity. Subsample volumes to 60k/sample to fit more iters in 30 min.
- **Change:** Rewrote `train.py` & `predict.py`. New `ResMLP` model (~17M params, hidden=512, n_blocks=8, expansion=4, n_freqs=32, fourier_sigma=4). Added subsample helper, warmup+cosine LR, grad clip 1.0, checkpoint by mean surface-pressure MAE. Wrapped main loop in `if __name__ == "__main__":` so predict.py can `from train import ResMLP` without firing the CLI parser.
- **Result:** Best epoch 6, mean val surf_p MAE = **123.87** (single_in_dist=146.14, geom_rc=144.04, geom_cruise=90.87, re_rand=114.44). val/loss=5.52. 8 epochs total in 30.7 min, ~231s/epoch. 45GB peak VRAM. Run: `frieren/resmlp-h512-b8-sw20-fourier` (`ieozo3vf`).
- **Verdict:** Kept — first submission to apr27-4 leaderboard; commit `98b9503`. But **way behind** the apr27-bis baseline (thorfinn @ 45.94) and apr27-4 alphonse @ 51.66. Frieren's mar27 baseline got 33–55 MAE *with chained training*; from-scratch in 30 min stalls at ~125.
- **Notes / next ideas:**
  - **Auto-predict failed (OOM)** — train.py's subprocess to predict.py held both copies of the 17M model on a 96GB GPU; the train process alone retained 90GB at termination because Python hadn't released the buffers. Need to `del model; torch.cuda.empty_cache(); gc.collect()` before subprocess, or just run predict separately. (For now ran predict separately, fine.)
  - **Convergence is the bottleneck.** 8 epochs is too few for a 17M model from scratch. Options to try next: (a) bf16/fp16 mixed precision for ~2× speedup, (b) smaller model (hidden=256, 6 blocks ≈ 4M params) to do 25+ epochs, (c) skip-validation-every-other-epoch, (d) larger batch + smaller subsample.
  - **Pure ResMLP is point-wise** — every node is processed in isolation. The geometry context is encoded in dsdf/saf input features, but global flow features (e.g. wake interactions) can't be modelled. Adding a small set-attention block or grid-pool token could help, especially for tandem-foil splits.
  - val MAE ordering: geom_cruise (90) < re_rand (114) < geom_rc (144) ≈ in_dist (146). So in-distribution is currently *worse* than cruise OOD — model is severely undertrained.
