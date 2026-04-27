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
