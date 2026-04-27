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
