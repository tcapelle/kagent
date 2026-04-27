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

### 2026-04-27 — iter3-fourier-features (BIG WIN, KEPT)
- **Hypothesis:** Pressure spikes at airfoil leading-edge stagnation are high-frequency in space; an MLP-only Transolver underfits them. Random Fourier features on the 2-D position (Tancik et al. / Aero-Nef recipe) provide a high-frequency basis that the model can blend.
- **Change:** new `FourierEmbedding` in `model.py`; `Transolver` accepts `use_fourier=True, n_fourier=32, fourier_sigma=8.0` and concatenates `[sin(B·pos), cos(B·pos)]` (64 extra features) into the preprocess input. Wired through `train.py`'s config.
- **Result:** 13/14 epochs. **avg val surf_p MAE 94.42 → 85.71 (-9.2%) — biggest gain so far.** Per-split:
  - in_dist 114.13 → 100.78 (-11.7%) — the peak-pressure split benefited most
  - rc 107.91 → 96.80 (-10.3%)
  - cruise 68.51 → 63.07 (-7.9%)
  - re_rand 87.12 → 82.18 (-5.7%)
  - **Test leaderboard: 79.95 → 75.06, jumped to rank #4.** Gap to leader thorfinn 45.94 still ~30 points.
  W&B `kagent-tandemfoil3/rdqtq1nu`.
- **Verdict:** kept. Cumulative val: 97.85 → 85.71 (-12.4%); cumulative test: 79.95 → 75.06 (-6.1%).
- **Notes:** Adds ~16K params, no measurable per-epoch overhead (142s same as before). Validation losses still decreasing at epoch 13 — model not converged. Next levers: push Fourier harder (more freqs, higher sigma), TTA y-flip at inference (free 3-6%), more aggressive var_floor.

### 2026-04-27 — iter2-balanced-loss (KEPT)
- **Hypothesis:** Pressure variance is 17–304 across domains (after global y_std=679 normalisation, per-sample variance ranges 0.0006 to 0.12 — 200x). Loss is dominated by raceCar tandem; low-Re cruise Part3 gets nearly no gradient. Divide squared error by per-sample variance (with floor 0.05 to cap upweight) → equalises gradient contributions. Also split surface weight: surf_p_w=2.5, surf_uv_w=0.5 (concentrate budget on the metric).
- **Change:** `train.py` — compute per-sample masked variance; `sq_err *= 1/(y_var_b + var_floor)`; per-channel split surface loss into `surf_uv_loss + surf_p_loss`. Ckpt selection switched to `avg val surf_p MAE` (the leaderboard metric) instead of `val/loss`.
- **Result:** 13/14 epochs. **avg_surf_p_mae 97.85 → 94.42 (-3.5%).** Per-split: in_dist 117.35→114.13 (-2.7%), rc 108.63→107.91 (-0.7%), cruise 73.79→68.51 (-7.2%), re_rand 91.61→87.12 (-4.9%). W&B `kagent-tandemfoil3/ecv6vhuq`.
- **Verdict:** kept. Direction confirmed: cruise (lowest pressure scale) gained the most.
- **Notes:** var_floor=0.05 was conservative — actual upweight ratio ~3-4x. More aggressive (smaller floor) might net more, but risk overfitting Part3. Single_in_dist still stuck near 114 — peak-pressure stagnation regions dominate. Next: Fourier features on position to capture high-freq pressure spikes (Aero-Nef recipe).

### 2026-04-27 — iter1-apr23-baseline (KEPT)
- **Hypothesis:** rebuild apr23 nezuko's validated config (h=128, L=6, slice=96, n_head=4, surf_weight=1.5, weight_decay=3e-5, epochs=14, lr=5e-4, bf16 autocast, grad_clip=1.0). Modular split: model classes -> `model.py` so `predict.py` can import without launching training. Mirror best ckpt to `checkpoints/best.pt` and PVC.
- **Change:** new `model.py` with Transolver classes; `train.py` imports from it and applies bf16 autocast + grad_clip + PVC mirror. `predict.py` loads via `model.py` + reads `config.yaml` from checkpoint dir.
- **Result:** 13/14 epochs in 30.4 min. Best val/loss=0.5956 at epoch 13. Val surf_p MAE: in_dist=117.35, geom_rc=108.63, cruise=73.79, re_rand=91.61. **Avg val surf_p MAE = 97.85.** W&B `kagent-tandemfoil3/hjwi94ao`.
- **Verdict:** kept, ckpt committed at `55049c8` for first submission.
- **Notes:** Slightly worse than apr23 best (94.5), likely seed/split differences. Validation losses still trending down at epoch 13 — model is undertrained. Next: tackle the dominant pathology (per-sample pressure-variance imbalance) and lift surf_p with Huber + heavier weight on the pressure channel only.
