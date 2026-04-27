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
### 2026-04-27 — v1-h256-L8-surfw30 (baseline submission)
- **Hypothesis:** Port my proven apr23 recipe (hidden=256, L=8, slice=64, EMA, AMP bf16) and tune the loss for the new metric (avg surface pressure MAE). Use surf_weight=30 with channel weights biasing surface-p (Ux=0.3, Uy=0.1, p=1.0) so the gradient signal flows mostly through surface pressure.
- **Change:** New `model.py` (Transolver pulled out of train.py); new `train.py` with EMA, channel-weighted vol/surf MSE, cosine LR (T_max=16, eta_min=lr*0.02), bf16 AMP, gradcip=1.0, train_max_points=80K subsample, AdamW lr=5e-4. `predict.py` adapted to load model from `config.yaml`.
- **Result:** 15 epochs in 30 min (2.1 min/epoch). Best `avg_surf_p = 128.34` at epoch 15 (still improving monotonically — train surf MSE 0.85→0.15). Per-split surface-p MAE: single_in_dist=141, geom_camber_rc=137, geom_camber_cruise=109, re_rand=126. W&B `alphonse/v1-h256-L8-surfw30` (`ylv1suio`). Submitted at commit `f5ebc28`.
- **Verdict:** Kept — first real submission, beats the 3325cb3 baseline (~360) by 64%, lands ~rank 4 (just below fern at 131.69). Still improving — warm-start should add a lot more.
- **Notes:** All tracks improved monotonically each epoch. Volume MSE also fell (1.47→0.27) so the channel-weighted loss is balanced enough. Time per epoch is ~2 min — full mesh would be ~5 min, so subsample is the right move for the first round; switch to full mesh once warm-start gains slow.
