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

### 2026-04-27 — v2-warm-lr2e4
- **Hypothesis:** v1 was still improving monotonically at epoch 15; warm-start with lower LR (5e-4 → 2e-4) should compound gains as in apr23 history.
- **Change:** `--resume models/model-ylv1suio/checkpoint.pt --lr 2e-4`. Otherwise identical recipe.
- **Result:** 15 more epochs in 30 min, all monotonic. Best `avg_surf_p = 109.03` at epoch 15. **15% over v1 (128.34→109.03), 70% over baseline.** W&B `alphonse/v2-warm-lr2e4` (`et5oaizr`). Submitted at commit `e6d8044`.
- **Verdict:** Kept — clean win, warm-start chain is working. Still improving at epoch 15.
- **Notes:** Train surf MSE 0.23→0.10 — keeps falling. Per-split p MAE: single_in_dist=119, geom_camber_rc=120, geom_camber_cruise=89, re_rand=108. The cruise camber split is now best — extreme p values aren't dominating like I'd feared.

### 2026-04-27 — v3-warm-fullmesh-lr1e4 (full-mesh breakthrough)
- **Hypothesis:** apr23 history showed full-mesh training (vs 80K subsample) was a ~22% breakthrough — train/eval distribution gap on the slice attention. Try `train_max_points=0 --batch_size=2 --lr=1e-4` warm-started from v2.
- **Change:** `--resume models/model-et5oaizr/checkpoint.pt --lr 1e-4 --train_max_points 0 --batch_size 2`. ~4.3 min/epoch (vs 2.1 min for subsample), so only 8 epochs in 30 min. VRAM peak 50.7 GB.
- **Result:** Best `avg_surf_p = 76.97` at epoch 8 (still monotonic). Per-split p MAE: single_in_dist=85, geom_camber_rc=90, geom_camber_cruise=59, re_rand=74. **29% over v2 (109.03→76.97), 79% over baseline.** W&B `alphonse/v3-warm-fullmesh-lr1e4` (`77rzoyyn`).
- **Verdict:** Kept — biggest single-step improvement so far. Confirms the apr23 finding: subsample causes a real train/eval distribution gap for slice attention.
- **Notes:** Beats nezuko (79.95) on the leaderboard — should now rank ~2. Train auto-submit OOMed because the train process held GPU memory while invoking predict.py; ran predict.py separately after kill. TODO: explicitly free GPU before subprocess invocation in train.py.

### 2026-04-27 — v4-warm-fullmesh-lr5e5
- **Hypothesis:** v3 was monotonic until timeout; standard warm-start with lower lr (1e-4 → 5e-5) for gentler refinement.
- **Change:** `--resume models/model-77rzoyyn/checkpoint.pt --lr 5e-5 --train_max_points 0 --batch_size 2`. Plus a small `train.py` fix to free GPU memory before invoking predict subprocess (v3 had OOMed).
- **Result:** 7 epochs in 30 min, all monotonic. Best `avg_surf_p = 70.55` at epoch 7. **8.3% over v3 (76.97→70.55), 80% over baseline.** Per-split p MAE: single_in_dist=78, geom_camber_rc=84, geom_camber_cruise=51, re_rand=68. W&B `alphonse/v4-warm-fullmesh-lr5e5` (`6ntu28pb`).
- **Verdict:** Kept — chain still working but gains diminishing. Auto-submit fix worked.
- **Notes:** Now ~rank 2 (between nezuko 79.95 and thorfinn 45.94). Gap to thorfinn is ~35%, which is too big to close with warm-start alone. Need a structural change (bigger model, surf_weight bump, or arch experiment). Continue chain one more iter while parallel-thinking new directions.
