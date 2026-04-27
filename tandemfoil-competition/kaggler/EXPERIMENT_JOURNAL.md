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

### 2026-04-27 — v5-warm-fullmesh-lr2e5-surfw50
- **Hypothesis:** Continue chain w/ lower lr (5e-5 → 2e-5) and bump surf_weight 30 → 50 since the leaderboard cares only about surface.
- **Change:** `--resume models/model-6ntu28pb/checkpoint.pt --lr 2e-5 --train_max_points 0 --batch_size 2 --surf_weight 50`.
- **Result:** 8 epochs in 30 min, monotonic. Best `avg_surf_p = 67.04` at epoch 8. **5% over v4 (70.55→67.04), 81% over baseline.** Per-split p MAE: single_in_dist=72, geom_camber_rc=82, geom_camber_cruise=49, re_rand=66. W&B `alphonse/v5-warm-fullmesh-lr2e5-surfw50` (`26rinnut`).
- **Verdict:** Kept — gain shrinking (29%→8%→5%). Time for a structural change.
- **Notes:** Plain warm-start chain at this rate would need ~10+ more iters to catch thorfinn (46). Trying L1 surface-p loss next — directly matches the metric.

### 2026-04-27 — v6-warm-fullmesh-l1surfp10
- **Hypothesis:** L1 on surface pressure directly aligns gradient with the metric (MAE). Add `surf_p_l1_weight=10` on top of warm-start chain.
- **Change:** `train.py`: new `surf_p_l1_weight` knob computing L1 only on surface pressure channel. Run: `--resume model-26rinnut --lr 2e-5 --batch_size 2 --train_max_points 0 --surf_weight 50 --surf_p_l1_weight 10`.
- **Result:** 8 epochs in 30 min, monotonic. Best `avg_surf_p = 63.39` at epoch 8. **5.5% over v5 (67.04→63.39), 82% over baseline.** Per-split p MAE: single_in_dist=68, geom_camber_rc=78, geom_camber_cruise=46, re_rand=62. W&B `alphonse/v6-warm-fullmesh-l1surfp10` (`wv3ygsye`).
- **Verdict:** Kept — small win, L1 helps but not a breakthrough. Chain still going.
- **Notes:** Cruise camber is now <50; geom_camber_rc is the lagging split (78). Maybe try heavier L1 next.

### 2026-04-27 — v7-warm-l1surfp30-lr1e5
- **Hypothesis:** Heavier L1 surface-p loss (10 → 30) plus lower LR (2e-5 → 1e-5) for further refinement.
- **Change:** `--resume model-wv3ygsye --lr 1e-5 --train_max_points 0 --batch_size 2 --surf_weight 50 --surf_p_l1_weight 30`.
- **Result:** 8 epochs in 30 min, monotonic. Best `avg_surf_p = 61.60` at epoch 8. **2.8% over v6 (63.39→61.60), 83% over baseline.** Per-split p MAE: single_in_dist=67, geom_camber_rc=76, geom_camber_cruise=43, re_rand=60. W&B `alphonse/v7-warm-l1surfp30-lr1e5` (`bu63jrg0`).
- **Verdict:** Kept, but gain shrinking fast (29%→8%→5%→5.5%→2.8%). LR may be too low (apr23 v14 lesson).
- **Notes:** Need a step-change — geom_camber_rc still bottleneck (76). Trying lr restart up + arch change next.

### 2026-04-27 — v8-warm-lr3e5-l1surfp30 (LR restart up)
- **Hypothesis:** apr23 v14 lesson — when chain gains shrink, lower LR may be the cause not convergence. Restart from 1e-5 → 3e-5.
- **Change:** `--resume model-bu63jrg0 --lr 3e-5 --train_max_points 0 --batch_size 2 --surf_weight 50 --surf_p_l1_weight 30`.
- **Result:** 8 epochs in 30 min, monotonic. Best `avg_surf_p = 59.10` at epoch 8. **4.1% over v7 (61.60→59.10), 84% over baseline.** Per-split p MAE: single_in_dist=64, geom_camber_rc=72, geom_camber_cruise=42, re_rand=58. W&B `alphonse/v8-warm-lr3e5-l1surfp30` (`2behuc15`).
- **Verdict:** Kept — LR-up restart worked again, confirming apr23 lesson.
- **Notes:** Gap to thorfinn now ~22% (59 vs 46). Continue chain with another LR-up restart.

### 2026-04-27 — v9-warm-lr5e5-l1surfp30 (LR restart up, 2nd time)
- **Hypothesis:** v8's LR-up worked (4.1%), try one more bump 3e-5 → 5e-5 (matches v4's LR).
- **Change:** `--resume model-2behuc15 --lr 5e-5 --train_max_points 0 --batch_size 2 --surf_weight 50 --surf_p_l1_weight 30`. Also extended `predict.py` to support comma-separated checkpoints (ensemble averaging).
- **Result:** 8 epochs in 30 min, monotonic. Best `avg_surf_p = 56.18` at epoch 8. **4.9% over v8 (59.10→56.18), 84% over baseline.** Per-split p MAE: single_in_dist=60, geom_camber_rc=68, geom_camber_cruise=42, re_rand=55. W&B `alphonse/v9-warm-lr5e5-l1surfp30` (`p6r6oy7j`).
- **Verdict:** Kept. LR-up restart pattern continues to work — stays in the 4-5% per-iter bucket.
- **Notes:** Gap to thorfinn ~18%. Chain still has room.

### 2026-04-27 — v10-warm-lr8e5
- **Hypothesis:** Bigger LR-up restart (5e-5 → 8e-5) — chain still has gas, push higher.
- **Change:** `--resume model-p6r6oy7j --lr 8e-5 --train_max_points 0 --batch_size 2 --surf_weight 50 --surf_p_l1_weight 30`.
- **Result:** 8 epochs in 30 min. Best `avg_surf_p = 54.13` at epoch 8. **3.6% over v9 (56.18→54.13), 85% over baseline.** Per-split p MAE: single_in_dist=58, geom_camber_rc=66, geom_camber_cruise=42, re_rand=51. W&B `alphonse/v10-warm-lr8e5` (`uncjx333`).
- **Verdict:** Kept — slightly noisier (epoch 2 small bump) suggesting lr=8e-5 is near the upper edge. Drop back to 5e-5 next round.

### 2026-04-27 — v11-warm-lr5e5
- **Hypothesis:** Drop back to lr=5e-5 after v10's 8e-5 was a bit noisy.
- **Change:** `--resume model-uncjx333 --lr 5e-5 --train_max_points 0 --batch_size 2 --surf_weight 50 --surf_p_l1_weight 30`.
- **Result:** 8 epochs, monotonic. Best `avg_surf_p = 51.54` at epoch 8. **4.8% over v10 (54.13→51.54), 86% over baseline.** Per-split p MAE: single_in_dist=55, geom_camber_rc=63, geom_camber_cruise=39, re_rand=49. W&B `alphonse/v11-warm-lr5e5` (`vtebwe5c`).
- **Verdict:** Kept. lr=5e-5 is the sweet spot — back to monotonic 4-5%/iter.
- **Notes:** Gap to thorfinn now 12%. Continue chain.
