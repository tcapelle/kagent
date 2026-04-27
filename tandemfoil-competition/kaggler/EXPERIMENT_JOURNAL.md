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

### 2026-04-27 — iter8 weighted ensemble [iter6:0.6, iter5:0.3, iter7:0.1] — submitted

- **Hypothesis:** A weighted average of normalized predictions from the chain (iter5/iter6) plus the divergent iter7 might recover a small gain by canceling per-model errors.
- **Change:** Added `--weights` to `predict_ensemble.py` and `eval_ensemble.py`; submitted ensemble with weights tuned on val (sweep around iter6/iter5 mix).
- **Result:** Val: ensemble [0.6,0.3,0.1] = 71.73 (vs iter6 alone 71.94, ~0.3% gain). Sweep showed iter7 contributes near-zero — most of the gain is from iter5+iter6 weighted average. Submission `apr27/fern/f11eda9`.
- **Verdict:** kept as submission, but the gain is too small to celebrate. Chain models too correlated for ensemble to break out.
- **Notes:** Equal averaging always *hurts* (drags down by the worst member). Only weighted toward the strongest member yielded marginal gain. Per-split: ensemble nudged `geom_camber_rc` 95.46 → 94.80 — the only meaningful per-split improvement.

### 2026-04-27 — iter7 divergent-from-iter1 — for ensemble diversity

- **Hypothesis:** The chain has converged into one basin; resuming from iter1 ckpt (113.26) with iter5's pressure-focused recipe should land on a *different* trajectory by epoch 9, providing genuine diversity for ensembling.
- **Change:** `--resume <iter1 ckpt> --lr 3e-4 --surf_weight 50 --p_weight 30.0 --v_weight 0.05 --epochs 9 --save_per_epoch --swa_last_n 4`. Also added `constant_lr`, `save_per_epoch`, `swa_last_n` flags to `train.py`.
- **Result:** 9 epochs / 26.7 min. Best epoch 9 (live) mean=83.00 (still improving — would need more epochs). Per-epoch SWA over last 4 = 83.57 (worse). Submission `apr27/fern/54ca672`.
- **Verdict:** kept as ensemble member; alone it's much weaker than iter6 (83 vs 72). Did add a tiny ~0.2 point lift to a weighted ensemble.
- **Notes:** From-scratch retraining can't catch up to a chain in just 9 epochs. To make iter7 a stronger ensemble partner I'd need 30+ epochs of training, but `MAX_TIMEOUT_MIN=30` blocks that in one shot. The post-training SWA over last 4 didn't help here because the trajectory was still descending — averaging recent ckpts with worse ckpts pulls mean back up.

### 2026-04-27 — iter6 pweight50-sw80-lr2e5 — chain plateaued

- **Hypothesis:** Push pressure focus to extreme (`p_weight=50, v_weight=0, surf_weight=80`) at lr=2e-5 to squeeze the last few % out of the chain.
- **Change:** `--resume checkpoints/best.pt --lr 2e-5 --surf_weight 80 --p_weight 50.0 --v_weight 0.0 --epochs 9`.
- **Result:** Best epoch 4 (live) mean=71.94 (-0.3% vs iter5 72.14). Trajectory was almost flat — every epoch within ±0.05 of best. Submission `apr27/fern/ae0ba7c`.
- **Verdict:** kept (technically better) but the chain is fully plateaued at ~72. Pressure focus was already saturated by iter5; iter6 essentially refined within the same minimum.
- **Notes:** Three datapoints — ensemble(iter4, iter5)=72.68, ensemble(iter5_live, iter5_ema)=72.24, all *worse* than iter5 alone — confirm that prediction averaging across the chain doesn't help (models too correlated). iter7 needs to break out: try a **divergent** training path (resume from iter1 with iter5's pressure-focused recipe) to get a genuinely different ckpt for ensembling.

### 2026-04-27 — iter5 pweight30-vweight0.05-sw50-lr5e5 — chain refinement

- **Hypothesis:** Iter4's `p_weight=10` worked but barely; push it higher (30) and zero out velocity loss (v_weight=0.05) since velocity is not on the leaderboard. Drop LR further (5e-5) for fine refinement.
- **Change:** `--resume checkpoints/best.pt --lr 5e-5 --surf_weight 50 --p_weight 30.0 --v_weight 0.05 --epochs 9`.
- **Result:** Best epoch 6 (live) mean=72.14 (-2.4% vs iter4 73.90). Per-split: single=65.29, geom_rc=94.98, geom_cruise=52.90, re_rand=75.40. Submission `apr27/fern/fb7caed`.
- **Verdict:** kept, marginal gain. Chain now firmly plateauing.
- **Notes:** All splits improved a small amount but `geom_camber_rc` (unseen front foil camber, raceCar) is still the bottleneck (95). It remains hardest because we can't add training samples covering those geometries.

### 2026-04-27 — iter4 pweight10-vweight0.1-sw30 — pressure-focused chain

- **Hypothesis:** *The leaderboard scores ONLY surface pressure MAE, not velocity.* (Confirmed: `avg/mae_surf_p` is the only ranking metric — Ux and Uy don't affect score.) Reweighting the loss to deeply focus on pressure should improve the metric without risking velocity-prediction accuracy that nobody scores.
- **Change:** Added `p_weight` and `v_weight` to channel-weighted Huber. Iter4 cfg: `p_weight=10, v_weight=0.1, surf_weight=30, lr=1e-4, epochs=9`. Resumed iter3 ckpt.
- **Result:** 9 epochs / 26.7 min. Best epoch 8 (live) mean=73.90. Per-split: single=67.56 (-8%), geom_rc=96.35 (-2%), geom_cruise=55.08 (-7%), re_rand=76.61 (-4%). Submission `apr27/fern/1668758`.
- **Verdict:** kept — 4.6% improvement (77.47 → 73.90). All splits improved; `geom_rc` still the bottleneck.
- **Notes:** Trajectory was fairly flat (77.4 → 73.9 over 9 epochs). Early epochs barely moved — most gain in last 4. The pressure-focus is working but slowly. Iter5: try even more aggressive pressure focus (p_weight=30, v_weight≈0) and lower LR to keep chaining.

### 2026-04-27 — SWA(iter1+iter2+iter3) and SWA(iter2+iter3) — discarded

- **Hypothesis:** Stochastic weight averaging across the chained checkpoints would land in a wider/flatter minimum than any individual ckpt.
- **Change:** New `swa.py` averages the model state-dicts and re-evaluates against val splits (no extra training).
- **Result:** SWA(iter1+2+3) = 92.85 (worse than iter1 alone 113? no — it's between but bad), SWA(iter2+iter3) = 79.60 — both worse than iter3 (77.47). Iter1 is too far in weight space; iter2+3 averaging just regresses toward iter2.
- **Verdict:** discarded — SWA doesn't help when the chain is monotonically improving (averaging pulls back toward worse ckpt).
- **Notes:** SWA needs ckpts sampled around the same minimum (e.g., last few epochs of one run). Could try checkpoint averaging *within* a single run by saving every epoch.

### 2026-04-27 — iter3 warmstart-cosine9-lr1e4

- **Hypothesis:** Continue chain from iter2 at a lower peak LR (1e-4) for 9 more epochs of fine refinement, with the cosine schedule fully completing.
- **Change:** `--resume checkpoints/best.pt --lr 1e-4 --epochs 9 --warmup_steps 30`. Same model + bs + subsample as iter2.
- **Result:** 9 epochs in 26.6 min. Best epoch 8 (live): mean_mae_surf_p=77.47. Per-split surf_p: single=73.48, geom_rc=98.11, geom_cruise=58.93, re_rand=79.37. Submission `apr27/fern/4051524`.
- **Verdict:** kept — 6% improvement over iter2 (82.48 → 77.47).
- **Notes:** Marginal returns vs. iter2's 27% gain — chain is plateauing. `geom_camber_rc` (unseen front foil camber, raceCar) remains the worst split (98). Surface weight (10) might be too low when surface error dominates the metric.

### 2026-04-27 — iter2 warmstart-cosine9-lr3e4

- **Hypothesis:** Resume from iter1 ckpt with a fully-completed cosine schedule sized to actual training time (epochs=9 ≈ 27 min) and a lower peak LR (3e-4) to fine-tune without destabilising. Iter1's cosine never decayed (sized for 50 epochs but only ran 11), so we never benefited from lr annealing.
- **Change:** Same model + recipe as iter1; resumed `checkpoints/best.pt`. CLI: `--resume checkpoints/best.pt --batch_size 4 --train_max_nodes 80000 --lr 3e-4 --epochs 9 --warmup_steps 50`. EMA reset on resume (cold-starts averaging from current weights).
- **Result:** 9 epochs in 26.6 min. Best `mean_mae_surf_p=82.48` at epoch 9 (live; EMA was close behind). Per-split: single=81.42, geom_rc=104.40, geom_cruise=61.36, re_rand=82.75. Submission `apr27/fern/d4221b0`.
- **Verdict:** kept — 27% improvement (113.26 → 82.48). Still 2× behind leader (frieren=42).
- **Notes:** Completed cosine made the difference — train loss kept dropping until last epoch. `geom_camber_rc` (unseen front foil camber, raceCar) is the bottleneck split (104) while cruise is easy (61). Live beat EMA in late epochs, suggesting EMA was lagging behind a still-improving live model. Idea for iter3: another warm-start chain at lower lr (1e-4) over 9 more epochs to keep dropping.

### 2026-04-27 — iter1 baseline-256h8L-bf16-ema-huber-sub80k

- **Hypothesis:** A bigger Transolver (hidden=256, L=8, slice_num=96, h=8) trained with bf16 AMP, warmup+cosine LR, EMA, Huber loss, and 80k-node training subsampling for speed should reach roughly frieren's ballpark (val mean surf_p ~50-100) in 30 min.
- **Change:** Refactored model into `model.py`. New `train.py` with: bigger Transolver, bf16 AMP autocast, warmup(100)+cosine LR, EMA(decay=0.999), Huber loss (β=1.0), grad_clip=1.0, training subsample to 80k nodes (keeps all surface), TF32, val on full mesh, best-by-mean-mae-surf-p, EMA-vs-live picked per epoch. Auto-submit at end.
- **Result:** 11 epochs × 177s training in 32.5 min wall-clock. Best `val/mean_mae_surf_p=113.26` at epoch 11 (EMA). Per-split surf_p: single=128.96, geom_rc=133.65, geom_cruise=85.94, re_rand=104.49. Submission: `apr27/fern/e34b551`.
- **Verdict:** kept — first valid fern submission this run.
- **Notes:** The cosine schedule was sized for `epochs=50` but we only ran 11, so LR only decayed ~5%. iter2 should size epochs to actual training time so cosine fully anneals. Trajectory was monotonically improving — more epochs would help. Frieren's iter1 (35 epochs) reached 54 surf_p, so our 11-epoch run is reasonably on-pace. VRAM was 40.8GB at bs=4 / 80k nodes — plenty of headroom.
