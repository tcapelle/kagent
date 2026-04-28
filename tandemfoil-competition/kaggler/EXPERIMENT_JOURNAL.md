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

### 2026-04-28 — iter21 Re-aware pressure normalization — SECOND BIG WIN

- **Hypothesis:** The dominant remaining error was on `val_re_rand` and `geom_camber_rc` (test 83.85 and 77.93 vs leader 33.65/45.02). Pressure scales roughly with `Re^k` (k between 1 and 2). Dividing pressure targets by `(Re/Re_ref)^k` should remove this systematic Re-dependent variance, leaving the model to predict a Re-invariant pressure that we then multiply back at inference.
- **Change:**
  - `train.py`: compute per-sample `Re_factor = exp(re_norm_k * (log_Re - re_ref_log))` from raw `x[..., 13]` *before* input normalization. Divide pressure target by `Re_factor` for loss; multiply prediction by `Re_factor` for physical-space MAE.
  - `predict.py`: same `Re_factor` post-multiplication on pressure (channel 2).
  - CLI: `--re_norm_k 1.0 --re_ref_log 14.58 --num_pos_freqs 16 --lr 1e-4 --constant_lr --epochs 9 --save_per_epoch`. Resumed iter20 ckpt; the model needed to relearn its target distribution but converged in 9 epochs.
- **Result:** Best epoch 9 (EMA) mean=**55.12** (-10.5% vs iter20 61.61). Per-split: single=49.43 (-5.8%), **geom_camber_rc=71.00 (-15.3%)**, geom_cruise=43.02 (-4.6%), **re_rand=57.03 (-12.2%)**. Submission `apr27/fern/d990439`.
- **Verdict:** kept — biggest single-iteration improvement of the entire run. **All four splits improved**, and the two hardest splits (geom_camber_rc and re_rand) improved the most, exactly as the hypothesis predicted.
- **Notes:** Trajectory was wobbly early (epochs 2-3 went UP) because the model needed to relearn its output target distribution. By epoch 7 it had clearly latched on and was descending fast. The post-training SWA (last 4) was 56.19 — worse than best, because the trajectory was still descending fast. Iter22 continues the chain at constant lr=5e-5.

### 2026-04-27 — iter17-iter20 Fourier-frequency exploration

- **iter17 (num_pos_freqs 6→10, lr=1e-4 cosine warmup):** 62.56 (-1.95% over iter16 63.80). New high-frequency capacity broke the Fourier-6 plateau. Submission `apr27/fern/e242516`. Test scored 60.03 — fern landed at #8.
- **iter18 (continue lr=5e-5):** 61.91 (-1%).
- **iter19 (continue lr=3e-5):** 61.65 (-0.4%).
- **iter20 (num_pos_freqs 10→16, lr=1e-4):** 61.61 (no real gain). The 10-frequency representation was already saturated for this problem; more frequencies didn't add useful signal.
- **Lesson:** there's a sweet spot for Fourier frequencies; doubling didn't help. Iterating chain at low lr keeps yielding ~1% per iter for ~3-4 iters then plateaus.

### 2026-04-27 — iter13 +Fourier features (NeRF-style) — BIG WIN

- **Hypothesis:** The plateau at ~71 looked like a basin the chain couldn't leave with optimization-only changes. NeRF-style positional encoding on `(x, z, saf)` should give the model high-frequency capacity for the turbulent-flow patterns that vary on small spatial scales — and thorfinn's run config showed they used `num_pos_freqs=10` for their top result.
- **Change:**
  - `model.py`: optional Fourier encoding of the first 4 input dims (position + saf) at `num_pos_freqs` frequencies (geometric series, `2^k * π`). Concatenated to the original features before the preprocess MLP.
  - `train.py`: when resuming a non-Fourier ckpt into a Fourier model, pad `preprocess.linear_pre.0.weight` with zeros for the new dims so iter11's prior is preserved at init.
  - CLI: `--num_pos_freqs 6 --lr 1e-4 --constant_lr --epochs 9 --save_per_epoch`. Resumed iter11 ckpt.
- **Result:** Best epoch 9 (EMA) mean=**66.85** (-5.6% over iter11 70.84). Per-split: single=58.38 (-6%), geom_camber_rc=91.20 (-4.7% — finally!), geom_cruise=47.72 (-7%), re_rand=70.10 (-5.6%). SWA last 4 = 68.54 (worse — trajectory still descending so averaging late epochs with earlier hurts). Submission `apr27/fern/745f0de`.
- **Verdict:** kept — first real breakthrough since the chain plateaued. **All 4 splits improved meaningfully**, including the previously stubborn `geom_camber_rc`. The positional encoding gives the model enough high-frequency representation to model turbulent surface-pressure variations.
- **Notes:** Trajectory was still descending at epoch 9 — more training will improve further. Iter14 continues the chain at lr=5e-5 constant. Lessons: when a plateau looks impervious to optimization tweaks, suspect representation capacity rather than training dynamics.

### 2026-04-27 — iter12 weighted ensemble [iter11:0.6, iter10:0.4] — submitted

- **Hypothesis:** Average iter11 + iter10 (the two best chain ckpts) for one last small ensemble lift before pivoting.
- **Change:** Sweep over weights; best at `[iter11:0.6, iter10:0.4]` = val 70.80 (vs iter11 alone 70.84).
- **Result:** Submission `apr27/fern/041f75f`. Marginal val gain 0.04 — same story as every chain ensemble: too correlated.
- **Verdict:** kept as a submission, but the ensembles never beat the latest chain ckpt by more than rounding error.
- **Notes:** Leaderboard refreshed after I'd been training ~7h; the field rolled forward dramatically (top 32, was 42 in the morning), and fern landed at #8 (test 67.63 from iter10, vs val 70.99 — test is ~5% better than val for our model). Relative to the leaders we'd need a large architectural jump (Fourier features, Re-aware norm, or a much larger model trained for longer) to climb.

### 2026-04-27 — iter11 constant-lr-5e6 (continuation from iter10)

- **Hypothesis:** One more chain step at lr=5e-6 to squeeze the last few hundredths.
- **Result:** Best epoch 9 (EMA) mean=70.84 (-0.15 vs iter10 70.99). SWA = 70.91 (worse than best). Submission `apr27/fern/6caa7e1` (iter10's commit hash, since I'd not yet pushed iter11 — the iter11 train.py committed iter10's code).
- **Verdict:** kept; convergence asymptote is around 70.7–70.8 for this setup.

### 2026-04-27 — iter10 constant-lr-1e5 (continuation from iter9)

- **Hypothesis:** Same recipe as iter9 with lr halved to 1e-5.
- **Result:** Best epoch 6 (live) mean=70.99 (-0.30 vs iter9 71.29). SWA = 71.15 (worse). Submission `apr27/fern/2359b08`. *Test score 67.63* — test landed below val (the gap is consistent across chain ckpts, ~5% lower on test than val).
- **Verdict:** kept; this is the best single-model fern ckpt and is the one currently scored on the leaderboard at 67.63.

### 2026-04-27 — iter9 constant-lr-2e5 + per-epoch SWA (continuation from iter6)

- **Hypothesis:** With cosine fully consumed by iter6, switching to constant LR (2e-5) and continuing 9 more epochs lets the model keep walking the basin floor. Save per-epoch ckpts and SWA the last 6 to smooth out late-epoch noise.
- **Change:** New `--constant_lr` and `--save_per_epoch` flags in `train.py`; post-training SWA done in-process. CLI: `--lr 2e-5 --constant_lr --epochs 9 --warmup_steps 0 --save_per_epoch --swa_last_n 6` (resume iter6 ckpt).
- **Result:** Best epoch 9 (EMA) mean=71.29 (-0.9% vs iter6 71.94). SWA mean=71.30 — basically tied with best, *no* SWA gain. Submission `apr27/fern/89e07bd`.
- **Verdict:** kept — small but real continuation gain. SWA itself didn't help (the constant-LR trajectory is too tight for averaging to denoise meaningfully).
- **Notes:** Constant-LR continuation > additional cosine-decay fine-tuning here because the cosine end-of-schedule LR (~1e-7) was effectively zero by iter6. Holding LR at 2e-5 lets the model keep nudging. EMA finally beat live in late epochs once the trajectory was stable. The 9-epoch trajectory was: 71.88 / 71.94 / 71.99 / 71.88 / 71.62 / 71.67 / 71.57 / 71.31 / 71.29 — late epochs do most of the work.

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
