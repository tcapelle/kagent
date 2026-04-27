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

### 2026-04-27 — iter8-surf_p5 (FAILED, discarded)
- **Hypothesis:** in Cp-space, surface-pressure errors are dimensionless O(0.1) — much safer to crank `surf_p_weight=5.0` (vs 2.5) without volume gradient starvation.
- **Change:** `train.py` — `surf_p_weight: 2.5 → 5.0`. Single knob.
- **Result:** 13/14 epochs. **avg val surf_p MAE 66.76 → 71.06 (+6.4% WORSE).** Per-split: in_dist +6.6%, rc +1.5%, **cruise +16.3%**, re_rand +6.5%. W&B run id `o5xfi8m9`.
- **Verdict:** discarded — reset to iter7 (`29489dd`).
- **Notes:** Counter-intuitive but consistent with iter4-era v12-v15 finding (apr23 journal): heavy surf weight starves the *volume* features the surface needs for context (wake, recirculation). In Cp-space the optimum surf_p_w is even lower (~2.5) because the loss is already balanced. **Lesson: don't increase surf_p_weight past ~2.5 — it inverted on cruise (+16%) which is exactly the regime that should benefit most from focus.**

### 2026-04-27 — iter7-cp-norm (HUGE WIN, KEPT)
- **Hypothesis:** kinematic pressure (`y[..,2]`, units m²/s²) scales as q_inf = 0.5·U_inf², and U_inf can be recovered from `x[..,13] = log(Re)` via `U_inf = exp(log_Re) * (nu/L)` (~m/s for unit chord). Training in Cp space `(p / U_inf², U/U_inf, U/U_inf)` makes targets dimensionless and *regime-invariant*: every sample has Cp_std ~ O(0.1-1) regardless of Re, so a unit-MSE loss is automatically balanced — the per-sample variance trick can't compensate the 137× cross-regime span the way Cp normalization does directly.
- **Change:** `train.py` — added `y_scale_from_x_raw` helper using `NU_OVER_L=1.5e-5`; replaced `(y - y_mean)/y_std` with `y / y_scale` in train + val; replaced `pred * y_std + y_mean` with `pred * y_scale` for MAE. `predict.py` mirrors. Reverted hparams to iter4 baseline (surf_p_w=2.5, var_floor=0.001 since Cp-space variance is naturally O(0.05-0.5)).
- **Result:** 13/14 epochs. **avg val surf_p MAE 85.18 → 66.76 (-21.6%) — biggest single-iter gain so far.** Per val split:
  - in_dist 100.78 → 71.42 (-29.1%)
  - rc 96.80 → 87.94 (-9.1%)
  - **cruise 63.07 → 42.41 (-32.8%) — leader-cluster level**
  - re_rand 82.18 → 65.27 (-20.6%)
  - **Test leaderboard: 75.06 → 60.92, gap to leader thorfinn=37.33 narrowed from 30 to 23.6 points.** Per test split: in_dist 69.48, rc 79.30, cruise 36.34, re_rand 58.56.
  W&B `kagent-tandemfoil3/qegxaiel`.
- **Verdict:** kept. Cumulative val: 97.85 → 66.76 (-31.8%); cumulative test: 79.95 → 60.92 (-23.8%).
- **Notes:** Cruise drop confirms the diagnostic — the gap with leaders was purely scale-mismatch in target normalization. Single_in_dist at 69 vs leader 36 still 2× gap; that's peak-pressure stagnation fitting, where extra capacity / aux-head / Fourier-tuning may help. var_floor=0.001 was probably overkill; check whether reverting helps.

### 2026-04-27 — iter6-surfp4-vf002 (FAILED, marginal regression)
- **Hypothesis:** raise surf_p_weight 2.5 → 4.0 (more focus on the metric) and tighten var_floor 0.05 → 0.02 (more aggressive low-Re upweight, since cruise-Part3 is bottlenecked by the floor at ~20× max upweight when it really wants ~5000×).
- **Change:** `train.py` — `surf_p_weight=4.0`, `surf_uv_weight=0.3`, `var_floor=0.02`.
- **Result:** 13/14 epochs. **avg val surf_p MAE 85.18 → 85.35 (+0.2%, marginally worse).** W&B run on commit `f0820fc`.
- **Verdict:** discarded — restored iter4 ckpt; reverted hparams to iter4 baseline.
- **Notes:** Confirms the cruise gap is NOT a single-knob problem within the existing loss formulation. Per-sample variance balancing has hit its ceiling at var_floor ≥ 0.02 because the actual Cp-vs-Re scaling needs full-variance correction (~1000× cross-sample) which the floor caps. Diagnosed: target reparametrization to Cp space (factoring out U_inf²) is needed — that's iter7.

### 2026-04-27 — iter5-ema (FAILED, discarded)
- **Hypothesis:** EMA decay 0.999 with 200-step warmup applied to model weights for eval+save would dampen late-training noise (~3-8% gain).
- **Change:** `train.py` — `copy.deepcopy(model)` for EMA, `ema_update` after each optimizer step (warmup→hard copy, then exponential update). Validation and ckpt save use EMA model.
- **Result:** 13/14 epochs. **avg val surf_p MAE 85.18 → 89.88 (+5.5%, WORSE).** Test 78.99 (vs iter3 75.06). All splits worse. W&B run 8w0vdx91 (or similar — see latest with ema name).
- **Verdict:** discarded. Reset code to iter4 (`19e74ee`).
- **Notes:** Same failure mode as apr23 v6: with cosine LR decaying to 0 over 14 epochs, the live model is the post-convergence model. EMA averages back over the previous ~1000 steps including high-LR transients → a worse-than-current model. EMA only helps when training continues past convergence and oscillates around the minimum. Don't retry under cosine-LR + short epoch budget.

### 2026-04-27 — iter4-fourier-pushed (KEPT, marginal)
- **Hypothesis:** push Fourier features (n=32 → 64, sigma=8 → 12) to capture even higher spatial frequencies for stagnation peaks.
- **Change:** `train.py` — `n_fourier=64, fourier_sigma=12.0`.
- **Result:** 13/14 epochs. **avg val surf_p MAE 85.71 → 85.18 (-0.6%, marginal).** Per-split mixed: in_dist -2.0%, rc +3.3% (REGRESSED), cruise -2.2%, re_rand -2.3%. W&B `kagent-tandemfoil3/zm5udrdj`.
- **Verdict:** kept (marginally better) but the sigma=12 hurts the OOD-camber split — likely overfitting high-freq modes that don't transfer.
- **Notes:** sigma controls frequency scale; sigma=8 had cleaner generalization. If iter6 regresses, consider rolling Fourier back to (n=32, sigma=8).

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
