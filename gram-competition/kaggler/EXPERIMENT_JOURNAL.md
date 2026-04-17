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

### 2026-04-17 — v_in_mean-anchor-violet-e3
- **Hypothesis:** residual from `v_in.mean(time)` gives a per-sample mean-flow anchor; with proj_out zero-init, prediction starts AT that anchor. Smaller model (256×6) fights overfitting seen in e1/e2.
- **Change:** `train.py` `PointNet` — residual from `v_in_mean`, zero-init output head, baseline-size model, no AMP. Added pre-training val check.
- **Result:** pre-train val/l2 = **1.42**, final val/l2 = **1.24** (best epoch 47). 50 epochs × 23 s, 4.2 GB. train 4.71 → 3.38. Run `z4w45445`. Auto-predict failed (predict.py import triggers train.py's argparse).
- **Verdict:** discarded — same plateau as e1/e2 (~1.24–1.26).
- **Notes:** Key insight: the pre-train val (predict=`v_in_mean` at every output step) is already **1.42** — far worse than baseline's 0.80. So `v_in_mean` is a **worse** estimator of the flow than baseline's learned `f(pos)`. Why? Baseline learns the true time-averaged mean flow by pooling across many training simulations at each position; `v_in_mean` is a 5-sample per-simulation estimate with substantial residual turbulent fluctuation. Three per-point-MLP experiments (e1/e2/e3) all plateau at ~1.25. The architecture class is the bottleneck, not residual vs. absolute nor richer features. **Next: either (i) go diagnostic — run baseline + no-slip only to verify infra, then add spatial aggregation (voxel-grid pooling or KNN) — or (ii) try a subsample+Transformer so the model can see global per-sample context.**

### 2026-04-17 — absolute-velocity-violet-e2
- **Hypothesis:** Predicting absolute velocity (not residual) with richer features + no-slip BC + 512×8 MLP should beat baseline 0.80. Thought e1 failed because residual pinned model to v_last (noisy).
- **Change:** `train.py` `PointNet` — predict absolute v_out via normalization-denormalization, added `v_in.mean(time)` feature, kept all others from e1.
- **Result:** val/l2 = **1.2618** (same failure as e1). 49 epochs × 37 s, 6.6 GB. train_loss 8.87 → 3.34. Run `dg4gt…` (AMP).
- **Verdict:** discarded — same floor as e1.
- **Notes:** Revised hypothesis: the baseline's apparent "reshape bug" (`velocity_in.reshape(B, N, T*C)` without permute) is actually **accidental spatial aggregation** — it scrambles velocity info across 5 adjacent memory-points (often spatially adjacent after preprocessing), forcing the model to learn `f(pos) → mean-flow-field` instead of fitting instantaneous turbulence. Fixing the reshape removes this implicit pooling AND exposes the model to pointwise-turbulent `v_in`, which the point-wise MLP overfits to. Both e1 and e2 train loss drops to ~3.3 while val sits at ~1.26 → clear overfit to turbulent noise. **Next: anchor residual on `v_in.mean(time)` (a less-noisy per-sample estimate of mean flow) and/or add real spatial aggregation.**

### 2026-04-17 — residual-prediction-violet-e1
- **Hypothesis:** Residual prediction (output = v_last + delta) should beat absolute-velocity prediction since v_last is a strong prior; adding no-slip BC + richer features (temporal diffs, magnitudes, dt_out, per-sample pos normalization) + 8-block 512-hidden MLP with AMP bf16 should crush the baseline (0.80).
- **Change:** rewrote `train.py` with `PointResNet`: residual head, no-slip via mask, permuted reshapes (fixed the baseline's reshape that scrambles points vs pos), bf16 autocast, AdamW+cosine.
- **Result:** val/l2 = **1.2547** (worse than baseline 0.80). 49 epochs × 37 s, 6.6 GB VRAM. train_loss 7.93 → 3.33. Run `xw4tdege`.
- **Verdict:** discarded — regressed vs. baseline.
- **Notes:** The residual bias (`output = v_last` at init) gives val/l2 ≈ 1.63 at epoch 1, which is already worse than the baseline's "fit f(pos) → mean-flow" starting point (~0.80 after training). The point-wise model can't cancel the instantaneous turbulent fluctuations in v_last fast enough. The baseline's reshape "bug" is consistent across input/output, so it just learns `f(pos) → mean velocity field` which is a surprisingly strong signal given the mostly-laminar flow. **Next**: drop residual; predict absolute velocity with the baseline's output structure but add no-slip BC, richer features, and bigger capacity — keep closer to the baseline recipe.
