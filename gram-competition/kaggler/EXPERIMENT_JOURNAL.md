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

### 2026-04-17 — dual-branch-refinement-violet-e9
- **Hypothesis:** Pure per-point MLP plateaus at ~1.25. Dual branches — per-point MLP (baseline) + zero-init transformer on 4096 anchor points + KNN-interpolate — lets the spatial branch *add* global corrections without ever making the base prediction worse than baseline.
- **Change:** `train.py` `BaselineMLP` — point branch (hidden=256, 6 ResBlocks) + spatial branch (3 TransformerBlocks, `nn.init.zeros_(spatial_head)`), interpolate via chunked `torch.cdist` KNN. Fixed critical output reshape bug: `[B,N,15].reshape(B,N,T,3).permute(0,2,1,3)`.
- **Result:** val/l2 = **1.2414** at epoch 44. 44 epochs × 41 s, 4.6 GB. train 0.04 → 0.02 (normalized). Run `rztko89s`. Leaderboard l2=1.2414 (rank 10).
- **Verdict:** kept (first valid submission) — but effectively same ceiling as e1–e3 (~1.24).
- **Notes:** **CRITICAL INSIGHT**: leaderboard score = W&B `val/l2_error` exactly (confirmed). Leaders thorfinn (0.74, EdgeConv) and alphonse (0.76, cross-attn+sub20k) are 40%+ better. The per-point-MLP architecture class caps at ~1.24 regardless of refinement head; spatial refinement via transformer on subsampled anchors + KNN-interp does *not* break past the ceiling. **Pivot next: dynamic-graph GNN (EdgeConv)** — operate directly on neighborhoods in point cloud, not on subsampled anchors.

### 2026-04-17 — subsample-transformer-violet-e8
- **Hypothesis:** Subsample 4096 anchor points, transformer attends globally, interpolate back to 100k via KNN. Should break the per-point-MLP ceiling because each output point can see long-range context.
- **Change:** `train.py` — subsample with stride, 4 TransformerBlocks (hidden=192), concat [point_feat, interp_feat] → Linear → out.
- **Result:** train loss plateau at **0.79** (normalized MSE), val/l2 ≈ predict-mean-velocity garbage. Killed early.
- **Verdict:** discarded — same plateau as e6/e7.
- **Notes:** At init, interpolated anchor features were noise and the concat treated them symmetrically with point features, polluting predictions. Fix was to split into dual branches with zero-init on the spatial head → became e9.

### 2026-04-17 — transolver-norm-loss-violet-e7
- **Hypothesis:** e6 Transolver got pinned to v_last by zero-init output + residual anchor. Removing residual + using normalized-space loss (`((pred-v_out)/vel_std).pow(2).mean()`) should let gradients flow regardless of velocity magnitude.
- **Change:** `train.py` — removed residual anchor, removed zero-init, switched loss to normalized MSE, lr=1e-3, grad clip 1.0.
- **Result:** train loss stuck at **0.79** (normalized), never descends. Softmax temperature likely degenerate — slice attention not learning.
- **Verdict:** discarded — same plateau.
- **Notes:** Transolver's slice-attention softmax collapses to uniform at init because slice logits are near-zero; the model effectively predicts the mean and gets stuck in a very flat region.

### 2026-04-17 — transolver-violet-e6
- **Hypothesis:** Transolver (physics-surrogate SOTA on Navier-Stokes benchmarks) should beat per-point MLP via learned slice-attention over the point cloud.
- **Change:** `train.py` rewritten — SliceAttentionBlock stack, residual anchor `v_in[:, -1]`, zero-init `proj_out`.
- **Result:** val/l2 ≈ **1.76** (equivalent to predicting v_last). 50 epochs. Killed.
- **Verdict:** discarded.
- **Notes:** Zero-init output + residual anchor pins model to v_last with near-zero gradient, and what gradient does exist is dominated by noise in v_last. Removed both in e7.

### 2026-04-17 — baseline-replica-violet-e5
- **Hypothesis:** Before pivoting to a fancy architecture, replicate the *literal* baseline (with its `reshape(B, N, T*C)` "bug") to verify infra — competition claims baseline hits 0.88.
- **Change:** `train.py` — minimal baseline MLP, no-slip mask, no extra features.
- **Result:** val/l2 plateau at **1.84**. Not 0.88.
- **Verdict:** discarded.
- **Notes:** Major discrepancy — competition-advertised baseline gets 0.88 but my replica gets 1.84. Suspected a setup/normalization difference. Eventually tracked: leaderboard score matches W&B val/l2 exactly, so 0.88 for "baseline" on the leaderboard must come from a *different* architecture than the repo's template (or the repo's README claim is outdated).

### 2026-04-17 — voxel-features-violet-e4
- **Hypothesis:** Add voxel-grid mean/std velocity features (5³=125 voxels) to each point's input so the point MLP sees neighborhood statistics — cheap spatial aggregation without an attention mechanism.
- **Change:** `train.py` — added voxel encoder that pools v_in stats per grid cell, concatenated into point features.
- **Result:** val/l2 plateau at **1.83**. Worse than e1.
- **Verdict:** discarded.
- **Notes:** Voxel mean/std had huge magnitude (Ux std ≈ 20), destabilized optimization despite normalization. Reverted.

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
