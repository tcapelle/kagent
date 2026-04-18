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

### 2026-04-18 — global-attention-violet-e16
- **Hypothesis:** Replace E15's coarse EdgeConv branch (2k anchors, k=32 graph) with **true global self-attention** — multi-head Transformer on the 2k anchors gives unlimited receptive field (any anchor can attend to any other) and should capture long-range wake/vortex structure that KNN neighborhoods cut off. Added absolute Fourier-pos embedding inside the coarse branch so tokens are spatially aware.
- **Change:** `train.py` `BaselineMLP.__init__` — swapped `EdgeConvBlock` → `TransformerBlock(num_heads=8)` for the coarse branch; added `coarse_pos_embed = Linear(fourier_dim, hidden)` and added pos feat to coarse input. `predict.py` kwargs synced. Launched `MAX_TIMEOUT_MIN=240 --epochs 280`.
- **Result:** val/l2 = **1.0286** (epoch 209). 212 epochs × 68 s, 6.1 GB bf16. train 0.044 → 0.0121. Timed out at 240 min at E212 (planned 280). W&B run `..(global-attn-e16)`.
- **Verdict:** **discarded** — regressed vs E15 (1.0140 → 1.0286, +1.4%). Reset code (`git reset --hard HEAD~1`) and restored E15 checkpoint via `git checkout fe4b347 -- checkpoints/best.pt`.
- **Notes:** Trajectory: e50=1.21 → e90=1.12 → e130=1.09 → e170=1.07 → e209=1.029. Consistently ~2-5% behind E15 at matched epochs. Global attention appears to *underperform* EdgeConv on this point-cloud task — EdgeConv's geometric inductive bias (edge features use relative position delta) seems more helpful than unlimited-range context on a sparse 2k-anchor grid. The absolute-Fourier pos embed couldn't fully substitute for local relative geometry. Also, self-attention on 2k tokens has far fewer params than 3 EdgeConv blocks on the same tokens — likely *under-parameterized*. Next push should go bigger rather than different: scale hidden + more fine-scale anchors + stronger augmentation (rotations). The architecture class seems to top out near ~1.01-1.03; need a more aggressive recipe to break below 1.00.

### 2026-04-18 — multiscale-edgeconv-violet-e15
- **Hypothesis:** Current single-scale EdgeConv (10k anchors, k=16, 5 blocks) has ~5-hop neighborhood reach → ~80-point receptive field, tiny relative to 100k-point global structure. Add a parallel **coarse** EdgeConv branch (2k anchors, k=32, 3 blocks) that covers global scale via sparser but wider neighborhoods. Two zero-init heads so total output at init = per-point baseline.
- **Change:** `train.py` `BaselineMLP.forward` — added `coarse_proj` + `coarse_blocks_list` + `coarse_head` (zero-init) running on stride-subsampled 2k anchors, k=32. Output sum: `point_pred + spatial_pred + coarse_pred`. `predict.py` kwargs synced. Launched `MAX_TIMEOUT_MIN=240 --epochs 280`.
- **Result:** val/l2 = **1.0140** (epoch 200). 204 epochs × 71 s, 6.3 GB. train 0.044 → 0.0117. Commit `fe4b347`. W&B run `49vkixu4`.
- **Verdict:** kept — beats E14 (1.0174 → 1.0140, 0.3% improvement). Small but monotone across runs.
- **Notes:** Best by epoch: e45=1.179 → e82=1.120 → e131=1.053 → e176=1.031 → e200=1.014. Trajectory matched E14 despite 71s vs 60s/epoch (14% coarse branch overhead). Gains are clearly diminishing from incremental feature engineering / branch additions — the architecture family is near its ceiling at ~1.01. Leaders (thorfinn 0.70, nezuko 0.75, alphonse 0.75) are 30%+ better. Next push needs something **architecturally different**: (a) global transformer attention on coarse anchors (2k is cheap for full attention), (b) physics-informed loss (divergence penalty for incompressibility ∇·v=0), (c) KPConv / Point Transformer, or (d) predict per-frame with temporal autoregression.

### 2026-04-18 — temporal-feats-violet-e14
- **Hypothesis:** E13 train loss plateaued at ~0.0125 with plenty of val headroom. The input was raw v_flat only — no *explicit* temporal-derivative signal. Add temporal velocity diffs (Δv between consecutive input frames, 4×3=12 dims) + velocity magnitude per frame (5 dims) so the model gets first-order turbulent acceleration/speed signals it would otherwise have to infer. Keep E13 architecture; train 240 min ≈ 240 epochs.
- **Change:** `train.py` `BaselineMLP.forward` — compute `v_diff = v[:,1:] - v[:,:-1]` and `v_mag = v.norm(-1)`; concat with existing `[pos, fourier_pos, v_flat]`. Input dim grows from 63 to 80. `predict.py` unchanged (kwargs same). Launched with `MAX_TIMEOUT_MIN=240 --epochs 280`.
- **Result:** val/l2 = **1.0174** (epoch 236). 240 epochs × 60 s, 6.0 GB bf16. train 0.020 → 0.0117 (normalized). Leaderboard l2=1.0174. Commit `9da4ee8`. W&B run `aqhb1y50`.
- **Verdict:** kept — modest improvement (1% over E13's 1.0276) but consistent with the longer schedule. Temporal features + extra time both helped.
- **Notes:** Best by epoch: e33=1.214 → e98=1.108 → e150=1.058 → e210=1.030 → e236=1.017. Train loss near 0.0117 suggests diminishing returns from raw-feature engineering. Trajectory was noisier than E13 in mid-training but converged cleanly. Ranking vs leaders (thorfinn 0.70, alphonse 0.75, nezuko 0.76) still large. Next: the per-point MLP + EdgeConv class may be ceiling'd near ~1.0 — consider (a) multi-scale coarse-to-fine GNN, (b) dedicated turbulence head (predict high-freq residual separately), (c) PDE-informed loss (divergence penalty for incompressibility), (d) cross-attention across input time frames — or pivot architecture toward a transformer-on-subsampled-anchors with far more aggressive feature mixing.

### 2026-04-18 — fourier-pos-violet-e13
- **Hypothesis:** E11 hit 1.1204 after only 60 epochs with train loss still descending — clearly under-trained. Raw xyz positions limit the per-point MLP's ability to express high-frequency spatial variation needed for turbulent flow. Add multi-resolution Fourier (sinusoidal) position encoding (standard NeRF trick) + train 3× longer (180 min ≈ 180 epochs). Expect break below 1.10.
- **Change:** `train.py` — new `FourierEmbed(n_freqs=8)` (base-2 geometric frequencies × π, sin+cos concat → 48-dim), concat with raw pos + v_flat at `proj_in`; kept E11's architecture (10k anchors, 5 EdgeConv blocks, k=16, bf16 autocast, y-flip aug) otherwise. `predict.py` kwargs synced. Launched with `MAX_TIMEOUT_MIN=180 --epochs 200`.
- **Result:** val/l2 = **1.0276** (epoch 179). 180 epochs × 60 s, 6.0 GB bf16. train 0.042 → 0.0125 (normalized). Leaderboard l2=1.0276 — **rank 8** (up from 9). Commit `e227457`.
- **Verdict:** kept — biggest improvement yet (9% over E11). Both longer training and Fourier encoding contributed.
- **Notes:** Best by epoch: e55=1.159 → e90=1.110 → e125=1.069 → e157=1.039 → e179=1.028. Still descending at cutoff — a 4× longer run would likely push to ~0.95. Train loss plateaued around 0.0125 at ep 180 (small headroom left). Rank-7 (askeladd 0.974) is now within reach. Leaders (thorfinn 0.71, alphonse 0.75, nezuko 0.78) still far ahead — next push needs either (a) even longer training run, (b) richer features (Fourier on velocity input too?), or (c) architectural change (e.g., temporal attention across the 5 input frames, or multi-scale GNN with coarse + fine anchors).

### 2026-04-17 — edgeconv-20k-violet-e12
- **Hypothesis:** E11 was under-converged at 60 min with train loss still descending — scaling anchor budget from 10k→20k should denser KNN interpolation to the 90k non-anchor points, plus +1 more edge block (5→6) for deeper multi-hop reach. 120 min budget for more epochs.
- **Change:** `train.py` — `BaselineMLP(edge_blocks=6, n_anchors=20000)`. `predict.py` synced. Launched with `MAX_TIMEOUT_MIN=120 --epochs 150`.
- **Result:** val/l2 = **1.1277** (epoch 52). 62 epochs × 117 s, 10.4 GB bf16. train 0.041 → 0.017. Commit `1e7dd92`.
- **Verdict:** discarded — regressed vs. E11 (1.1204 → 1.1277). Code reset to E11 config.
- **Notes:** Per-epoch descent was **identical** to E11 at same epoch count (e.g., e28=1.2275 vs E11 e30=1.2279). But at ~2× wall-clock cost per epoch, we got fewer total epochs in 120min than E11 got in 60min relative to schedule. Takeaway: **scaling anchors alone provides no per-epoch speedup**; the learning bottleneck is optimization/training steps, not anchor density. Better path is extending training time with a leaner architecture — confirmed by E13.

### 2026-04-17 — edgeconv-scaled-violet-e11
- **Hypothesis:** E10 EdgeConv tied e9 at 1.24 — likely under-powered at 3 blocks/k=12/4k anchors and under-trained (30-epoch budget). Scale the GNN (5 edge blocks, k=16, 10k anchors) + add y-flip data aug (F1 wake symmetry) + bf16 autocast for 2x speedup, extend timeout to 60 min. Expect break below 1.24 ceiling.
- **Change:** `train.py` — `BaselineMLP(edge_blocks=5, edge_k=16, n_anchors=10000)`; training loop: 50% chance to flip `v_in`/`v_out`/`pos` Y-axis; wrap forward+loss in `torch.autocast(bfloat16)`. `predict.py` kwargs synced.
- **Result:** val/l2 = **1.1204** (epoch 58). 60 epochs × 60 s (6.0 GB bf16). train 0.028 → 0.017 (normalized). Leaderboard l2=1.1204 — **rank 9** (up from 10; e9 was 1.2414). Commit `89b6a80`.
- **Verdict:** kept — first experiment to break 1.24 ceiling (~10% improvement). Still substantially behind leaders (thorfinn 0.73, alphonse 0.76).
- **Notes:** val/l2 was still descending at the 60-min cutoff (train loss 0.028 → 0.017, no sign of overfit) — **model is under-trained**. Smart moves to try next: (a) even more epochs / longer timeout to fully converge; (b) larger anchor budget (15k–20k) to reduce KNN interpolation loss on the 90k non-anchor points; (c) multi-scale — two EdgeConv stacks at coarse/fine resolutions; (d) drop the per-point MLP branch entirely (it contributes mean-flow only; GNN already does that better) and let the EdgeConv own the whole prediction with more blocks. The scaled-EdgeConv class clearly descends past the per-point ceiling so the architecture direction is correct.

### 2026-04-17 — edgeconv-gnn-violet-e10
- **Hypothesis:** E9 spatial branch was Transformer on 4k subsampled anchors — too coarse for local flow structure. Replace with DGCNN-style EdgeConv (KNN-graph message passing with MLP on edge features, max-pool neighbors), which is how leader thorfinn reaches 0.73. Keep dual-branch architecture with zero-init spatial head.
- **Change:** `train.py` — new `EdgeConvBlock` module (gather K neighbors, MLP on [x_i, x_j - x_i] edge features, max-pool, residual); replaced the Transformer stack with 3 EdgeConv blocks, `edge_k=12`, `n_anchors=4096`. Zero-init `spatial_head`. `predict.py` kwargs synced.
- **Result:** val/l2 = **1.2422** (epoch ~28). 30 epochs × 41 s, 5.4 GB. train 0.032 → 0.025 (normalized). Leaderboard l2=1.2422 — tied e9 (0.0008 worse). Commit `7104b10`.
- **Verdict:** kept as stepping stone (architecture promising, under-scaled). Became basis for e11.
- **Notes:** EdgeConv per epoch cost ≈ Transformer. Gradient flow was clean (no zero-grad collapse like Transolver). 30-min budget + only 3 edge blocks / k=12 / 4k anchors was too small to show the gain. Key insight: need (a) more blocks for multi-hop neighborhood reach, (b) larger k for edge diversity, (c) more anchors for denser KNN interpolation to the 100k points. All folded into e11.

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
