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

### 2026-04-17 — iter15: Transolver hybrid + epochs=22 (match schedule to budget)
- **Hypothesis:** iter14 converged faster than iter11 (e9 val 1.10 vs 1.19; e14 val 1.03 vs 1.08) — Transolver works — but the 30-epoch cosine was too long for the 83s/ep budget, and the run hung at e17 before LR had annealed. Shortening to 22 epochs makes the schedule fully anneal within the achievable window.
- **Change:** `train.py` — `epochs: int = 22`. Everything else identical to iter14 (Transolver depth=2, slice=32).
- **Result:** val/l2 = **0.9578** at epoch 20 / 22 (best). 30.2 min wallclock, 83-98s/ep, 8.2 GB peak. Run `1y6185vx`. Preds at `/mnt/new-pvc/predictions/apr16/nezuko/35b7997/`. Trajectory beat iter8 from e16 onward (e16 val 0.98 vs iter8 0.99; e20 val 0.96 vs iter8 0.97).
- **Verdict:** **kept** — new best val. Beats iter8's 0.9670 val by 0.009 and iter14's 0.9675 by 0.01. LB score pending (with TTA, iter8 went 0.967 val → 0.9299 LB, so iter15 LB should land ~0.92).
- **Notes:** Architecture was the lever, not loss tricks. Next levers for iter16: (a) scale Transolver depth 2→3, (b) slice_num 32→48 or 64, (c) finer voxel grid (48³→64³), (d) distance-biased attention. Budget is tight: 83s/ep × 22 = 30 min — any capacity bump needs compute offset elsewhere.

### 2026-04-17 — iter14: Hybrid VoxelUNet + Transolver Physics-Attention
- **Hypothesis:** #1 thorfinn (0.79) uses Transolver; my voxel-UNet gives strong geometric prior but lacks global physics-aware context. 2 Physics-Attention blocks (M=32 slices, 8×32 heads) on per-point features after voxel-sample should give linear-in-N global mixing and close the 0.06 gap to alphonse.
- **Change:** `train.py` — new `PhysicsAttention` + `TransolverBlock`, plumbed into `VoxelUNet` (applied to `head_in=256` features between voxel-sample and head). Loss = raw MSE. Depth=2, slice_num=32.
- **Result:** best val/l2 = **1.0081** at epoch 16 (training hung during e17 — process stuck after training loop, never wrote epoch summary; killed manually after 47 min wallclock). Trajectory was great: e9 val 1.10 (vs iter11 1.19), e14 val 1.03 (vs iter11 1.08), ~0.06–0.10 ahead throughout. 83s/epoch (47% slower than iter8). 8.2 GB peak. Predictions saved to `nezuko/4e716e8`.
- **Verdict:** discarded for now (not better than iter8 LB 0.9299), but architecture is right — retry with proper schedule (iter15).
- **Notes:** (a) next run needs shorter `epochs=22` to fit 83s/ep in 30-min cap; (b) hang root-cause unclear — stale predict.py from iter11 or PVC I/O glitch. Local-only ckpt saving (ff57419) worked — we recovered e16 weights. (c) MAE at e16 val 1.01: Ux 0.67, Uy 0.33, Uz 0.47 — similar Uy/Uz ratio as iter8.

### 2026-04-17 — iter13: MSE + input velocity Gaussian noise (σ=0.05·vel_std)
- **Hypothesis:** val plateaus at 0.97–1.00 while train loss keeps dropping = clear overfit. Input-noise is a cheap regularizer that should unlock lower val.
- **Change:** `train.py` — `v_in += randn_like(v_in) * model.vel_std * 0.05` after y-flip aug. Loss reverted to raw MSE.
- **Result:** val/l2 = **1.0217** at epoch 26 / 30. 28.5 min. Run `n3o4c0tw`. Preds at `/mnt/new-pvc/predictions/apr16/nezuko/d4a45d5/val.pt`. MAE: Ux 0.665, Uy 0.345, Uz 0.477.
- **Verdict:** discarded. Worse than iter8's 0.9670. Noise σ=0.05·vel_std (≈1 m/s on Ux) is too aggressive given input is physically clean — slowed convergence without yielding the generalization bump.
- **Notes:** Val was still slowly improving at e26; more epochs might help. But the architecture is the bigger lever — research pointed to Transolver (current #1 thorfinn uses it). Iter14 = hybrid VoxelUNet + Transolver Physics-Attention blocks on per-point features.

### 2026-04-17 — iter12: L2-norm loss (matches leaderboard metric exactly)
- **Hypothesis:** iter11 over-corrected the component balance — fully normalized MSE gave equal gradient weight to all components and hurt Ux. L2-norm loss is the exact leaderboard metric and its gradient naturally balances components (a point's Uy gradient ∝ Uy_err/||err||), dampening only when the big one is large — softer, correct balancing.
- **Change:** `train.py` — `loss = (pred - v_out).norm(dim=3).mean()`. Same model/schedule.
- **Result:** killed at epoch 5 (val=1.76, barely moved from 1.77 @ e1). L2-norm's unit-magnitude per-point gradients ≈ 10–20× smaller than MSE's err-magnitude gradients → effective LR is too low, can't converge in 30-epoch budget.
- **Verdict:** discarded. Not fundamentally wrong but needs LR re-tuning to compete; not worth another 30-min run in the time budget.
- **Notes:** If revisiting: bump LR ≥5× or combine MSE + λ·L2 (MSE drives convergence, L2 does metric-aligned fine-tuning).

### 2026-04-17 — iter11: normalized MSE loss (divide by vel_std per component)
- **Hypothesis:** MSE on raw velocity weights gradient by variance; Ux std≈20 dominates Uy std≈7 / Uz std≈9. My MAE ratios (Ux:Uy:Uz = 1:0.52:0.73) are worse than leaders' (alphonse 1:0.47:0.69, thorfinn 1:0.46:0.69) — I'm relatively *most* behind on Uy. Normalizing per-component equalizes gradient weighting and should close the Uy/Uz gap directly.
- **Change:** `train.py` — `loss = ((pred - v_out) / model.vel_std).pow(2).mean()`. 1-line change, back on iter8 base (reverted iter9/iter10).
- **Result:** val/l2 = **1.0003** at epoch 24 / 30; train loss 0.0092 at epoch 29; 57 s/epoch, 5.5 GB peak. Run continued to epoch 29; best at 24.
- **Verdict:** discarded. Worse than iter8's 0.9670. Over-correction: normalized MSE weights Uy/Uz gradient by 1/std² which *over-weights* them relative to Ux, hurting the dominant component. Metric is L2 norm not per-component MSE, so the right balance is weaker than full normalization.
- **Notes:** L2-norm loss (iter12) naturally picks a softer, metric-aligned balance — that's the right correction direction.

### 2026-04-17 — iter10: stacked bottleneck self-attention (depth=3)
- **Hypothesis:** iter8's bottleneck had only 1 self-attn block; stacking 3 should give more global mixing capacity on the 12³=1728 tokens at the coarsest grid, targeted at the Uy/Uz gap.
- **Change:** `train.py` — `attn_depth=3` in `VoxelUNet`, same rest as iter8.
- **Result:** epoch 14 val/l2=1.0366 before process hung in disk wait; trajectory matches iter5 (depth=1 baseline) rather than beating iter8's 0.9670. Train loss 1.79 at e14 vs iter5 e20 was 1.02 — deeper attn converges slower.
- **Verdict:** discarded. Reverted to iter8 base.
- **Notes:** Key realization: MSE loss is mis-specified. My MAE ratio `Ux:Uy:Uz = 1:0.52:0.73` vs leaders' `1:0.47:0.69` says I'm relatively worst on Uy. Adding architectural capacity can't fix a loss that under-weights Uy/Uz gradient by a factor of ~8 (variance ratio).

### 2026-04-16 — iter9: EMA shadow model (decay=0.999)
- **Hypothesis:** iter8 plateaued at e25 and noised up. EMA averaging the last many epochs should smooth late-training variance and find a better minimum.
- **Change:** `train.py` — `ema_model = deepcopy(model)`; update after each optimizer step with decay 0.999; validate + save EMA weights.
- **Result:** val/l2 = 0.9866 at epoch 23 (EMA); 0.9610 on leaderboard (EMA + TTA).
- **Verdict:** discarded. EMA needs *longer* training for MA to catch up to online weights; with only 23 epochs reached, EMA lagged online by ~10 epochs.

### 2026-04-16 — iter7: TTA y-flip averaging on iter5 checkpoint (no retrain)
- **Hypothesis:** averaging `f(x)` and `flip(f(flip(x)))` gives a free ensemble with uncorrelated errors; should improve over iter5's 0.9867 without any retraining cost.
- **Change:** `predict.py` — replace single forward with `0.5 * (p1 + flip(p2))` over y. Training-loop y-flip aug reverted (was only in iter6).
- **Result:** val/l2 = **1.0504** vs iter5's 0.9867 — **TTA hurts by +0.064**. MAE: Ux 0.68 (was 0.63), Uy 0.35 (was 0.33), Uz 0.50 (was 0.47). All components worse.
- **Verdict:** discarded. TTA on a non-aug-trained model is worse because the flipped input is out-of-distribution; the flipped prediction is bad and averaging it in pulls quality down.
- **Notes:** Confirms that iter6's training-time aug is the right direction — model needs to learn the symmetry in training, not assume it at inference. Next iter: y-flip aug + 30 epochs + TTA (aug-trained model should tolerate flipped input, so TTA becomes beneficial again).

### 2026-04-16 — iter6: y-flip data augmentation (undertrained)
- **Hypothesis:** F1 front wing has approximate y-axis symmetry; 50% random flip of pos_y + Uy (in/out) should double effective data and close the Uy MAE gap (0.33 → 0.29 like alphonse).
- **Change:** `train.py` training loop — `if rand < 0.5: v_in[...,1].neg_(); v_out[...,1].neg_(); pos[...,1].neg_()`. Rest of iter5 config unchanged.
- **Result:** val/l2 = **1.0029** at epoch 20 / 20 (still improving). 57 s/epoch, 5.5 GB peak, 19.2 min. Run `wrgpx6iv`.
- **Verdict:** discarded for now — **undertrained, not a failure of the hypothesis**. Train loss still dropping (1.46) and val monotonically improving for the last 7 epochs. 20 epochs is insufficient for the augmented (harder) loss surface.
- **Notes:** Ideas: (a) add TTA (y-flip averaging) at inference on iter5 checkpoint — free, targets the same symmetry without retraining; (b) if TTA helps, retrain iter5 config with y-flip aug for 30+ epochs. Going with (a) first as a cheap iter7.

### 2026-04-16 — iter5: scaled VoxelUNet (base_ch=128, blocks=3, bottleneck self-attn)
- **Hypothesis:** capacity + global mixing at the 12³ bottleneck will close the 0.09 gap to alphonse. My Uy/Uz MAE (0.34/0.48) is weakest vs alphonse (0.29/0.42) — turbulent transverse components need both more features and global context, which pure local convs at 48³ with receptive field ~16 voxels can't give.
- **Change:** `train.py` — `VoxelUNet` config: `base_ch` 96→128, `blocks_per_level` 2→3, added `VoxelBottleneckAttn` (self-attn + FF over 12³=1728 flattened bottleneck tokens, 8 heads × 64 dim). `point_dim` 192→256, `head_hidden` 320→384. Fourier/scheduler unchanged.
- **Result:** val/l2 = **0.9867** at epoch 16 / 20. 56 s/epoch, 5.5 GB peak, 19.1 min total. Run `ehxwqpcz`. Predictions at `/mnt/new-pvc/predictions/apr16/nezuko/8b84f85/val.pt`. Train loss still falling at epoch 20 (1.02) while val plateaued at 0.99 since epoch 16 → nearing model saturation / mild overfit.
- **Verdict:** kept (−0.026, −2.6% vs iter4). Still #2 behind alphonse (0.9089, advanced).
- **Notes:** MAE: Ux=0.63, Uy=0.33, Uz=0.47 (alphonse: 0.61, 0.29, 0.42). Gap concentrated in Uy — spanwise axis, which should have approximate y-flip symmetry for an F1 front wing. Next iter: y-flip data augmentation (flip pos_y → −pos_y, Uy → −Uy) to double effective training set; cheap, principled, likely-large gain on the exact components I lag in.

### 2026-04-16 — iter4: Voxel-UNet (grid=48, base_ch=96) — point→3D grid→U-Net→trilinear sample
- **Hypothesis:** my Perceiver lacks spatial-locality inductive bias; turbulence is a *local* phenomenon. Alphonse leads the board at 0.9228 with voxel-unet, Transolver is #2 at 1.07 — both exploit explicit spatial structure. Scattering points onto a dense 3D grid + 3D conv U-Net + trilinear sampling back should close the obvious gap.
- **Change:** `train.py` — new `VoxelUNet` + `ResConv3d` + `voxelize` (scatter_add mean-pool) + `sample_voxel` (F.grid_sample, z/y/x order). Per-point encoder MLP over Fourier(pos, 12 bands) + v_in + v_mean + mask + time-cond → mean-pool into 48³×96 grid → 3-level U-Net (48³×96 → 24³×192 → 12³×384 → up with skip concats) → trilinear sample → concat with per-point skip → pointwise head predicts [5,3] residual on v_in[-1]. Same schedule as iter3 (grad_accum=4, warmup=300, cosine).
- **Result:** val/l2 = **1.0124** at epoch 17 / 20. 33 s/epoch (2.5× faster than iter3), 3.0 GB peak, 11.1 min total. Run `v1ent3l0`. Predictions at `/mnt/new-pvc/predictions/apr16/nezuko/e70f2a3/val.pt`.
- **Verdict:** kept (−0.33, −25% vs iter3). Moves me from #5 to #2, past thorfinn (1.0745) and askeladd (1.214).
- **Notes:** Memory still tiny (3/96 GB) — plenty of room to scale grid or channels. Alphonse at 0.9228 is within reach: I'm 0.09 away. Next iter: either (a) grid=64 + base_ch=128 + 4-level U-Net (more resolution/capacity), or (b) Transolver-style attention at the bottleneck to fuse global context with local conv features, or (c) small pointwise attention over KNN neighbors on top of voxel-sampled features for sub-voxel detail. (a) is the simplest first step.

### 2026-04-16 — iter3: larger Perceiver (L=192, d=512, 8 proc) + grad_accum=4 + warmup-cosine
- **Hypothesis:** iter2 plateaued because (a) LR was too high at end of short run, (b) grad noise from batch=1 was limiting, (c) capacity was under-utilized (only 3 GB/96 used). Bigger model + warmup + proper cosine to 20 epochs + effective batch 4 should compound.
- **Change:** `train.py` — `MODEL_CFG` bumped to `point_dim=320, latent_dim=512, n_latents=192, n_process_blocks=8, heads=8, dim_head=64`. Warmup 300 steps then cosine over the actual optimizer-step count. Gradient accumulation 4, grad-norm clip 1.0. LR schedule stepped per optimizer step not per epoch.
- **Result:** val/l2 = **1.3456** at epoch 20 (full run, 28.3 min). Steady decrease: 2.39 (warmup) → 1.69 → 1.35; smoothest convergence so far. 4.2 GB peak. Run `<iter3>`. Predictions saved to `/mnt/new-pvc/predictions/apr16/nezuko/e4ea026`.
- **Verdict:** kept (−0.06 vs iter2). Cleaner training curve, no LR-end noise.
- **Notes:** Leaderboard now tops at alphonse/v2-voxel-unet = **0.9228**, thorfinn/transolver = 1.0751, askeladd/knn-gnn = 1.214. My global-attention Perceiver lacks spatial locality inductive bias — turbulence is a local phenomenon. Next iter: voxel-UNet (scatter points → 3D grid, 3D conv U-Net, trilinear-sample back to points) — same family as current leader, closes the obvious architecture gap.

### 2026-04-16 — iter2: Perceiver-IO latent bottleneck (L=128)
- **Hypothesis:** pointwise ResMLP saturated because it cannot exchange information between neighbors; adding a Perceiver-IO latent bottleneck with L=128 learned queries will let the model aggregate global context for turbulent components.
- **Change:** `train.py` — replaced `ResMLP` with `Perceiver(point_dim=256, latent_dim=384, n_latents=128, n_process_blocks=6, heads=6, dim_head=64)`. Fourier(16 bands) pos features, time-mean velocity extra feature, time sinusoidal embed conditions the learned latent init. Encoder cross-attn + 6 self-attn processor + decoder cross-attn + residual/no-slip head.
- **Result:** val/l2 = **1.4033** at epoch 9; plateaued/oscillated 1.40–1.46 for epochs 10–11. 69 s/epoch, 3.1 GB. Run `y6e02zev`. Predictions saved to `/mnt/new-pvc/predictions/apr16/nezuko/b6d05db/val.pt`.
- **Verdict:** kept — strong improvement over iter1 (−0.43, −23%). Current #2 on leaderboard (thorfinn 1.30 leads with Transolver).
- **Notes:** Memory only 3.1/96 GB — huge room for capacity. LR schedule (cosine T_max=50) barely annealed by epoch 11 → LR too high at convergence. Oscillations likely batch_size=1 noise. Next: bigger model + grad accum + LR fit to actual epoch count.

### 2026-04-16 — iter1: physics-aware ResMLP (FiLM-t, residual, no-slip)
- **Hypothesis:** strong physics priors (residual to last input timestep, no-slip BC, per-sample pos+vel normalization, FiLM time conditioning) should close most of the easy laminar gap on top of the pointwise baseline.
- **Change:** `train.py` — replaced `BaselineMLP` with `ResMLP(hidden=384, n_blocks=8)`: FiLM-conditioned ResBlocks with sinusoidal time embedding of all 10 `t` values, residual delta prediction, airfoil-mask feature, hard no-slip BC.
- **Result:** val/l2 = **1.833** at epoch 1; train MSE plateaued at ~8.24 by epoch 2; val oscillated in 1.83–1.92 with no improvement through epoch 4. 9.4 GB peak, 62 s/epoch. Run `g5wlqt2e`.
- **Verdict:** kept as baseline (first checkpoint on disk). Killed early — pointwise model saturated without spatial interaction.
- **Notes:** Pointwise architecture can't model turbulence (no neighbor exchange). Need spatial operators. Try Perceiver-IO next (latent bottleneck, O(N·L) attention). Also consider: training-time point subsampling for more epochs, and per-component loss weighting (vel_std varies 7× between Uy and Ux, so MSE is Ux-dominated).

