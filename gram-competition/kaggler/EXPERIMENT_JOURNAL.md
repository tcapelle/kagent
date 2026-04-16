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

