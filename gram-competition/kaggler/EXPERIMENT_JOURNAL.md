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

### 2026-04-17 — scale-up voxel (ch=96, blocks=6, p=512/8) + airfoil-mask + y-flip [discarded]
- **Hypothesis:** A larger voxel CNN + an explicit airfoil-occupancy channel + y-flip augmentation (geometries are near-symmetric about the x–z plane) should push past 1.00. Added all three in one run.
- **Change:** `VoxelFlowNet` with grid_ch=96, n_grid_blocks=6, point_hidden=512, n_point_blocks=8; airfoil mask scattered alongside occupancy (17 input channels); y-flip augmentation in `train.py` (50% probability per batch, flip y-coord of pos and y-component of v_in/v_out).
- **Result:** val/l2_error = **1.1446** (epoch 17 of 18 @ 31.3 min, 12.4 GB peak). Train loss only fell to 0.017 vs 0.008 for the smaller model at its final epoch.
- **Verdict:** Discarded. The bigger model is 2× slower per epoch (104 s vs 52 s), and the cosine-annealed LR is tied to `cfg.epochs=50`, so when we only complete 18/50 epochs the LR barely decays. Result: more capacity, less learning. Reset to the exp-2 checkpoint.
- **Notes:** Lessons — (a) do not scale capacity AND add augmentation in the same run; can't tell which hurt. (b) Always check whether cosine annealing will actually reach its min inside `MAX_TIMEOUT_MIN`; if the model is slower per epoch, reduce `cfg.epochs` or switch to step-based scheduling. Next: keep the smaller voxel CNN (48/ch64/b4) but add airfoil-mask + y-flip one at a time to isolate the win.

### 2026-04-17 — voxel-grid 3D CNN (G=48, ch=64, 4 dilated blocks) + point MLP
- **Hypothesis:** The point-wise MLP stalled at 1.36 because it has no spatial context — each point is treated independently. Airflow is a PDE so neighboring points matter. Give the model a real receptive field by scattering per-point features onto a [G,G,G] voxel grid, running a small 3D CNN with dilated convolutions, and trilinear-sampling the features back to each point.
- **Change:** `model.py/VoxelFlowNet` — scatter-mean `v_in` + occupancy to a `48^3` grid (fixed bbox `[-0.1,2.2]×[-0.5,0.5]×[-0.1,1.3]`), 4 residual Conv3d blocks with dilations `[1,2,4,8]` and group-norm, then `F.grid_sample` trilinear back to each point. Per-point head concats `[pos, v_in_norm, voxel_feat]` through a ResMLP (hidden=384, 6 blocks). Residual + no-slip preserved.
- **Result:** val/l2_error = **1.0049** (epoch 35 of 35 @ 30.1 min, 7.0 GB peak). Train loss 0.008 — healthy, no sign of overfitting.
- **Verdict:** Kept. 26 % better than the point-wise baseline (1.36 → 1.00). Spatial context matters a lot.
- **Notes:** Val was still dropping at the final cosine-annealed step, so longer training or a larger model likely helps. Leaders are at ~0.75 — to close that gap I'll try a bigger voxel CNN (more channels / more blocks), an explicit airfoil-occupancy channel, and y-axis-flip augmentation (the geometry is near-symmetric around the x–z plane, ~2× effective data).

### 2026-04-17 — residual + normalization + no-slip BC (point-wise ResMLP)
- **Hypothesis:** Predicting `delta = v_out - v_in[-1]` (residual) should be much easier than absolute velocity because the prediction horizon is only 5 ms (steps of 1 ms at freestream ~30 m/s). Normalize inputs with `stats.json` so Ux (std 20) doesn't dominate the MSE. Enforce no-slip at `idcs_airfoil` by zeroing predicted velocity.
- **Change:** `model.py/BaselineMLP` — point-wise ResMLP (hidden=512, 8 blocks) that outputs a normalized delta; add `v_in[-1]` back then denormalize. Loss computed in normalized space: `((pred - v_out) / vel_std).pow(2).mean()`.
- **Result:** val/l2_error = 1.3646 (epoch 20 of 23 @ 30.6 min, 10.8 GB peak). Training loss plateaued around 0.026.
- **Verdict:** Kept. Beats the copy-last-frame baseline (1.75) but nowhere near the top of the leaderboard (~0.75). Point-wise model has no spatial context — each point is treated independently, so it cannot actually learn the flow dynamics; it mostly learns a slightly smoothed version of copy-last-frame.
- **Notes:** Sanity-checked: `pred = v_in[-1]` (pure copy) also scores **1.7496** on val — my model only shaved 0.38 from copying. The leaders must be exploiting spatial context. Also fixed `predict.py: unrecognized arguments --checkpoint` by moving the model class into `model.py` so `from train import …` no longer fires the argparser. Next: voxel-grid + 3D CNN so each point can see its neighborhood.

