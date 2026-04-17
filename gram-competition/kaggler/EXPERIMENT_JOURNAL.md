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

### 2026-04-17 — SDF-to-airfoil per-point feature [discarded]
- **Hypothesis:** Add per-point unsigned distance to the nearest airfoil point as a scalar feature concatenated with Fourier pos + voxel feat. Strong geometric prior for boundary-layer behavior (airflow close to the surface differs sharply from freestream). Implemented via chunked `torch.cdist` inside the model forward pass.
- **Change:** `model.py/VoxelFlowNet._airfoil_sdf` — chunked cdist (chunk=5000 along N) to nearest airfoil point; concat result into the per-point feature. `point_in` +1. No other changes vs exp 5.
- **Result:** val/l2_error = **0.9648** (epoch 23 of 24 @ 31.0 min, 7.6 GB peak). Essentially a tie with exp 5 (0.9640). Per-epoch time jumped 59 s → 81 s (+37 %) from the cdist overhead, so only 23 of 24 epochs completed before timeout.
- **Verdict:** Discarded. No improvement; the extra compute cost is real and crowds out other experiments. Reverted commit, restored exp 5 checkpoint.
- **Notes:** Possible reasons SDF didn't help: (a) voxel CNN at G=64 already implicitly encodes distance to the airfoil via the occupancy/scatter signal — SDF is redundant; (b) what's missing may not be geometric info but representational capacity for velocity gradients; (c) no-slip BC is already enforced on output, so the gain from knowing "I'm near the airfoil" in the hidden state may be small. Next: spend the compute budget on more resolution (G=80, ch=32 — actually cheaper than G=64/ch=48 since 80³·32² < 64³·48²) or on more epochs / deeper grid CNN.

### 2026-04-17 — higher voxel resolution G=64 ch=48 + epochs=24
- **Hypothesis:** Exp 4's voxel grid (48³ over a 2.3×1×1.4 m bbox = ~4 cm/cell) is too coarse to resolve boundary layers around the airfoil. Bump grid resolution to 64³ (~3 cm/cell) and compensate for the extra 3D-conv compute by dropping grid channels 64→48 so per-epoch time stays similar. Drop cfg.epochs to 24 so cosine still fully anneals within 30 min.
- **Change:** `train.py`+`predict.py` — `VoxelFlowNet(grid_res=64, grid_ch=48, n_grid_blocks=4, ...)`, `epochs: int = 24`. Nothing else changed.
- **Result:** val/l2_error = **0.9640** (epoch 24 of 24 @ 23.7 min, 7.6 GB peak). Train loss 0.0098. 3.2 % improvement over exp 4, ~4 % in absolute val. Finished 6 min under budget.
- **Verdict:** Kept. Biggest single-change improvement since going from point-MLP → voxel (exp 1→2). Higher spatial resolution clearly matters.
- **Notes:** Three signals suggest more room to run: (1) finished 6 min before timeout so we have compute slack, (2) val was still dropping at the final epoch (0.9700 → 0.9643 → 0.9640), (3) train loss (0.0098) is close to exp 4's final (0.0082) but val is much better — less spectral bias / better regularization. Next: bump epochs to ~30 to use the slack, OR go G=80 (ch=40 to compensate), OR add SDF-to-airfoil feature now that spatial resolution is higher.

### 2026-04-17 — Fourier positional encoding (L=8) + cfg.epochs=35 LR fix
- **Hypothesis:** (a) MLPs struggle to represent high-frequency spatial variation from raw 3-D coords (spectral bias) — adding Fourier features (sin/cos at 8 scales, 2^k·π) should let the per-point head model finer spatial structure. (b) Exp 2 reported val was still dropping at the cosine min; the min wasn't reached because cfg.epochs=50 with MAX_TIMEOUT_MIN=30 and 52 s/epoch → only ~35 epochs actually ran, so LR only decayed to ~21 % of init. Set `cfg.epochs = 35` so cosine fully anneals within the budget.
- **Change:** `model.py/VoxelFlowNet` — add `_pos_enc` method producing `[pos, sin(w·pos'), cos(w·pos')]` with 8 log-spaced frequencies, replacing raw `pos` in the per-point concat (point input grows by 48). `train.py` — `epochs: int = 35`. No other changes vs exp 2.
- **Result:** val/l2_error = **0.9956** (epoch 35 of 35 @ 30.3 min, 7.0 GB peak). Train loss 0.0082 (same as exp 2). First time below 1.0.
- **Verdict:** Kept. Small (~1 %) but real improvement; val was *still* improving at epoch 35 (34: 0.9957 → 35: 0.9956), so additional gains may be reachable with more training. Auto-submitted predictions to PVC.
- **Notes:** Bundled two changes (Fourier + LR fix) so can't isolate which helped. Next candidates: (a) bigger batch via grad accumulation for stabler gradient at the end of cosine, (b) SDF-to-airfoil per-point feature (strong geometric prior for boundary layers), (c) larger grid resolution (G=64 with ch=48 to keep compute) — should help boundary-layer resolution since 48³ over a 2.3×1×1.4 m box = ~4 cm/cell, coarser than boundary-layer thickness. Leader still at 0.7475, so 0.20 gap — need a bigger jump than Fourier+LR gave.

### 2026-04-17 — airfoil-mask channel + y-flip augmentation (small voxel 48/64/4) [discarded]
- **Hypothesis:** After exp 3's failure, isolate on the small architecture: add (a) per-voxel airfoil-occupancy channel and (b) y-flip augmentation (~2× effective data since geometries are near-symmetric across x–z plane). Keep architecture identical to exp 2 so epoch time is unchanged and cosine LR still reaches its planned min.
- **Change:** `VoxelFlowNet._voxelize` takes `idcs_airfoil` and scatters an extra airfoil-mask channel (17 input channels: 15 v_in + 1 occupancy + 1 airfoil). `train.py` y-flip aug (50% prob: flip pos[...,1], v_in[...,1], v_out[...,1]). No capacity changes.
- **Result:** val/l2_error = **1.0198** (epoch 34 of 35 @ 30.1 min, 7.0 GB peak). Train loss 0.0111 (vs exp 2's 0.008 at the same point).
- **Verdict:** Discarded. Slightly worse than exp 2 (1.0049) on both train AND val. Reset to exp 2 checkpoint.
- **Notes:** Both train AND val regressed → aug/features didn't help fitting. Most likely y-flip is the culprit: F1 front wings are close to but not exactly symmetric (asymmetric wake structure, suspension fairings), so flipping produces slightly invalid (velocity-in, velocity-out) pairs that dilute the signal. Airfoil mask alone is probably neutral-to-helpful but got blamed-by-association. Bundled two factors again — violated the lesson from exp 3. Next: try just the LR schedule fix (cfg.epochs=35 to fully anneal cosine within 30 min), a pure hygiene change with no architectural risk.

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

