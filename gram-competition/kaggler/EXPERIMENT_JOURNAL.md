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

### 2026-04-16 — v5 SDF-to-airfoil feature
- **Hypothesis:** near-wall flow physics (boundary layer, pressure gradient) depend strongly on distance to the wall. The airfoil-mask bit tells the model *if* a point is on the wing, but not how far off-wall. Add per-point Euclidean distance to the nearest airfoil point as an input feature (both raw/5 and log1p-transformed so the model can key on both near-field and far-field scales).
- **Change:** `train.py` — `compute_sdf()` (GPU cdist, chunked), precompute per sample once at startup (~20s for 810 samples), `SDFDataset` wrapper + `collate_sdf`. Model `in_dim` 19→21 (added sdf_raw, sdf_log). `predict.py` does the same precompute. Arch identical to v2 otherwise.
- **Result:** val/l2 = **0.9089** at epoch 35 (vs v2's 0.9228 at epoch 31). 30.6 min, 52s/epoch, peak 6.1 GB. 7.69M params. W&B run in project `kagent-v5`.
- **Verdict:** kept — clear win of 0.014 and still descending at timeout (best epoch = last epoch).
- **Notes:** SDF gives epoch-1 val/l2 = 1.52 vs v2's 1.64 — model uses it from the start. Cost is ~20s startup + 0 per-epoch overhead (SDF is a fixed feature). Next (v6): could try multi-scale voxel, or pair SDF with Fourier-encoded pos, or train longer (still descending).


### 2026-04-16 — v4 DISCARDED — bigger UNet (voxel_mid=96) slightly worse
- **Hypothesis:** v2 was still descending at timeout (epoch 31). Doubling spatial capacity (voxel_mid 64→96, params 7.7M→15M) with more epochs (40→50) and more time (27→35 min) should push lower without aug/EMA confounds.
- **Change:** `train.py` — `voxel_mid=96`, `epochs=50`; `predict.py` updated to match.
- **Result:** val/l2 = **0.9349** at epoch 32 (vs v2's 0.9228). 35.5 min, 6.8 GB, 67s/epoch. W&B run in project `kagent-v4`.
- **Verdict:** discarded — slightly worse. Best checkpoint was the last epoch (still descending), so given more budget it might eventually beat v2, but not a convincing win.
- **Notes:** Big-model slower per step (67 vs 52 s/epoch) → fewer effective epochs in same wall-clock. Val noise pattern is the same shape as v2 — just offset. Capacity alone isn't the bottleneck. Next (v5): add a real physics feature — signed distance to airfoil — so the model has an explicit wall-distance prior.


### 2026-04-16 — v3 DISCARDED — EMA + y-mirror aug regressed ~0.06
- **Hypothesis:** stack two free wins on v2: (1) EMA(0.999) weights smooth noisy B=1 val; (2) random y-mirror augmentation doubles effective data (F1 wing is y-symmetric). Epochs=50, MAX_TIMEOUT=30 min.
- **Change:** `train.py` — added `EMA` class, update each step, swap in before validate/save. Random flip of `pos[...,1]`, `v_in[...,1]`, `v_out[...,1]` with p=0.5 during training.
- **Result:** val/l2 = **0.9861** at epoch 35 (vs v2's 0.9228 at epoch 31). Consistently ~0.05–0.08 behind v2 throughout training. 30.7 min, 6.2 GB. W&B run `7c7qljbi` (project `kagent-v3`).
- **Verdict:** discarded — reset to v2. Hurt not helped.
- **Notes:** Can't separate EMA vs mirror-flip effects in this run. Most likely culprit: y-mirror assumption may be wrong (dataset has yaw or asymmetric wing geometries → flipping invents OOD data). EMA by itself usually helps; but v3 might be stuck in a "not-converged early EMA lag" regime combined with harder targets. Next (v4): isolate by trying more capacity + more epochs without aug or EMA.


### 2026-04-16 — v2 voxel-UNet spatial context (64³)
- **Hypothesis:** v1 was a per-point MLP — zero spatial interaction. Near-wall flow depends on neighbors (wakes, pressure coupling). A 3D voxel-UNet (scatter-mean features into 64³ grid, run UNet, trilinear scatter-back) gives every point global+local context with the bottleneck giving receptive field ≫ wing chord. Residual around v1's per-point backbone so spatial block only needs to learn the correction.
- **Change:** `train.py` — added `VoxelSpatial` (scatter/gather + 3-level UNet3D, GroupNorm), inserted between 2 pre-blocks and 4 post-blocks of ResMLP. Zero-init UNet output conv → block starts as identity. Axis permutation `[2,1,0]` on grid_sample coords to match (x,y,z)↔(W,H,D). `hidden=256, voxel_res=64, voxel_mid=64`. Moved training code into `main()` so predict.py import doesn't trigger `sp.parse`. 7.69M params.
- **Result:** val/l2 = **0.9228** at epoch 31 (timeout cut), mae (Ux,Uy,Uz)=(0.624, 0.286, 0.419). 27.1 min, 52s/epoch, peak 6.1 GB. W&B run `eji6edpc`. Predictions at `predictions/apr16/alphonse/cde4a6b`.
- **Verdict:** kept — **30% improvement over v1** (1.3200 → 0.9228). Mae dropped across all components; largest in Ux (0.884 → 0.624), the hardest/largest-std axis.
- **Notes:** Smooth descent, still dropping at timeout (epoch 30: 0.9303, 31: 0.9228) — more epochs would keep winning. Val noise persists (batch_size=1). Next (v3): give it more time. Easy wins: larger unet_mid=96, per-point kNN for fine detail the 64³ voxel misses (airfoil is only ~5 voxels wide in some axes), EMA weights, 60-epoch budget with smaller MAX_TIMEOUT overhead.


### 2026-04-16 — v1 residual ResMLP + no-slip + normalized loss
- **Hypothesis:** baseline predicts absolute velocity from scratch — a residual around `velocity_in[-1]` is a much stronger starting point because frame-to-frame changes are small relative to the mean flow (~35 m/s mean Ux). Hard no-slip BC guarantees zero at airfoil. Normalized MSE loss stops the ~20 m/s Ux std from dominating the gradient.
- **Change:** `train.py` — `ResidualPointMLP` (hidden=384, n_blocks=8). Input features: normalized velocity_in (15) + pos (3) + airfoil mask (1) = 19. Output: delta in normalized space; denormalize and add to last input frame. Zero-init last linear → starts at exact persistence. Post-process no-slip mask. Loss is MSE on (pred - gt)/vel_std. Grad clip 1.0.
- **Result:** val/l2 = **1.3200** at epoch 21, mae (Ux,Uy,Uz)=(0.884, 0.375, 0.641). 26 epochs in 25 min, ~55s/epoch, peak 8.1 GB. 4.75M params. W&B run `ajszccxm`. Commit `adeebc6`.
- **Verdict:** kept — clean win vs baseline ~1.76 on mar29 val; zero-init residual made training stable from epoch 1 (epoch 1 already 1.59, below baseline's final).
- **Notes:** Val oscillates 0.05 between epochs — batch_size=1 is noisy. Loss kept dropping at end, so more epochs likely helps. predict.py broke because importing train.py triggered `sp.parse(sys.argv)` on predict's args; fixed by wrapping train.py body in `main()` + `if __name__ == "__main__":`. Per-point MLP — no spatial interaction. Next (v2): voxel-UNet spatial module.

