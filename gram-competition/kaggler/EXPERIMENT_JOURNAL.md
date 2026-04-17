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

### 2026-04-17 — v10 DISCARDED — EMA(0.999) weights regressed ~0.036
- **Hypothesis:** v6's 0.8707 was likely a lucky raw-val point at `batch_size=1` (val noise ~0.02 between epochs). EMA(0.999) tracks a moving average of weights, which should give a smoother, more reliable val curve and let the best point reflect stable generalization rather than noisy peaks. Isolating EMA from v3's combined EMA+mirror-flip failure.
- **Change:** `train.py` — added `EMA` class (shadow = clone of params, `update()` after each `optimizer.step()`, `apply()/restore()` around validation + checkpointing). `cfg.ema_decay=0.999`. Otherwise identical to v6.
- **Result:** val/l2 = **0.9067** at epoch 52 (45.7 min, 53 s/epoch, 6.2 GB). W&B project `kagent-v10`. Val descended monotonically (every epoch a new best) — EMA smoothing worked as expected for noise reduction. But the EMA val magnitude plateaued at ~0.907, well above v6's lucky 0.8707.
- **Verdict:** discarded — 0.036 worse than v6.
- **Notes:** The monotone descent is evidence EMA removed val noise. But the average level EMA settles at is higher than the best raw-val point v6 hit — i.e. the variance we smoothed away contained the 0.8707 win. At `decay=0.999` (half-life ~693 steps ≈ 1 epoch) the EMA weights trail the current model by ~1 epoch of progress; with cosine LR decaying over 60 epochs, that lag costs a few hundredths of val/l2. Possible rescue: (a) `decay=0.9995` (half-life ~2 epochs) with a longer run — likely still trails. (b) EMA only over *last N epochs* so early random init doesn't pollute. (c) raw-model checkpoint + tester-side ensembling. For now, reverting to v6. Next (v11): try the one spatial thing not yet done — multi-scale voxel (concatenate outputs of 32³ + 64³ UNets). Gives the model both long-range (32³ bigger effective receptive field) and fine (64³) spatial context. Memory+compute <2× because of how the UNet scales with grid volume.


### 2026-04-16 — v9 DISCARDED — hidden=384 per-epoch gain lost to 2.3× slowdown
- **Hypothesis:** v6 train loss kept descending (0.007 at ep52), val plateauing → generalization limited by capacity, not optimization. Bump `hidden` 256→384 (MLP width), keep everything else v6. Initial try at `epochs=60` aborted because cosine T_max=60 stretched past actual run time (same trap as v8); restarted as v9b with `epochs=30` so cosine anneals to 0 exactly at timeout.
- **Change:** `train.py` — `hidden: int = 384`, `epochs: int = 30`. Launched with `MAX_TIMEOUT_MIN=60`.
- **Result:** val/l2 = **0.9222** at epoch 30 (last, 56.3 min, 75–300 s/epoch depending on GPU contention, 8.4 GB). W&B project `kagent-v9`. Per-epoch val was clearly ahead of v6/v8 at the same epoch (ep20 v9b 0.9707 vs v6 ~1.05), confirming the capacity helps — but each epoch took 2.3× longer, so only 30 epochs fit.
- **Verdict:** discarded — 0.05 worse than v6's 0.8707. Capacity trade-off lost against wall-clock.
- **Notes:** GPU contention from shared workload caused erratic epoch times (75s best, 295s worst). Even without contention, at ~75s/epoch steady-state we could only fit ~46 epochs in 60 min; still short of v6's 52 well-annealed epochs. The capacity-gain-per-epoch is real but the time tax is too steep on this GPU share. Next (v10): isolate EMA weights (decay=0.999) from v3's combined EMA+mirror failure — EMA is a cheap, known-good variance-reduction for noisy `batch_size=1` val.


### 2026-04-16 — v8 DISCARDED — more training time alone did not beat v6
- **Hypothesis:** v6 was still descending at timeout (0.8745 → 0.8707 in last 2 epochs). Extend `epochs=60→75` and `MAX_TIMEOUT_MIN=45→60` — same architecture, same features, just more wall-clock and a cosine schedule that stays warmer longer in the ep 50-60 zone where v6 was still finding wins.
- **Change:** `train.py` — `epochs: int = 75` (no other code change). Launched with `MAX_TIMEOUT_MIN=60`.
- **Result:** val/l2 = **0.8760** at epoch 68 (69 epochs run, 60.3 min, 52s/epoch, 6.1 GB). W&B project `kagent-v8`. Descent pattern: 0.9080@ep48 → 0.8928@ep53 → 0.8796@ep62 → 0.8760@ep68 → 0.8762@ep69 (last).
- **Verdict:** discarded — 0.005 worse than v6's 0.8707, inside val noise (batch_size=1 oscillates ~0.02 epoch-to-epoch). No clear win.
- **Notes:** Why not better? v6's cosine with T_max=60 cooled LR to near-zero by ep52 and that low-LR phase is probably what locked in the 0.8707. v8's cosine with T_max=75 left LR ~3-4× higher at the same wall-clock, so it kept moving around instead of annealing into a basin. Lesson: if I want more-training wins, I should *also* keep the final-LR-near-zero phase long enough to exploit. Next (v9): the true bottleneck is almost certainly model capacity + noisy val — try `hidden=384` with T_max matching actual-run epochs (~58). If capacity helps, val/l2 should drop cleanly at any epoch count; if not, move to multi-scale voxel or batch_size=2 to reduce val noise.


### 2026-04-16 — v7 DISCARDED — wall-offset vector feature regressed ~0.036
- **Hypothesis:** scalar SDF tells the model *how far* to the wall but not *which direction*. Adding the 3D offset vector (`nearest_airfoil_pos - pos`) should give an explicit wall-normal-ish direction prior for pressure-gradient physics, on top of v6's longer training.
- **Change:** `train.py` — `compute_sdf()` also returns per-point offset vector (N,3); `SDFDataset`/`collate_sdf` carry it; model `in_dim` 21→24 with `offset_feat = offset/5.0` concatenated. `predict.py` updated to compute and pass offset. Otherwise identical to v6.
- **Result:** val/l2 = **0.9070** at epoch 51 (vs v6's 0.8707 at 52). Strictly worse the entire run (epoch 1: 1.58 vs v6 1.52; epoch 24: 1.03 vs v6 ~0.95). 52s/epoch, 6.1 GB. W&B project `kagent-v7`.
- **Verdict:** discarded — reverted to v6.
- **Notes:** 3 extra input features increased in_dim by 14% (21→24) but the scaling `offset/5.0` may have interacted poorly with early training dynamics. Offset magnitude is correlated with SDF magnitude (they measure the same thing up to a direction), so the model gets redundant signal that slows feature disentanglement. A cleaner direction prior would be a unit-normalized offset (wall-normal unit vector) separate from the scalar SDF, so the two features have independent meaning. Next (v8): try something architecturally simpler — bigger model (hidden=384) + more time, capitalizing on v6 still-descending curve.


### 2026-04-16 — v6 longer training (60 epochs, 45 min)
- **Hypothesis:** v5 was still dropping val/l2 in the last 5 epochs (0.9253 → 0.9089). Cosine LR schedule tuned to 60 epochs (so LR is still decent at epoch ~50) plus 45-min timeout should let it converge further.
- **Change:** `train.py` — `epochs=60`, MAX_TIMEOUT=45. No architecture changes.
- **Result:** val/l2 = **0.8707** at epoch 52. 45.4 min, 52s/epoch, 6.1 GB. W&B project `kagent-v6`. Mae: check W&B.
- **Verdict:** kept — clean 0.038 improvement over v5. Still descending at timeout (0.8745 → 0.8707 last 2 epochs).
- **Notes:** Confirms more time helps. Per-epoch improvement slows but doesn't plateau — more training budget would keep extracting wins. Next (v7): combine long training with a better wall-feature — replace scalar SDF with the full 3D vector to the nearest airfoil point (encodes both distance AND wall-normal direction, which drives pressure gradient physics).


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

