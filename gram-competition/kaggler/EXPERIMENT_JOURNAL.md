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

### 2026-04-18 — gradient clipping at norm=1.0 [discarded]
- **Hypothesis:** Single-sample batch gradients are noisy. Clipping global grad norm to 1.0 caps occasional blow-ups, acting as free training hygiene. Typical gain 0.5–1 %.
- **Change:** `train.py` — add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `backward()` and `step()`. Nothing else.
- **Result:** val/l2_error = **0.9163** (epoch 28 of 28 @ 30.4 min, 8.2 GB peak). Train loss 0.0094 (vs 0.0096 exp 16). **Worse than exp 16 (0.9147) by 0.17 %** — below noise floor of n=80 val samples but technically worse.
- **Verdict:** Discarded. Roughly neutral; reverted to keep exp 16 as best.
- **Notes:** The clip *did* help early (ep 1 val 2.77 vs exp 18's 4.54, ep 8 1.08 vs exp 18's 1.14) — clipping smooths the initial volatility when EMA hasn't converged. But by ep 28 it matched exp 16 within ~0.002. Interpretation: AdamW already handles gradient scale well; clipping only shaves off the rare spikes that didn't matter for final convergence. Four straight discards (17/18/19/20) — three radical/structural (deeper U-Net, LR warmup, linear extrap) and one hygiene (grad clip). Exp 16 (U-Net + dropout + EMA + 28 ep cosine) is a strong local optimum. Next strategy: target loss-metric alignment (current normalized MSE equal-weights components, but val L2 metric is dominated by Ux which has std=20 vs Uy=7, Uz=9.5 — normalized MSE effectively over-trains Uy at expense of Ux). Unnormalized MSE gives per-component gradient scale ~ vel_std, which matches how val weights per-component errors. Single-factor change, clean single-line diff.

### 2026-04-18 — linear velocity extrapolation as residual baseline [discarded]
- **Hypothesis:** Change the residual anchor from `pred = last + delta·vel_std` to `pred = last + (last − prev)·t + delta·vel_std` — give the network a better zero-order prediction (first-order temporal extrapolation) so it only has to learn the 2nd-order correction. In steady freestream, `last − prev ≈ 0`, so extrap ≈ last (no loss). In unsteady wake, extrap captures local acceleration.
- **Change:** `model.py:207-213` — compute `dv = last − velocity_in[:,-2:-1]`, broadcast with `dt_offsets = [1..5]`, form `v_extrap = last + dv · dt_offsets`, then `pred = v_extrap + delta_norm · vel_std`.
- **Result:** val/l2_error = **1.0089** (epoch 28 of 28 @ 30.4 min, 8.2 GB peak). Train loss 0.014. **Worse than exp 16 (0.9147) by 10.3 %**. Every epoch 0.07–0.13 worse than exp 18 (which was already discarded). Trajectory NEVER caught up to exp 16's baseline.
- **Verdict:** Discarded. Reverted; restored exp 16 checkpoint.
- **Notes:** The linear extrapolation is WRONG for most points — it assumes constant acceleration, but CFD flows have highly non-linear time evolution (vortex shedding, separation, etc.). For wake points where `dv` is large, the extrap OVERSHOOTS by a factor ~5 (`dv · 5` at t=5 is 5× the actual velocity change, because flows decelerate as they evolve). The network then spends most of its capacity CANCELLING the bad prior, leaving nothing for learning genuine dynamics. Train loss confirms: 0.014 vs 0.0096 for exp 16 — residual magnitude stayed ~1.5× higher throughout. Lesson: hardcoded priors only help if the prior is approximately correct on the data distribution; "physically plausible" ≠ "numerically close to truth" for CFD. Three straight discards (17/18/19) — need to stop adding capacity/priors and instead tune what's working. Next: (a) unnormalized MSE loss — match val L2 weighting (current normalized MSE gives equal weight to tiny-std components that don't matter for val metric); (b) grad clipping 1.0 as hygiene; (c) point_hidden 384 → 512 as pure capacity.

### 2026-04-18 — LR warmup 1 ep linear + 27 ep cosine [discarded]
- **Hypothesis:** Adam's second-moment variance estimate is poorly calibrated at step 0 (few samples). Warming LR up linearly over 1 epoch (0.01× → 1× peak) before cosine should reduce early optimization instability and let the cosine hit a slightly better minimum by end of budget. Standard "training hygiene" win across many CNN/transformer recipes.
- **Change:** `train.py` — replace `CosineAnnealingLR(T_max=MAX_EPOCHS)` with `SequentialLR([LinearLR(start=0.01, iters=1), CosineAnnealingLR(T_max=MAX_EPOCHS-1)], milestones=[1])`. Nothing else.
- **Result:** val/l2_error = **0.9363** (epoch 28 of 28 @ 30.3 min, 8.2 GB peak). Train loss 0.0106 (vs 0.0096 exp 16 — model trained less). **Worse than exp 16 (0.9147) by 2.4 %**. Trajectory stayed ~0.03 worse than exp 16 the whole run, never caught up.
- **Verdict:** Discarded. Reverted; restored exp 16 checkpoint.
- **Notes:** Two likely mechanisms: (a) The first epoch at 0.005–0.5× peak LR is nearly wasted for the actual weights, but the EMA (decay=0.999, need ~1000 steps = 2 epochs to converge) started from random init and lost its first-epoch mixing opportunity — EMA at ep 1 val was 4.54, vs ~3 for a normally-initialized run. (b) Losing 1 full epoch of peak-LR training from a 28-epoch budget is a real compute hit here. Lesson: warmup is a free win with *many* epochs, but at 28 epochs it's a net loss. If we ever run ≥50 epochs it's worth revisiting. Not a blanket "warmup doesn't work" — it's "warmup + EMA at short budget doesn't work." Next candidates: (a) stride-2 conv downsampling (learned, not avg_pool3d); (b) grad clipping (gradient norm at 1.0); (c) point hidden 384 → 512 (capacity — risk is throughput); (d) multi-timestep loss weighting (harder future timesteps get higher weight).

### 2026-04-18 — U-Net 4-level (G/8 bottleneck) [discarded]
- **Hypothesis:** Adding a fourth U-Net level (bottleneck at G/8 = 10 voxels across 2.3 m bbox) gives the model a global receptive field over the whole domain — useful for capturing freestream context + large-scale wake. Compute at G/8 is tiny (10³×256² < 2 GFLOPs), so per-epoch should barely grow.
- **Change:** `model.py` — add `enc3` at G/8 with 8×grid_ch channels + `dec2` at G/4 (skip from enc2). Pool/interp chain grows to 3 levels deep.
- **Result:** val/l2_error = **0.9226** (epoch 27 of 28 @ 30.0 min, 8.3 GB peak). Train loss 0.0097. **Worse than exp 16 (0.9147) by 0.9 %**. Lost the final epoch (27 of 28) — exp 16 completed all 28.
- **Verdict:** Discarded. Reverted; restored exp 16 checkpoint.
- **Notes:** Likely causes: (a) G/8 bottleneck = 23 cm/cell is too coarse for the airflow structures that matter — most flow variation happens at 2–6 cm scales; squeezing through this bottleneck destroys information that the skip connections can't recover; (b) extra params (the 8ch-level has 256×256×27 weights per conv = 1.8 M new params) need more epochs to fit but we're budget-limited. Lesson: deeper U-Net isn't free — the bottleneck has to match the feature-scale distribution of the data. Next candidates: (a) stride-2 conv downsampling (learned instead of avg_pool3d); (b) LR warmup (1 ep linear → 27 ep cosine) for training stability — untested hyper-hygiene; (c) grad clipping.

### 2026-04-18 — U-Net voxel CNN (3-level multi-scale)
- **Hypothesis:** Current 4 dilated same-res blocks at G=80 have limited multi-scale reasoning — the 3D CNN sees only fixed-resolution voxel features. A U-Net (G → G/2 → G/4, with skip concatenations) gives proper multi-scale: bottleneck at G/4 (~12 cm/cell) for global context, skip paths for fine detail at G=80 (~2.9 cm/cell). Napkin math says U-Net at (32, 64, 128) ch is slightly cheaper than current 4 × 32-ch same-res blocks because each deeper level has 8× fewer voxels than the one above.
- **Change:** `model.py` — new `GridConvBlock` (two 3×3×3 + GN + GELU, residual with 1×1 channel-projection). `VoxelFlowNet` replaces `self.grid_blocks` ModuleList with `enc0/enc1/enc2/dec1/dec0` and forward does `avg_pool3d` downsampling + `F.interpolate(..., mode="trilinear")` upsampling + skip concat. `grid_in` → `enc0` → down → `enc1` → down → `enc2` (bottleneck) → up → cat(e1) → `dec1` → up → cat(e0) → `dec0` → interp to points.
- **Result:** val/l2_error = **0.9147** (epoch 28 of 28 @ 30.4 min, 8.2 GB peak). Train loss 0.0096. 0.35 % improvement over exp 15 (0.9179 → 0.9147). Per-epoch time 65 s (vs 67 s at 4-block dilated — slightly cheaper as predicted; fit all 28 epochs vs exp 15's 27).
- **Verdict:** Kept. The full 28-epoch run + multi-scale features together recovered one lost epoch AND added a modest architectural gain.
- **Notes:** Val still dropping at ep 28 (0.9154 → 0.9150 → 0.9147), so cosine not quite fully tapped. Early trajectory lagged exp 15 through ep 12 (U-Net adapts slower initially — more params to calibrate), then overtook from ep 22 onward. Peak memory actually dropped (8.5 → 8.2 GB) since the top-level feature tensor shrank from 4×80³×32 to 2×80³×32 (enc0 + dec0 only, vs 4 same-res blocks). Next candidates: (a) bump grid_ch 32→40 (costs ~1.56× voxel compute — likely won't fit at 28 ep); (b) 4-level U-Net with bottleneck at G/8 = 10 (more global context, cheap); (c) cfg.epochs=30 if per-epoch holds at 65s (30×65 = 32.5 min — too tight); (d) widen point head with savings elsewhere; (e) stride-2 conv instead of avg_pool3d (learned downsampling).

### 2026-04-18 — dropout p=0.15 (bump from 0.1)
- **Hypothesis:** Exp 14 (p=0.1) still had train loss 0.0096 vs val l2 0.9193 — gap remained, regularization might still be under-applied. Single-factor test: bump `point_dropout` 0.1 → 0.15, nothing else.
- **Change:** `train.py`+`predict.py` — `point_dropout=0.15`. Identical trajectory through ~epoch 15; consistent small lead thereafter.
- **Result:** val/l2_error = **0.9179** (epoch 27 of 28 @ 30.3 min, 8.5 GB peak). Train loss 0.0096. 0.15 % improvement over exp 14 (0.9193 → 0.9179).
- **Verdict:** Kept. Marginal but consistent across training (val lower from ~ep18 onward, monotonic through final).
- **Notes:** Gain is right at the noise floor for 80 val samples, but val kept dropping by 0.0004–0.0006/epoch through ep 27, suggesting either more epochs or p=0.2 might still help. Given diminishing returns from tuning dropout, next experiment should be structural rather than hyperparameter. Gap to leader 0.75 now 0.17. Next: U-Net voxel CNN (multi-scale features, similar compute to current 4-block dilated CNN per napkin math) is the most promising structural change that hasn't been tried.

### 2026-04-17 — dropout p=0.1 in point ResBlocks
- **Hypothesis:** At the end of exp 12 training loss was 0.0088 (very low) while val/l2 plateaued at 0.9277 — the point MLP is likely fitting noise in the per-point mapping. Add standard MLP dropout (p=0.1, after GELU in each ResBlock) to regularize the point head. No other changes.
- **Change:** `model.py/ResBlock` — optional `dropout` param (default 0) inserted as `nn.Dropout(p)` between GELU and second Linear. `VoxelFlowNet.__init__` — new `point_dropout` arg wired to ResBlocks. `train.py` + `predict.py` — `point_dropout=0.1`.
- **Result:** val/l2_error = **0.9193** (epoch 27 of 28 @ 30.3 min, 8.5 GB peak). Train loss 0.0096. **0.9 % improvement over exp 12 (0.9277 → 0.9193)**, even though dropout added ~4 s/epoch and cost the final epoch (27 of 28 ran).
- **Verdict:** Kept. Clear, single-factor win; regularization is underexploited even at modest p=0.1.
- **Notes:** Per-epoch cost: 63 s → 67 s (+6 %), pushed finish to 30.3 min which tripped the 30-min MAX_TIMEOUT check after epoch 27 (epoch 28 would've completed at ~31.4 min). Val kept improving through ep 27 (0.9202 → 0.9197 → 0.9193), so cosine not fully annealed either — another 0.001–0.002 likely available if we drop `cfg.epochs` to 27 so the schedule fully anneals within 30 min. Train/val gap still large (train 0.0096 vs val equiv much higher), so p=0.15 or 0.2 may help further. Gap to leader 0.75 now 0.17. Next candidates: (a) higher dropout (p=0.15) with cfg.epochs=27 to stay in budget cleanly, (b) multi-factor: dropout + deeper point head (since regularization was the bottleneck, capacity is now uncorked), (c) structural: U-Net voxel CNN for multi-scale features.

### 2026-04-17 — trilinear splatting voxelization [discarded]
- **Hypothesis:** At G=80 most cells are empty (512k cells vs 100k points ⇒ ~80 % empty) and hard-scatter creates sharp cross-cell discontinuities. Trilinear splatting distributes each point's contribution to its 8 surrounding cell centers with trilinear weights summing to 1 → fewer empty cells and smoother grid features. Mean-pool uses splat; max-pool stays hard-assigned (max is not linear).
- **Change:** `model.py/_voxelize` — replace single `scatter_add_` for mean with 8-fold splat loop over (dx,dy,dz) ∈ {0,1}³, accumulating weighted feature and weight sums; normalize by weight sum.
- **Result:** val/l2_error = **0.9404** (epoch 26 of 28 @ 29.7 min, 8.0 GB peak). Train loss 0.0096. **Worse than exp 12 (0.9277) by 1.4 %**. Per-epoch time essentially unchanged (63 s → 63–64 s).
- **Verdict:** Discarded. Consistent ~0.02 lag vs exp 12 from epoch 3 onward, on both train and val. Reverted; kept exp 12 checkpoint.
- **Notes:** Two plausible explanations: (a) the CNN's 3×3 conv + `F.grid_sample` trilinear readout already smooth across cells, so pre-smoothing the grid costs information without adding any the network couldn't already get; (b) splatting dilutes the per-cell signal-to-noise (one point's energy is spread across 8 cells, so any single cell sees mostly "near-neighbor noise" with smaller coefficients from actually-close points) — hard scatter preserves the cleanest single-cell reading. Lesson: when a downstream module is already smoothing, pre-smoothing the input usually loses information. Next: try something that changes *what the model sees*, not *how smoothly it sees it*. Candidates: dropout in the point MLP (regularize the train/val gap), weighted MSE (upweight hard points near the airfoil), point-MLP capacity bump (hidden 384→512, cost <10 % per epoch).

### 2026-04-17 — EMA weight averaging (decay=0.999)
- **Hypothesis:** Cosine-annealed weights near the end of training still oscillate around a local optimum because of the stochastic gradient; averaging the last ~1/(1-decay) ≈ 1000 optimizer steps should land in a flatter region and reduce val noise. Near-zero compute cost (one extra float tensor per parameter and a fused mul-add per step), so cfg.epochs stays at 28.
- **Change:** `train.py` — `EMA` class (decay=0.999) updated every optimizer step. At validation time, swap EMA weights into the model, run validate, and if best, save **the EMA weights** to `best.pt`. Swap base weights back before next epoch's training.
- **Result:** val/l2_error = **0.9277** (epoch 28 of 28 @ 29.5 min, 8.0 GB peak). Train loss 0.0088. 0.3 % improvement over exp 9 (0.9304 → 0.9277).
- **Verdict:** Kept. Small but real, with zero extra epoch time and no architectural risk.
- **Notes:** EMA visibly lagged the non-EMA trajectory through epoch ~12 (shadow is still catching up from init — half-life ≈ ln(0.5)/ln(0.999) / 63 steps ≈ 11 epochs). By epoch 15 it caught up (0.9611 ≈ exp 8's best), then edged past every subsequent epoch. Val dropped monotonically through ep 28 (0.9283→0.9281→0.9278→0.9278→0.9277) — plateau consistent with LR fully annealed. Decay=0.9995 or cfg.epochs=35 might extract more. Gap to leader 0.75 is still 0.18. Next candidates that don't cost epochs: dropout in the point MLP (regularization), weighted MSE (emphasize near-airfoil / high-velocity points), or a second head on pressure-gradient-like features. If willing to trade epochs: point_hidden 384→512 or n_point_blocks 6→8 (sub-10 % per-epoch cost — can likely absorb within budget).

### 2026-04-17 — grid dilations extended to (1,2,4,8,16), n_grid_blocks=5 [discarded]
- **Hypothesis:** Adding a 5th grid block with dilation=16 increases effective receptive field of the 3D CNN from ~30 cells to ~60 cells (at G=80 voxels), useful for capturing large-scale wake structure that forms downstream of the wing. Drop cfg.epochs=28→25 to keep cosine fully annealed within budget.
- **Change:** `model.py` — replace hardcoded `dilations=[1,2,4,8]` with configurable tuple, bump default to `(1,2,4,8,16)` and `n_grid_blocks=5`. `train.py` — `epochs: int = 25`.
- **Result:** val/l2_error = **0.9459** (epoch 25 of 25 @ ~30 min). Worse than exp 9's 0.9304.
- **Verdict:** Discarded. Extra grid capacity cost ~3 epochs (28→25), and the gain wasn't enough to compensate. Reverted commit, restored exp 9 checkpoint.
- **Notes:** Two-factor change (extra block + fewer epochs) — can't isolate. Most likely the receptive field of exp 9 (dilations 1,2,4,8 → ~30 cells × 2.9 cm = ~90 cm) was already sufficient for the relevant spatial scale; the wake structure at larger scales may not be predictable from 5 ms of history anyway. Lesson reinforced: at the 30-min budget, capacity increases that cost epochs are usually net-negative.

### 2026-04-17 — test-time augmentation via y-flip ensemble [discarded]
- **Hypothesis:** The geometry is near-symmetric about the x–z plane. Even if the model isn't trained to be exactly y-symmetric (exp 3b showed training-time y-flip hurt), averaging predictions from `(v_in, pos)` and the y-flipped version at inference time might reduce variance.
- **Change:** Standalone `eval_tta.py` (not committed) running exp 9 checkpoint with and without y-flip TTA on val split.
- **Result:** TTA val/l2 jumped from 0.9304 to ~1.03. Non-trivial regression.
- **Verdict:** Discarded. Flipping creates out-of-distribution inputs (model is not y-symmetric), so averaging pulls predictions toward a worse operating point. Consistent with exp 3b's training-time y-flip failure.
- **Notes:** Data itself may be more asymmetric than visual inspection suggests (asymmetric wake, suspension fairings, mirror effects). Don't revisit y-symmetry unless there's an explicit symmetrization during training.

### 2026-04-17 — longer training cfg.epochs=28 (use 5-min slack)
- **Hypothesis:** Exp 7 and exp 8 both finished with val still dropping and ~5 min under the MAX_TIMEOUT budget. Pure-training-time test: does bumping `cfg.epochs` from 24 to 28 (which fully anneals the cosine schedule at epoch 28 rather than 24) give measurable improvement?
- **Change:** `train.py` — `epochs: int = 28`. Nothing else.
- **Result:** val/l2_error = **0.9304** (epoch 28 of 28 @ 29.4 min, 8.0 GB peak). Train loss 0.0087. 2.4 % improvement over exp 8 (0.9533 → 0.9304).
- **Verdict:** Kept. Big bang for zero architectural change — the bottleneck for the last two experiments was LR not fully annealing within budget.
- **Notes:** Budget is now nearly fully used (29.4/30 min). Val still dropping slowly at the end (27: 0.9310 → 28: 0.9304, only −0.0006); LR is fully annealed so additional epochs would require a new lr restart or larger lr. Next candidates: (a) capacity in the per-point head (point_hidden 384→512 or n_point_blocks 6→8 — cheap), (b) data augmentation (y-flip alone to re-test the exp-3b confound), (c) test-time augmentation (average predictions on original + y-flipped input at inference). Gap to leader 0.75 now 0.18.

### 2026-04-17 — scatter mean + max voxelization
- **Hypothesis:** Scatter-mean loses within-cell variation. Concatenating a scatter-max projection lets the 3D CNN see both the typical and the most extreme velocity in each voxel — useful for turbulent / separation regions where peak values carry information about mixing.
- **Change:** `model.py/_voxelize` — alongside the mean, run `scatter_reduce_(reduce="amax", include_self=False)` on the 15 v_in channels. Concat mean+occupancy (16 ch) with max (15 ch) → 31 grid input channels (was 16). `grid_in = Conv3d(31, grid_ch, 1)`.
- **Result:** val/l2_error = **0.9533** (epoch 24 of 24 @ 25.2 min, 8.0 GB peak). 0.1 % improvement over exp 7 (0.9547). Per-epoch time unchanged (63 s).
- **Verdict:** Kept. Tiny but consistent-direction improvement; the extra capacity is effectively free (only grid_in has more params, and voxelization isn't the bottleneck). Leaves room to push.
- **Notes:** Val still improving at epoch 24 (22: 0.9575 → 23: 0.9564 → 24: 0.9533). Finished ~5 min under budget, same as exp 7. This is a consistent signal that more epochs is on the table without any other change. Next: exp 9 = just bump `cfg.epochs=28` to spend the slack. After that, consider qualitative changes (k-NN features, U-Net grid CNN, ensembles).

### 2026-04-17 — higher voxel resolution G=80 ch=32
- **Hypothesis:** Exp 5 showed higher grid resolution helps. Go further: G=80 (~2.9 cm/cell), compensate by dropping channels 48→32. Compute budget: 80³·32² = 524 B ops < exp 5's 64³·48² = 604 B, so per-epoch time should be similar or lower.
- **Change:** `train.py`+`predict.py` — `grid_res=80, grid_ch=32`. Nothing else changed vs exp 5 (epochs=24, Fourier L=8, 4 dilated grid blocks, point_hidden=384, n_point_blocks=6).
- **Result:** val/l2_error = **0.9547** (epoch 24 of 24 @ 25.1 min, 7.9 GB peak). Train loss 0.0101. 1 % improvement over exp 5 (0.9640 → 0.9547). Finished ~5 min under budget.
- **Verdict:** Kept. Pushing spatial resolution keeps paying off. 63 s/epoch as expected.
- **Notes:** Val was still dropping at final epoch (23: 0.9706 → 24: 0.9547, big drop). With 5 min slack I could bump cfg.epochs 24→28. Next experiment just uses the slack (exp 8 = epochs=28) to probe whether more training helps. After that: either go even higher resolution (G=96) or add a qualitative feature (k-NN neighborhood, temporal derivatives). Gap to leader 0.75 is still 0.20 — pure resolution scaling is showing diminishing returns (+3.2 % then +1 %).

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

