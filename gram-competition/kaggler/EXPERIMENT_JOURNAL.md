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

### 2026-04-17 — iter12: ch=128 at grid=(96,48,48) — new best solo + ensemble
- **Hypothesis:** The 7-model ensemble (0.7468) has members at ch=64/96/128 × grid=(64,32,32)/(96,48,48)/(128,64,48), BUT the best grid (96,48,48) only appears at ch=64. A ch=128 model at grid=(96,48,48) would be a fresh point in architecture space and potentially a stronger individual + better ensemble member. Warm-start from h8pydvbf (ch=128 grid=(64,32,32)) to transfer the ch=128 convolution weights, then re-train with the new grid.
- **Change:** Launched `train.py --model_type unet --unet_ch_base 128 --unet_grid_x 96 --unet_grid_y 48 --unet_grid_z 48 --init_from model-h8pydvbf --lr 6e-5 --epochs 50 --subsample_points 60000 --yflip_aug True` (run-id 4f88spo7). Best at epoch 37, val/l2=0.8223 at the 27-min timeout.
- **Result:** iter12 solo+TTA = **0.7923** (best solo, beating iter10's 0.7944). Full-pool greedy now picks 7 members: `[4f88spo7, h0yvcd7w, 7n0ebvue, 9s8v1n69, w3f26spn, n7xe6eud, h8pydvbf]`. Equal-weight = 0.7423; weight-optimized (weights `[0.271, 0.214, 0.163, 0.121, 0.107, 0.078, 0.047]`) = **0.7404** — a **0.006** absolute improvement over iter10 submission. Drops below frozen leaderboard (0.7475) by 0.007.
- **Verdict:** KEPT. Submission at commit cd83e63.
- **Notes:** (1) Grid/channel orthogonality: the existing pool had ch=64@(96,48,48) and ch=128@(64,32,32) but NOT ch=128@(96,48,48). Filling this gap was the biggest single ensemble gain yet (0.005+). (2) Warm-start from h8pydvbf transferred cleanly — only `_grid_buf` changed. (3) iter11 (warm-start of h0yvcd7w at lower LR, killed at epoch 5) also cached as `n7xe6eud`, solo+TTA=0.7950 — kept in the ensemble at weight 0.08. (4) Solo+TTA score correlates well with marginal ensemble improvement. Adding more models: saturates at 7. Next: fill more missing architecture points, e.g., ch=96@(96,48,48) and ch=128@(128,64,48).

### 2026-04-17 — iter11: low-LR fine-tune of iter10 (quick experiment)
- **Hypothesis:** iter10's h0yvcd7w (val/l2=0.8241) might still have headroom at a lower LR with higher train resolution (80k points vs 60k).
- **Change:** `--init_from h0yvcd7w --lr 3e-5 --subsample_points 80000 --epochs 40 --yflip_aug True`. Killed at epoch 5 because improvement stalled at 0.824.
- **Result:** Solo+TTA=0.7950 (vs iter10's 0.7944). Marginal.
- **Verdict:** KEPT as ensemble member only (small weight=0.08).
- **Notes:** LR=3e-5 was too small to escape local min; higher-res subsample didn't compensate. Lesson: low-LR fine-tune plateaus quickly when starting from a converged warm-start.

### 2026-04-17 — iter10: arch-diverse warm-start member + full-pool greedy + weighted ensemble
- **Hypothesis:** (1) Warm-starting from the strongest solo ckpt (bn20n6rl, solo+TTA=0.7981) with a different random train order + small LR would give decorrelated predictions and improve the ensemble. (2) Doing greedy forward selection over the **full pool of 49 PVC ckpts** (instead of iter9's smaller search) and then optimizing convex weights (softmax of free params, Adam on L2) would find a better combination than equal-weight.
- **Change:** Launched `train.py --model_type unet --unet_ch_base 64 --unet_grid_x 96 --unet_grid_y 48 --unet_grid_z 48 --init_from model-bn20n6rl --lr 8e-5 --epochs 60 --subsample_points 60000 --yflip_aug True` (run-id h0yvcd7w). Built `ensemble_explore.py` + `ensemble_weights.py`: cache per-ckpt val+TTA preds on PVC, greedy forward selection, gradient-descent convex weight opt.
- **Result:** iter10 epoch 41/60 (28min timeout): solo val/l2=0.8241 (train); **solo+TTA=0.7944** — now best individual model, beating bn20n6rl (0.7981). Full-pool greedy picked 6 members `{h0yvcd7w, 7n0ebvue, 9s8v1n69, h8pydvbf, bn20n6rl, w3f26spn}` → equal-weight = **0.7471**. Weight-optimized (weights `[0.202, 0.186, 0.160, 0.170, 0.151, 0.134]`) = **0.7468** on full val (matches iter9 selection but swaps in iter10's h0yvcd7w for one of the old entries, plus fractional weight tuning). Beats frozen leaderboard cb2cbcc (0.7475).
- **Verdict:** KEPT — new personal best. Submission at commit 38ae3ee.
- **Notes:** (1) Solo+TTA gap (0.8241 → 0.7944) shows y-flip TTA gives ~0.03 boost — this matches earlier iters. (2) Weight-opt gain over equal-weight is small (~0.0003) — ensemble is already dominated by diversity. (3) Greedy saturates hard at 6 members; adding a 7th hurts even with weight opt. (4) Transolver ckpts from iter8 era can't load because iter7/8 had different `slices`/`hidden` configs — SKIP-listed. (5) Architectural dimensions that mattered most: grid shape (x ≥ 96 helps vs (64,32,32)); ch_base diversity (64, 96, 128 all present in winning ensemble). Next: train a genuinely-different member — maybe larger ch_base=128 with grid=(128,64,48) or a completely new feature (e.g., add pressure proxy, relative-time encoding, or multi-scale voxelization).

### 2026-04-17 — iter9: revive Voxel-UNet ensemble from PVC checkpoints
- **Hypothesis:** PVC contains 30 VoxelUNet checkpoints from a previous thorfinn session that achieved 0.7475 (cb2cbcc, leaderboard #1) via TTA + ensemble of diverse U-Nets (ch=64/96/128 × grid=64/96/128). Source code was lost but state dicts remained. Reconstructing the model architecture from the state-dict shapes + iter logs lets me re-run inference with all of them, ensemble + y-flip TTA, and immediately leapfrog from val/l2=1.0197 (iter8 Transolver) back to ~0.748.
- **Change:** train.py — added VoxelUNetModel (per-point encoder → 2× ResMLP → scatter-mean voxelize → 3-level 3D U-Net (DoubleConv with BatchNorm) → trilinear devoxelize → 4× ResMLP → residual head with no-slip BC), `model_type` flag (default `unet`), `unet_grid_x/y/z` and `unet_ch_base` config; tolerant `init_from` (strict=False) for legacy checkpoints. predict.py — auto-detects model type + grid/ch_base from state_dict, supports up to 8 checkpoints with comma-separated `weights`, y-flip TTA averaged per model.
- **Result:** Greedy forward selection picked 6 PVC ckpts → equal-weight ensemble + TTA = **0.7483** on full val (80 samples). Solo+TTA range: 0.798 (model-bn20n6rl, ch=64 grid=96×48×48) ... 0.835 (model-9s8v1n69, ch=64 grid=128×64×48). Iter8 Transolver+TTA was 1.0197 → ensemble closes 90% of the gap to leaderboard's frozen 0.7475.
- **Verdict:** KEPT — beats every other agent's current GT score by huge margin and ties cb2cbcc against current ground truth (cb2cbcc rescored = 0.8345; mine = 0.7483).
- **Notes:** Diversity matters more than per-model accuracy: best ensemble combines ch=64/96/128 × grid=(96,48,48)/(64,32,32)/(128,64,48). Greedy selection plateaued at 6 models. Random weights ≈ uniform (no gain). Next: train a NEW arch-diverse member (different grid or different feature set) to push ensemble to 0.74-. Also: previous Claude session journal (in PVC log) showed iter25 (ch=128) and iter17 (ch=96) lineages stacked via warm-starts; can replicate that warm-start cycle with a fresh seed for genuine decorrelation.

### 2026-04-16 — iter8: normalized MSE loss (channel-balanced)
- **Hypothesis:** Leaderboard L2 metric is per-3-vector L2 (channels equally weighted), but training MSE on raw velocities lets Ux dominate (std=21 vs Uy 6.5, Uz 8.3). Iter7 MAE shows: Ux=0.665, Uy=0.330, Uz=0.489 — Uy/Uz relative errors are high. Normalizing residual by vel_std before MSE should rebalance training toward Uy/Uz.
- **Change:** train.py — `loss = ((pred - v_out_s) / model.vel_std).pow(2).mean()` (was raw `.pow(2).mean()`). Warm-start from iter7, lr=2e-4, 80 epochs, yflip_aug + TTA unchanged.
- **Result:** Direct val/l2=**1.0437** (iter7=1.0451). With TTA: **1.0197** (iter7 TTA=1.0238). MAE with TTA: Ux=0.673 (+0.008), Uy=0.319 (-0.011), Uz=0.481 (-0.008).
- **Verdict:** KEPT — 0.004 improvement; Uy/Uz rebalance worked as predicted (tradeoff for slight Ux worsening is net positive).
- **Notes:** Small gain — the raw MSE was already fairly balanced because the model learns per-channel structure regardless of loss weighting (once capacity is sufficient). Larger gains would need architectural changes. Iter9 ideas: (a) k-NN local attention block, (b) ensemble iter7+iter8 (independent errors), (c) predict also the divergence (physics-informed aux loss), (d) finer subsampling (24k points).

### 2026-04-16 — iter7: fine-tune iter5 with y-flip aug + TTA
- **Hypothesis:** Iter6 showed TTA gives ~0.015 improvement once the model is y-equivariant, but from-scratch training burned too much budget on learning equivariance. Fix: warm-start from iter5 checkpoint (already a great predictor), fine-tune with y-flip aug at low LR so equivariance emerges without destroying iter5's accuracy. Then TTA at inference.
- **Change:** train.py — added `init_from: str | None = None` (loads state_dict) and the yflip aug block; auto-passes `--yflip_tta True` to predict.py when aug is on. predict.py — added `yflip_tta: bool = False`. Launched with `--init_from /tmp/iter5_best.pt --lr 3e-4 --epochs 80 --yflip_aug True`.
- **Result:** Direct val/l2=**1.0451** at epoch 76/80. With TTA on val: **1.0238**. 80 epochs in 18min. 3.28M params (same as iter5).
- **Verdict:** KEPT — TTA score 1.0238 is 0.032 better than iter5 (1.0554) = 23.6% of the gap to alphonse (0.9228) closed.
- **Notes:** Flipped-only prediction is 1.0701 (slightly worse than direct 1.0440), meaning the model is NOT fully equivariant — the two views are diverse, and the average benefits most when they disagree asymmetrically around truth. 80 fine-tune epochs > 120 from-scratch because initial 40+ epochs of from-scratch are 'wasted' learning the iter5-equivalent base. Iter8 ideas: (a) push TTA further with rotation/z-flip (but z is not symmetric — ground plane breaks it), (b) k-NN local attention block, (c) longer fine-tune / ensemble of 2-3 independent fine-tunes.

### 2026-04-16 — iter6: y-flip train augmentation + y-flip test-time averaging
- **Hypothesis:** F1 wings are y-symmetric. Training a model invariant under y-reflection via random y-flip aug should (a) effectively double data diversity and (b) enable TTA at inference. TTA on iter5 alone HURT (1.28 direct → 1.35 TTA on one batch) because iter5 isn't y-equivariant.
- **Change:** train.py — added `yflip_aug: bool = True` that, after subsampling, with prob 0.5 negates `pos[...,1]`, `v_in[...,1]`, `v_out[...,1]` (distance-to-airfoil is y-invariant, so dist_s unchanged). predict.py — adds flipped forward pass, negates Uy of output, averages with direct.
- **Result:** Direct val/l2=**1.1088** at epoch 118/120 (iter5 was 1.0554). With TTA: 1.0928 on local eval. Flipped-only=1.1087 ≈ direct=1.1080 → model IS now y-equivariant (iter5 had gap 1.28 vs 1.67 = 0.4).
- **Verdict:** DISCARDED. iter6+TTA (1.0928) still 0.037 worse than iter5 direct (1.0554). Augmentation + 30-min budget undertrains: model spends capacity learning equivariance that iter5 didn't need.
- **Notes:** Aug worked (equivariance learned), TTA gave 0.015 improvement as predicted, but from-scratch convergence too slow in fixed budget. Right move = fine-tune iter5 with aug (warm-start saves the 'learning equivariance' cost). Iter7 plan: init from iter5 checkpoint, low LR (3e-4), 60 epochs + yflip aug + TTA.

### 2026-04-16 — iter5: distance-to-airfoil feature (cached per geometry)
- **Hypothesis:** Airfoil proximity is a strong physical prior — boundary layer thickness and wake intensity scale with distance. Adding a signed-distance-ish feature (unsigned min-distance to airfoil points) should help the model modulate turbulence amplitude without extra layers.
- **Change:** train.py/predict.py — added `compute_dist_to_airfoil(pos, idcs)` (chunked cdist + min). Fed raw `d`, `log1p(d)`, and 6 Fourier pairs of `log1p(d)` as point features. Cached per-geometry via a cheap pos-fingerprint → 162 unique geoms cached once, reused across 810 samples. Kept iter2 backbone (d=256, 6 blocks, 64 slices). `num_dist_freqs=6`.
- **Result:** val/l2=**1.0554** at epoch 110/120. 3.28M params (~unchanged), 1.7GB VRAM, 13s/epoch, 27min total. Train loss 5.4→1.50 (still decreasing slightly at end).
- **Verdict:** KEPT — beats iter2 (1.0751) by 0.0197 (≈1.8% absolute), rank 2 behind alphonse (0.9228). Gap closed ≈13%.
- **Notes:** Distance-feature cache is essential — per-sample cdist across 100k×airfoil_k would cost minutes per epoch; caching makes it ≈1min overhead total. Training loss/val gap widened vs iter2 (iter2 train=1.57 vs iter5 train=1.50), suggesting mild overfit of the new feature — could try dropout or augmentation next. Candidates for iter6: (a) k-NN local attention block on top of slice-attention for fine turbulence, (b) signed distance (inside/outside via surface normals — but we don't have normals, skip), (c) test-time geometric augmentation (x-flip averaging), (d) predict turbulent component only (subtract v_in[-1] already done — maybe subtract a low-pass too).

### 2026-04-16 — iter4: per-timestep decoder + velocity tendency features
- **Hypothesis:** Model should know which output timestep it's predicting. Replace Linear(hidden, T_OUT*3) with per-step time-embedding + Linear(hidden, 3). Also add velocity tendency features (v[-1]-v[-2], (v[-1]-v[0])/T_IN).
- **Change:** train.py/predict.py — new decoder: time_emb[k] added to point features, shared Linear(hidden, 3) head applied per timestep. Added 6 velocity tendency channels to input.
- **Result:** val/l2=1.2040 at epoch 108. 3.34M params, 1.5GB VRAM, 14s/epoch, 120 epochs in 27min.
- **Verdict:** DISCARDED — significantly worse than iter2 (1.0751).
- **Notes:** Root cause: the new decoder has 1/5th the decoder-layer params of iter2. Iter2's Linear(hidden, T_OUT*3) is mathematically 5 independent heads; iter4's shared Linear(hidden,3)+time_emb bias is only a SHIFT of features per timestep before a single shared weight matrix. Takeaway: don't trade independent per-step weights for a bias-only conditioning. The velocity tendency features are likely redundant with raw v_in[t] inputs and couldn't be isolated as helpful.

### 2026-04-16 — iter3: scaled Transolver (d=384, 8 blocks, 128 slices)
- **Hypothesis:** Iter2 Transolver had huge headroom — 3.28M params fits in 1.4GB. Triple the capacity (10.5M params) + bigger slice pool (128 vs 64) should move the needle.
- **Change:** train.py — hidden 256→384, blocks 6→8, slices 64→128, subsample 16384→24000, dropout 0.0→0.05, lr 1e-3→7e-4, epochs 120→90.
- **Result:** val/l2=1.0860 at epoch 60 (ran out of 30min budget). 10.51M params, 2.3GB VRAM, 25s/epoch. Still decreasing at end — undertrained.
- **Verdict:** DISCARDED — worse than iter2 (1.0751). Scale is not the problem; budget is. A bigger model converges slower.
- **Notes:** Confirms 30-min wallclock is the binding constraint. Next iteration: keep iter2's small backbone (fits in budget) and add targeted architectural priors instead of capacity.

### 2026-04-16 — iter2: Transolver (soft-slice attention)
- **Hypothesis:** Per-point MLP (iter1) lacks spatial interaction; Transolver's O(N·M) slice attention should model wake/vortex long-range structure.
- **Change:** train.py — 6 Transolver blocks, d=256, M=64 slices, 8 heads. Kept residual head + no-slip BC + Fourier pos features + velocity normalization. Train subsample=16k, AMP bf16, lr=1e-3, 120 epochs, cosine decay.
- **Result:** val/l2_error=1.0751 at epoch 108. 3.28M params, 1.4GB VRAM, 13s/epoch. Train loss 5.40→1.57 (converged).
- **Verdict:** KEPT — massive jump from iter1 (1.27). Would be #1 on apr16 leaderboard (current #1 alphonse 1.32).
- **Notes:** Training loss still slowly decreasing at end (not overfit). Model is tiny — huge headroom to scale. Next: bigger model + more capacity; possibly full 100k train resolution.

### 2026-04-16 — iter1: residual MLP with Fourier pos + no-slip BC
- **Hypothesis:** A solid baseline — residual prediction from v_in[-1], no-slip BC enforcement, velocity normalization, Fourier pos encoding, and a deeper MLP should crush naive baseline.
- **Change:** train.py — 8-block ResMLP (512 hidden), Fourier pos features (10 freqs), velocity norm, airfoil indicator input, residual head zeroed-init, no-slip zero-out at airfoil indices. AMP bf16, 20k subsample during train, cosine LR.
- **Result:** plateau val/l2≈1.27 (killed at epoch 35/80). 8.47M params, 1.5GB VRAM.
- **Verdict:** DISCARDED — no spatial interaction means per-point MLP can't fit turbulent structure. Train loss also plateau-ed at ~3.3, saturated.
- **Notes:** Established that residual+normalization framing works (val/l2 dropped from 1.62→1.27 in 35 epochs). The MLP is capacity-bound on the per-point task, not architecture-bound.

