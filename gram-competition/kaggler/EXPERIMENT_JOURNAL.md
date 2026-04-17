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

### 2026-04-17 — Rel-vector-to-airfoil feature + 5-model ensemble (iter 16, KEPT)
- **Hypothesis:** (A) SDF alone is scalar; pos-rel-vector = pos − nearest_airfoil_pos adds direction (upstream vs downstream, above vs below suction side) on top of the same distance signal. Zero-init additive branch preserves warm-start. (B) iter 13–16 are a warm-start chain with correlated errors, but weighted averages of several warm-started checkpoints should still decorrelate a bit and ensemble for ~free.
- **Change:** `train.py` — added `compute_sdf_and_rel()` returning (sdf, pos−nearest); new `rel_embed: MLP(27→H→H)` on Fourier(L=4)-encoded rel vector, final layer zero-initialised and added as a residual exactly like sdf branch. Also moved SDF computation to be optional-precomputed so the point-subsampling path still works. Added `validate(...)` call BEFORE training begins when `--resume` is set so we never overwrite `best.pt` with a worse warm-start E1 (iter 16 v1 subsample-40k attempt silently clobbered iter 15's 1.066 with 1.10 — fixed). Also added `eval.py` for quick one-off ensemble scoring on val. Finally, `train.py` auto-triggers ensemble predict if a `/mnt/new-pvc/.../ensemble_ckpts/iter*.pt` directory exists; writes all prior-iter checkpoints + current one through `predict.py --checkpoints`.
- **Result:** single-model iter16 best = **1.0628** at E10/21 (31 min). Warm-start init 1.0656, E1 regressed to 1.073 (warmup + new branch learning), crossed floor at E10, noisy plateau at 1.064 thereafter. Train loss hit 0.77 (vs val 1.06 — deepening overfit). **Ensemble scan** (eval.py):
  - iter 16 alone: 1.0628
  - iter 15 + 16: 1.0594 (-0.3%)
  - iter 14 + 15 + 16: 1.0573 (-0.2%)
  - iter 13 + 14 + 15 + 16: 1.0555 (-0.2%)
  - **iter 12 + 13 + 14 + 15 + 16: 1.0537 (-0.2%)** ← submitted
  - iter 10 + 12 + 13 + 14 + 15 + 16: 1.0580 (+0.4%, iter10 at 1.22 is too weak, hurts)
- **Verdict:** KEPT. **Single 1.0656 → 1.0628 = 0.3% (rel branch), 1.0628 → 1.0537 = 0.9% (5-model ensemble).** Combined 1.1% drop. Cumulative since iter 10: **1.2218 → 1.0537 = 13.8%.**
- **Notes:** Ensemble gain plateaued once members weaker than ~1.15 were included (iter 10's 1.22 was too far off and hurt the mean). Warm-started chains do have *some* error decorrelation even though they share a warm-start path — not pure noise but enough to help. Every future iteration will auto-include the ensemble path via `ensemble_ckpts/` dir on PVC, so no iteration "loses" the 0.9% ensemble gain. Remaining val-L2 gap to #1 (thorfinn 0.81) is still 24%; need a bigger architectural lift. Ideas for iter 17: (a) wider model (hidden=512, n_blocks=10) trained from scratch 25 epochs — would add a *genuinely* decorrelated ensemble member; (b) second Transolver stage with a smaller M (16 slices) stacked after the current trunk to model finer substructure; (c) semi-Lagrangian latent-warping temporal decoder (warp features along v_last per output step); (d) surface-normal feature in addition to rel vector. (a) has the biggest ceiling and would be the strongest ensemble partner.

### 2026-04-17 — SDF-to-airfoil input feature + warm-start (iter 15, KEPT)
- **Hypothesis:** Model has near-zero explicit geometric awareness — airfoil index is only used as an output mask. Adding a signed-distance-to-nearest-airfoil-point feature should directly encode wake/boundary-layer proximity, the dominant structural signal for F1 wing flows. Use a *zero-init additive branch* (`sdf_embed` MLP with final layer zero-initialized) so warm-start from iter 14's checkpoint starts at the exact same output, then the branch learns its contribution.
- **Change:** `train.py` — added `compute_sdf(pos, idcs_airfoil, cache)` (chunked cdist, cached by pos-hash so each of ~146 unique train geometries is computed once). Added Fourier(L=4) on the SDF to capture multi-scale decay (9-dim feature). Added `sdf_embed: MLP(9 -> hidden -> hidden)` with zero-init on the last layer. In forward: `x = proj_in(...) + sdf_embed(fourier(sdf))`. Warm-start via `load_state_dict(strict=False)` skips missing sdf keys. Also bug-fix: `train.py` now skips the in-repo `best.pt` copy in `--debug` mode (a previous debug run silently clobbered the good checkpoint).
- **Result:** best val/l2_error = **1.0656** at epoch 17/21 (31 min). Trajectory: 1.091 → 1.084 → 1.089 → 1.087 → 1.084 → 1.082 → 1.076 → 1.081 → 1.083 → 1.073 → 1.072 → 1.072 → 1.076 → 1.066 → 1.067 → 1.067 → **1.066** → 1.066 → 1.066 → 1.066 → 1.066. Clean plateau after E17.
- **Verdict:** KEPT. **1.0752 → 1.0656 = 0.9% improvement.** Cumulative from iter 10: **1.2218 → 1.0656 = 12.8% drop**.
- **Notes:** First epoch 94s (SDF cache cold), rest 88s. Train loss reached 0.82 (vs val 1.07) — deeper overfit but val still improved beyond iter 14. The SDF branch visibly helps even when zero-init: LR warmup nudges both existing weights (off-optimum by design for joint learning) AND the new branch (building up SDF contribution). Next (iter 16): the diminishing-returns pattern of warm-start+minor-tweak means future gains probably need a bigger architectural change. Options: (a) wider model (hidden=512) — can't warm-start cleanly; (b) velocity-at-nearest-airfoil-point feature (richer than just SDF); (c) test-time ensemble across multiple checkpoints (iter 13, 14, 15); (d) train a wider model from scratch with all the tricks already locked in.

### 2026-04-17 — Stochastic Sobolev/gradient-matching loss (iter 14, KEPT)
- **Hypothesis:** Iter 13 hit an overfitting wall at val≈1.087 (train 0.87 vs val 1.09). Adding a spatial-gradient-matching loss should regularize toward physically smooth fields and reduce the gap. Use a stochastic approximation: at each step pick 1024 random anchor points, compute kNN (k=8) in position space, and penalize `|| (pred[nei] - pred[anchor]) - (gt[nei] - gt[anchor]) ||`.
- **Change:** `train.py` — added `sobolev_anchor_loss()` (chunked cdist on [1, 1024, 100k] + topk + gather + diff-match). Added `--sobolev_lambda/--sobolev_anchors/--sobolev_k` args. Combined `loss = l2 + λ * sob`. Only ~3ms extra per step after CUDA warmup.
- **Result:** best val/l2_error = **1.0752** at epoch 15/21 (30 min). `--lr 1e-4 --warmup_steps 30 --sobolev_lambda 0.1`. Warm-started from iter 13's best.pt (val=1.0867 init).
- **Verdict:** KEPT. **1.0867 → 1.0752 = 1.1% improvement.** Trajectory: 1.0948 (E1, regressed from init 1.087) → 1.089 → 1.091 → 1.095 → 1.093 → 1.090 → 1.086 → 1.083 → 1.082 → 1.084 → 1.080 → 1.078 → 1.077 → 1.079 → **1.075** → 1.078 → 1.077 → 1.076 → 1.076 → 1.076 → 1.076. Val took 6 epochs to recover from Sobolev reshaping, then surpassed iter 13 at E7 (1.086).
- **Notes:** E1 regressing above warm-start init is expected — adding a new objective pulls weights off the L2-only optimum before settling into the joint optimum. Plateau from E15 onwards suggests we're at the Sobolev-regularised ceiling for this architecture. Cumulative since iter 10: 1.2218 → 1.0752 = **12% drop**. Next (iter 15): either (a) add a signed-distance-to-airfoil (SDF) input feature — richest geometric signal we're missing (use warm-start with zero-init new input weight column), or (b) stronger Sobolev (λ=0.2) with longer cosine. (a) is higher-risk, higher-reward.

### 2026-04-17 — Warm-start round 2 (iter 13, KEPT)
- **Hypothesis:** Iter 12 was still descending at timeout (1.1398, lr had only decayed to 1.2e-4). A second warm-start round with even lower peak LR (1.5e-4) and shorter warmup (30 steps) should squeeze more descent out of the same checkpoint.
- **Change:** same `--resume` arg, launched with `--lr 1.5e-4 --warmup_steps 30 --epochs 25`.
- **Result:** best val/l2_error = **1.0867** at epoch 18/22 (31 min). Descent: 1.138 → 1.135 → 1.126 → 1.124 → 1.123 → 1.116 → 1.111 → 1.107 → 1.106 → 1.100 → 1.100 → 1.095 → 1.093 → 1.093 → 1.090 → 1.090 → 1.088 → **1.087** → 1.088 → 1.087 → 1.088 → 1.087. Plateaued after E18.
- **Verdict:** KEPT. **1.1398 → 1.0867 = 4.7% improvement.** Cumulative from iter 10: **1.2218 → 1.0867 = 11% drop.**
- **Notes:** Train-val gap widened (0.87 train vs 1.09 val) — clearly overfitting, but val still improved until E18. At the overfitting wall. Further warm-start rounds at even lower LR will give <1% gains. Next (iter 14): add Sobolev/gradient-matching loss (kNN diff matching, λ≈0.1) — directly attacks the generalization gap by forcing the model to match local spatial gradients, not just point values. Cache kNN once per geometry. Alternative: wider model, but that requires fresh training (can't warm-start across architecture change).

### 2026-04-16 — Warm-start fine-tune from iter 10 (iter 12, KEPT)
- **Hypothesis:** Iter 10 was still monotonically descending at its timeout (1.22 at epoch 22/22). The phase transition at epoch 3 "wastes" 3/22 epochs. Load iter10's best.pt as init, use lower peak LR (3e-4 vs 1e-3) and short 50-step warmup so we immediately resume descent from the already-trained regime.
- **Change:** `train.py` — added `--resume <path>` arg; if set, load `state_dict` after model init. Launch: `--resume checkpoints/best.pt --lr 3e-4 --warmup_steps 50 --epochs 25`.
- **Result:** best val/l2_error = **1.1398** at epoch 14/14 (31 min). train=1.0219 by end. Still descending monotonically — lr decayed to 1.2e-4 by timeout; more budget would keep helping. VRAM 15.3 GB.
- **Verdict:** KEPT. **1.2218 → 1.1398 = 7% improvement.** Biggest single-iter jump since the decoder fix.
- **Notes:** Per-epoch trajectory: 1.2334 (warm E1) → 1.217 → 1.226 → 1.216 → 1.209 → 1.196 → 1.199 → 1.183 → 1.174 → 1.163 → 1.157 → 1.150 → 1.146 → 1.140. Train-val gap growing slowly (1.022 vs 1.140) but not obviously overfit. Next (iter 13): either (a) another warm-start round with even lower LR (1e-4) to squeeze more descent, or (b) add Sobolev/vorticity loss term (λ=0.1 kNN gradient-matching) — val metric is L2-per-point; adding spatial smoothness prior could help generalization especially now that we're past the easy wins. First epoch took 422s — GPU was shared at launch; real epoch time is 85s.

### 2026-04-16 — Warmup + higher LR + 25-epoch cosine (iter 10, kept)
- **Hypothesis:** Iter 9 still descending at epoch 22/50; T_max=50 means cosine barely decayed. Tune schedule to match actual budget: lr=1e-3 peak, 200-step linear warmup, cosine over 25 epochs so LR→0 by the end.
- **Change:** `train.py` — `lr=1e-3`, `warmup_steps=200`, `epochs=25`, `SequentialLR(LinearLR + CosineAnnealingLR)` stepping per-batch.
- **Result:** best val/l2_error = **1.2218** at epoch 22/22. train=1.1421 by end. Val was still monotonically descending — 3 more epochs would have helped.
- **Verdict:** KEPT (1.2594 → 1.2218, 3% improvement). Phase transition faster (epoch 3 vs iter 9's epoch 5). Still at #7.
- **Notes:** Still compute-bound. Real blocker is 85s/epoch (21 epochs in 30 min). Need per-epoch speedup to descend further. Next (iter 11): subsample to 30k training points (eval stays at 100k) — this should roughly triple epochs/minute, letting cosine fully anneal with far more descent budget.


- **Hypothesis:** All prior iterations plateaued at copy-last (1.75) because a single `Linear(hidden->15)` forced all 5 output time steps to share one predictor. The model cannot distinguish "easy" near-future (t+dt) from "hard" far-future (t+5dt) — gradient steps that help one step hurt another, so the net effect is the safe zero-delta solution. Replace with a shared MLP + per-step time embedding so each output step has its own effective decoder.
- **Change:** `train.py` BaselineMLP — remove `proj_out = Linear(hidden, 15)`. Add `time_embed = nn.Embedding(T_OUT=5, 32)` + `decoder = MLP(hidden+32 -> hidden -> 3)`. Forward expands features to `[B, T_OUT, N, hidden]`, concats time embedding, applies shared decoder, adds back `v_last`.
- **Result:** best val/l2_error = **1.2594** at epoch 22/22 (31 min). train=1.189, val=1.26 — still descending when timeout hit. VRAM 15.3 GB. 85s/epoch (slower than iter 8's 79s; the per-step expansion adds memory+compute).
- **Verdict:** KEPT. **1.7497 -> 1.2594 = 28% improvement**, moves fern from 8th to ~6th. The first iteration that actually beat copy-last.
- **Notes:** Training dynamics show a clear phase transition — epochs 1-4 stuck at 1.75 (copy-last), epoch 5 jumps to 1.64, epoch 6 to 1.40, then smooth descent. Confirms the copy-last attractor was the problem. The time-conditioned decoder lets the model allocate capacity per time horizon. Next: (a) run longer — loss still descending at epoch 22; (b) try richer per-step conditioning (FiLM on trunk?); (c) now that the output decoder is unlocked, revisit data augmentation (reflection) — it should finally help since the model is no longer in copy-last.

### 2026-04-16 — y-reflection aug + TTA (iter 8, TIED)
- **Hypothesis:** 146 geometries is too few; train 1.67 < val 1.75 suggests overfitting. y-flip aug doubles effective data; TTA averages original + mirrored predictions, enforces y-symmetry exactly.
- **Change:** `train.py` — `reflect_y()` flips pos_y, v_in_y, v_out_y, applied with p=0.5 during training. `predict_tta()` averages original + mirrored prediction. Used at val and predict time. Also `predict.py` updated to use TTA at submission.
- **Result:** best val/l2_error = **1.7498** at epoch 21/23. Train 1.6738, val 1.7498 — same floor.
- **Verdict:** tied. Reflection aug did NOT break the ceiling. **The problem is NOT data scarcity/overfitting alone.**
- **Notes:** Three back-to-back ties at ~1.75 across totally different architectures (per-point MLP, Perceiver, Transolver) + augmentation now strongly implicate the OUTPUT decoder. All 5 output time steps share one Linear(hidden→15). The model has no way to distinguish easy (t+dt) from hard (t+5dt) predictions — errors are averaged into a single predictor that effectively collapses to delta=0. Next: iter 9 = per-step time-conditioned decoder.

### 2026-04-16 — Transolver Physics-Attention (iter 7, TIED — and copy-last baseline discovered)
- **Hypothesis:** Perceiver's fixed learnable queries aren't geometry-aware. Transolver's data-dependent slice tokens (softmax(point→slice), M=32) cluster points by physical state (wake vs freestream vs boundary) per-sample. This is SOTA on AirfRANS/DrivAerML.
- **Change:** `train.py` — replaced PerceiverBlock with `PhysicsAttentionBlock(dim, n_slices=32, n_heads=8)` using orthogonal-init slice projection and learnable per-head temperature (0.5). Trunk = 8 × PhysicsAttention (each with own MHSA across slice tokens + FFN), no ResBlocks.
- **Result:** best val/l2_error = **1.7501** at epoch 22/24, train=1.6743 by end. 76s/epoch, VRAM 13.5 GB.
- **Verdict:** discarded (tied with iter 3/6). **CRITICAL DISCOVERY:** computed the copy-last baseline `v_out = v_in[-1]` on val — it scores **1.7496**. All my models (iter 3 1.7497, iter 6 1.7496, iter 7 1.7501) are converging to *exactly copy-last* on val. My "improvements" have been training-set overfitting.
- **Notes:** Train 1.67 vs val 1.75 confirms overfitting. Only 146 unique train geometries — not enough data for the model to learn *nonzero* deltas that generalize. Need DATA AUGMENTATION (reflection), stronger regularization, or richer input features (spatial gradients, vorticity). Next: iter 8 = reflection aug + TTA. Leader at 0.92 is ~47% improvement over copy-last — real generalizing model needed, not architecture tweaks.

### 2026-04-16 — Perceiver-style latent cross-attn + bf16 AMP (iter 6, TIED)
- **Hypothesis:** Per-point MLP is capacity-bound; fixed-query Perceiver (M=256 latents) at O(N*M) gives cheap global spatial context at full 100k points. AMP bf16 makes attention at this scale affordable.
- **Change:** `train.py` — `PerceiverBlock(dim, n_latents=256, n_heads=8)` with 3-stage attn (latent←point xattn, latent self-attn+FFN, point←latent xattn). 1 perceiver after 4 ResBlocks (via `perceiver_every=4`). bf16 AMP on train and val forward.
- **Result:** best val/l2_error = **1.7496** at epoch 44/44 in 30 min. bf16 cut per-step from 417ms→54ms fw+bw → **41s/epoch** (iter 3 was 56s/epoch). VRAM 6.6 GB.
- **Verdict:** tied with iter 3 (1.7497). bf16 AMP kept as a speedup, Perceiver did NOT meaningfully help. Train loss floor 1.673 ≈ val 1.75 → not overfit, the *type* of global context is wrong.
- **Notes:** This is the same ceiling iter 3, 4, 5 hit. Next: replace fixed-query Perceiver with **Transolver Physics-Attention** — data-dependent slice tokens (softmax(point→slice), M=32-64) give geometry-aware global context. Top teams on AirfRANS/DrivAerML all hard-code geometry awareness. Research findings saved to memory `gram_next_ideas.md`.


- **Hypothesis:** (a) Per-point MLP is capacity-bound — go wider/deeper. (b) NeRF-style Fourier pos features let the MLP learn higher-frequency spatial response. (c) Inter-step velocity diffs carry accel info. (d) MSE training is outlier-dominated; L2-per-point loss directly matches the val metric.
- **Change:** `train.py` — hidden=384, n_blocks=8; Fourier pos encoding (L=6, gives 39-d pos feat); v_norm + inter-step v_diff as inputs (27-d); loss = `(pred-v_out).norm(dim=3).mean()`.
- **Result:** best val/l2_error = **1.7497** at epoch 32/50 (30 min timeout). VRAM 8.1 GB. train=1.6731.
- **Verdict:** kept — improved 1.7778 → 1.7497. Run `fern/iter3-fourier-l2loss`. Train/val gap tiny → not overfit, adding more signal (spatial) is the next lever.
- **Notes:** Epoch 1 already hit 1.759 — L2 loss converges orders of magnitude faster than MSE; losses plateau by epoch ~10 but minor val improvements continue through epoch 32. Most wins probably come from L2 loss + Fourier features.

### 2026-04-16 — kNN EdgeConv + Fourier (iter 2, DISCARDED)
- **Hypothesis:** Per-point MLP ignores neighbors; interleave 2 DGCNN-style EdgeConv layers (kNN=16) between ResBlocks to add spatial context.
- **Change:** `train.py` — EdgeConvBlock with max-pooled edge MLP, chunked self-kNN with per-geometry cache, Fourier pos.
- **Result:** best val/l2_error = 1.7791 at epoch 3, then plateaued. 115 s/epoch (5× slower) → only 16 epochs in 30 min budget.
- **Verdict:** discarded. Roughly tied with iter 1 despite more capacity — the slowdown ate the epoch budget before the model could converge. Reverted via `git reset --hard HEAD~1`.
- **Notes:** VRAM 22.6 GB (fits but heavy). To retry spatial context cheaply: (i) voxel-grid pool (O(N), measured ~4 ms/layer with `index_add`), or (ii) precompute kNN per unique geometry once.

### 2026-04-16 — residual + no-slip + input-norm (iter 1)
- **Hypothesis:** Baseline predicts full velocity from scratch and ignores the fact that flow is nearly stationary over 5 steps. Predicting `delta = v_out - v_in[-1]`, enforcing zero velocity at `idcs_airfoil`, and normalizing inputs with dataset stats should each help.
- **Change:** `train.py` — BaselineMLP now stores vel_mean/vel_std buffers, normalizes v_in, adds v_in[-1] to the MLP output, and zeros predictions at airfoil indices. Same hidden=256, 6 blocks.
- **Result:** best val/l2_error = **1.7778** (epoch 19/50, ~19 min). VRAM peak 4.2 GB. `train/epoch_loss` ~7.8 by the end.
- **Verdict:** kept. First checkpoint committed as `checkpoints/best.pt`. Run `fern/residual-noslip-norm` (W&B id `bw4wpqff`).
- **Notes:** `predict.py` failed to auto-submit because `from train import BaselineMLP` triggered `sp.parse(Config)` in train.py's argparse. Fixed by wrapping train.py's script logic in `main()`/`__main__`. Resubmitted val predictions manually. Next ideas (biggest): (1) spatial context via kNN EdgeConv — the MLP is still per-point; (2) Fourier-feature pos encoding; (3) time-delta conditioning per output step.

