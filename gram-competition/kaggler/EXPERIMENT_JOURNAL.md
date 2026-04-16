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

