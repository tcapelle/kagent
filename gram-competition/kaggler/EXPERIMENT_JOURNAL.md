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

### 2026-04-17 — iter27: second fresh-init training + 4-model ensemble
- **Hypothesis:** iter26's fresh-init gave −0.016 ensemble gain. If the gain scales with *diversity of inits*, a second fresh-init model (iter27) should give another −0.005 to −0.010.
- **Change:** re-ran `python train.py --epochs 28 --agent nezuko` fresh (another seed by default).
- **Result:** iter27 standalone val/l2 = **0.9348** at e26/28 (similar to iter26's 0.9377 — fresh inits converge to similar quality). Ensemble tests:
  - iter19 + iter24 + iter26 + **iter27** (4-model): val l2 = **0.8553** (direct, with TTA) — **new best**
  - iter26 + iter27 only (pure fresh-init): 0.8743 — worse than 4-model, pure fresh-inits miss the chain-honed quality of iter19/iter24
- **Verdict:** **kept** — −0.007 vs iter26 ensemble (0.8622 → 0.8553). Trend: each new fresh-init model adds ~0.007-0.016 to the ensemble. Submission at `nezuko/94002e8`.
- **Notes:** The ensemble sweet spot is (chain-trained strong models) + (fresh-init diverse models). Pure fresh-inits aren't enough on their own; chain-trained alone saturates. Next iter: train iter28 with a deliberately different architecture (depth=4? grid=64? bigger slice_num?) so error correlation drops further.

### 2026-04-17 — iter26: fresh-init training for ensemble diversity + best 3-model ensemble
- **Hypothesis:** iter25 ensemble gain came almost entirely from iter19+iter24 (the chain endpoints). A *fresh-initialized* model will have genuinely decorrelated errors (different local minimum basin), so swapping iter23 for a fresh model should give a bigger ensemble gain. Train iter26 from scratch with same config as iter19 — diversity is in the init, not the architecture.
- **Change:** re-ran `python train.py --epochs 28 --agent nezuko` (no `--resume_from`). Different random seed by default.
- **Result:** iter26 standalone val/l2 = **0.9377** at e25/28 (worse than iter19's 0.9316 — slightly different init, slightly worse optimum, expected). BUT ensemble results transform:
  - iter19 + iter24 + **iter26** (3-model): val l2 = **0.8622** (direct, with TTA)
  - iter19 + iter23 + iter24: val l2 = 0.8784 (previous iter25 best)
  - iter19 + iter23 + iter24 + iter26 (4-model): val l2 = 0.8634 (iter23 adds correlated noise)
- **Verdict:** **kept** — new best ensemble. −0.016 vs iter25 (0.8784 → 0.8622). Gap to alphonse (0.7800) narrowed from 0.098 to 0.082. Submission at `nezuko/1f958fd`.
- **Notes:** Confirms that *diversity of init* beats *depth of chain*: a single fresh-init model contributes more ensemble signal than two chain-continuation models. For future iter: train iter27 with yet another fresh init, or with a deliberately different architectural knob (depth=4, grid=64, different loss) — each genuinely-decorrelated model should halve the remaining gap.

### 2026-04-17 — iter25: 3-model ensemble (iter19 + iter23 + iter24), each with y-flip TTA
- **Hypothesis:** chain warm-starts are saturating (iter23→iter24 gained only 0.0014 val). But iter19 and iter24 are 6-7 "equivalent epochs" apart on the chain and converge to slightly different local minima — averaging their predictions at inference should offset uncorrelated errors. Free gain, no training.
- **Change:** new `ensemble_predict.py` — loads N checkpoints, does y-flip TTA per model, averages. Also prints val l2_error directly.
- **Result:** 2-model (iter19+iter24) val l2=**0.8788**. 3-model (iter19+iter23+iter24) val l2=**0.8784**. Saved to `nezuko/d414130`. For reference: iter24 single-model LB 0.8827.
- **Verdict:** **kept** — new best LB (direct-measured 0.8784, expected to match on leaderboard). Huge gain: −0.04 vs iter24 alone (0.9179 → 0.878 with ensemble+TTA).
- **Notes:** Adding iter23 (which sits between iter19 and iter24 on the chain) gave only +0.0004 over 2-model — iter23's errors are mostly linearly in-between iter19's and iter24's, so contributes little independent signal. The bulk of the ensemble gain comes from the chain endpoints. Next: could train an *architecturally different* model (e.g., different seed, different depth) for genuinely decorrelated errors. Every new model halves the remaining ensemble gap roughly.

### 2026-04-17 — iter24: chain warm-start from iter23 ckpt (lr=5e-5, epochs=15)
- **Hypothesis:** iter23 chain worked once (-0.012 val) — chaining again with smaller LR should extract another fraction of an epoch from the same architecture. Lower LR = smaller perturbation = less "unlearning" at warmup, tighter anneal.
- **Change:** `python train.py --lr 5e-5 --epochs 15 --resume_from checkpoints/iter23_start.pt`.
- **Result:** val/l2 = **0.9179** at epoch 10 / 15 (best). 57 s/ep, 14.3 min wallclock. Run `7cw3e86t`. Preds saved to `nezuko/dd5cfed`. Trajectory: e1=0.922 → e7=0.918 → e10=0.918 → plateau 0.921 for e11-15.
- **Verdict:** **kept** — marginal −0.0014 vs iter23, but strictly better. Diminishing returns.
- **Notes:** 2nd chain gain (0.0014) ≪ 1st chain gain (0.012). Suggests we've nearly exhausted what fine-tuning-from-fixed-checkpoint can do for this architecture. For larger gains, need architectural change, not schedule change. Next iter ideas: (a) fine-tune with a different loss (pure Ux-weighted, since Ux MAE dominates), (b) ensemble iter19+iter23+iter24 (may offset overfitting of each), (c) architectural: add KNN local attention on top of voxel-sample features for sub-voxel detail.

### 2026-04-17 — iter23: warm-start fine-tune from iter19 ckpt (extend effective budget)
- **Hypothesis:** iter19 (val 0.9316) + every capacity-bump iter21/22 loses because a single 30-min run can't fit both capacity *and* the full cosine tail. Warm-starting from iter19's checkpoint with a short cosine (lr=1e-4, 20 ep, warmup=50) effectively extends total training to ~48 epochs, squeezing more annealing out of the existing architecture at no capacity cost.
- **Change:** `train.py` — add `--resume_from` CLI flag; load state_dict and use a shorter warmup (50 steps) when resuming.
- **Result:** val/l2 = **0.9193** at epoch 12 / 20 (best). 57 s/ep, 19.1 min wallclock, 7.7 GB peak. Run `wmn53ce5`. MAE Ux=0.591, Uy=0.311, Uz=0.435. Preds saved to `nezuko/d62e091`. Trajectory: e1=0.938 (warmup perturbation) → e6=0.931 (recovered iter19) → e10=0.925 → e12=0.919 → plateau at 0.920 for e13-20 as LR→0 (mild overfit, train 1.08→0.76).
- **Verdict:** **kept** — new personal best val, beats iter19 by 0.012 (-1.3%). LB pending.
- **Notes:** Warm-start chain works. Budget-saving trick confirmed: effective 28+20=48 epochs across runs. Best was at e12 (not e20) — schedule overshoots; e12-e15 is the plateau. Next iter: (a) warm-start from iter23 again (chain), (b) try lr=5e-5 for 15 ep (shorter, tighter anneal), or (c) y-flip TTA already baked in at predict-time, so LB should land well.

### 2026-04-17 — iter22: base_ch 128→160 (DISCARDED)
- **Hypothesis:** wider voxel features close the gap; bf16 headroom from iter18 should absorb the 25% cost bump.
- **Change:** `train.py` — `base_ch=160`, `epochs=22`.
- **Result:** 82s/ep (too slow) — cosine only reached e20 before 28-min timeout; best val **0.9489**. LB with TTA: **0.9171**.
- **Verdict:** **discarded** — worse than iter19 (val 0.9316, LB 0.8969) by +0.02 val / +0.02 LB. Same pattern as iter16/17: capacity bump eats the cosine tail.
- **Notes:** Any change that raises s/epoch above ~60 loses. bf16 headroom is not enough for +25% width when we're already running depth=3.

### 2026-04-17 — iter21: Transolver slice_num 32→64 (retry iter17 with bf16+depth3 context) (DISCARDED)
- **Hypothesis:** iter17's slice=64 failed because epoch budget ran out (fp32 + depth=2). With bf16 + depth=3 headroom, slice=64 should finally add useful capacity.
- **Change:** `train.py` — `transolver_slice_num=64`, `epochs=25`.
- **Result:** val **0.9319** (essentially tied with iter19's 0.9316). LB with TTA: **0.9102** (vs iter19's 0.8969, +0.013).
- **Verdict:** **discarded** — LB regressed. TTA benefit halved; slice=64 model is less y-flip-robust.
- **Notes:** Slice count beyond 32 over-parameterizes the physics-clustering; soft-assignments get noisier and break the y-symmetry TTA relies on. Stick with slice=32.

### 2026-04-17 — iter19: Transolver depth=3 + bf16 (retry iter16 with budget headroom)
- **Hypothesis:** iter16 (depth=3) failed purely because of budget — 96s/ep × 20 epochs ate the cosine tail. With iter18's bf16 speedup, depth=3 should run at ~60s/ep, fitting 28 epochs in ~28 min. The +1 Transolver block adds a round of global mixing; iter16's curve was 1 epoch ahead of iter15's early, so the architecture is right — it just needs the full cosine.
- **Change:** `train.py` — `transolver_depth=3` (on top of iter18 bf16 + epochs=28).
- **Result:** val/l2 = **0.9316** at epoch 27 / 28 (best). 57s/ep, 26.8 min wallclock, 7.7 GB peak. Preds saved to `nezuko/8b864d8`. LB with TTA: **0.8969** — jumped to **#3** on leaderboard (passed tanjiro 0.9035). Smooth monotone descent through e27 (0.948, 0.938, 0.935, 0.934, 0.933, 0.933, 0.932).
- **Verdict:** **kept** — new personal best on both val and LB. Beats iter18 (val 0.939, LB 0.906) by 0.007/0.009. Capacity *did* help once budget allowed it.
- **Notes:** Confirms iter16's hypothesis was right, budget was wrong. bf16 budget headroom is the lever that unlocks capacity. Next iter: keep pushing — depth=4? Or bigger base_ch? Gap to alphonse (#2 at 0.7993) still 0.10; thorfinn (#1 at 0.7738) 0.12. MAE Uy still the weakest at 0.2993.

### 2026-04-17 — iter18: bf16 autocast + epochs=28 (use Blackwell mixed precision to run more epochs)
- **Hypothesis:** iter16 (depth 3) and iter17 (slice 64) both failed — not because capacity didn't help, but because adding params eats epoch budget and iter15's 0.9578 is a tight optimum on the `epochs × capacity` frontier. Free more compute via bf16 autocast (Blackwell RTX PRO 6000 has native bf16), target ~1.5x speedup → 60s/ep → 28-30 epoch budget. Same iter15 model.
- **Change:** `train.py` — wrap train/val forward in `torch.amp.autocast('cuda', dtype=torch.bfloat16)`, cast val `pred` back to fp32 for metrics; bump `epochs: 22→28`.
- **Result:** val/l2 = **0.9390** at epoch 25 / 28 (best). 51s/ep (vs iter15 83s — **37% faster**). 24 min wallclock, 6.7 GB peak (vs iter15 8.2 GB). Preds saved to `nezuko/449aa3e`. Late-epoch trajectory: e21 0.953, e22 0.948, e24 0.941, e25 0.939.
- **Verdict:** **kept** — new personal best. Beats iter15 val by 0.019 (2% relative). LB with TTA pending.
- **Notes:** bf16 is a massive win — zero accuracy regression (epoch-by-epoch trajectory overlaps iter15's), but 1.6x speedup opens 6-8 more training epochs within budget. VRAM drop 8.2→6.7 GB gives headroom for larger models too. Next iter: invest the saved budget in capacity (depth=3 or bigger base_ch) since epoch-count is no longer the bottleneck.

### 2026-04-17 — iter17: Transolver slice_num 32→64 (DISCARDED)
- **Hypothesis:** more physics-slice tokens = more expressive soft-clustering without budget cost (N*M*C scatter doubles but dominated by MLP/qkv cost, ~5% epoch-time impact).
- **Change:** `train.py` — `transolver_slice_num=64`.
- **Result:** val/l2=**0.9662** at epoch 20, 88s/ep × 21 = 31 min (timed out before e22). LB with TTA: **0.9315** (worse than iter15's 0.9266). Run commit `b79d0ff`.
- **Verdict:** **discarded** — trained slightly slower (88 vs 83 s/ep), got one fewer epoch, and more slices didn't help optimization. 32 slices are apparently enough; doubling just added optimization noise.
- **Notes:** Pattern emerging: iter15 sits on a tight frontier — *any* change that reduces total epoch count (or subtly complicates optimization) loses. Need to either (a) extend the budget via compute efficiency (bf16, torch.compile), or (b) find an orthogonal improvement that doesn't touch the cosine schedule.

### 2026-04-17 — iter16: Transolver depth 2→3 (DISCARDED)
- **Hypothesis:** more global mixing via one extra Transolver block should close the 0.10 gap to alphonse.
- **Change:** `train.py` — `transolver_depth=3`, `epochs=20`.
- **Result:** val/l2=**0.9668** at epoch 19 (timeout). 96s/ep × 19 = 30.4 min; 9.5 GB. LB with TTA: **0.9375** (worse than iter15's 0.9266). Run commit `c5fdd77`.
- **Verdict:** **discarded** — depth=3 converged one epoch earlier than iter15 (e8 val 1.10 vs iter15 e9 1.10) but the shorter cosine schedule (20 vs 22) meant the annealed tail — the most productive epochs — was shorter. Capacity ≠ free.
- **Notes:** Key lesson: adding layers without extending total compute loses the cosine tail. For capacity bumps, prefer knobs that *don't* reduce epoch count (slice_num, head_dim). Also VRAM jumped 8.2→9.5 GB (headroom for 4 blocks if wanted — but not the right axis).

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

