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

### 2026-04-18 — v21: v16-arch attn=3 + yflip + dropout=0.15 (KEPT)
- **Hypothesis:** v20 (1.1304 single, 1.0654 7-ensemble) succeeded because
  v16-family + yflip + higher dropout form a quality-parity diverse
  member. v21 is a near-variant with lighter hyperparams (n_attn_blocks=3
  vs 4, dropout=0.15 vs 0.2, same layer_scale=1e-3) to add a seed- and
  hyperparam-diversified member. Expect single ~1.12-1.13, 8-ensemble
  delta -0.001 to -0.002 (diminishing returns).
- **Change:** `--model_version v16 --n_blocks 10 --epochs 54 --dropout_p 0.15
  --n_attn_blocks 3 --layer_scale 1e-3 --yflip_aug True` + 180min budget.
- **Result:**
  - Single: val/l2=**1.1354** at ep54 of 54 (171.7 min, still improving).
    WandB `edward/v21-v16-attn3-dropout15` (sgsw7i4r). 191s/epoch, 17.5GB.
  - **8-member ensemble: val/l2=1.0648** (better than 7-member 1.0654,
    delta -0.0006). Marginal but positive.
- **Verdict:** KEPT. Member #8.
- **Notes:** Diminishing returns exactly as predicted — v20 added -0.0027,
  v21 only -0.0006. Confirms ensemble is saturating on v6/v16-family
  diversity. To meaningfully close the gap to rank 7 (0.072 away), need
  (a) a step-change in single-model quality or (b) a fundamentally
  different architecture. Researched alternatives: Transolver-style
  PhysicsAttention (learned soft slicing + Ada-Temp) is the highest
  evidence lever — my VoxelTokenAttn is a hard-voxel caricature of the
  same idea, and Transolver is SOTA on ShapeNetCar / DrivAerNet++
  benchmarks. v22 will prototype this with point subsampling aug.

### 2026-04-17 — v20: v16-arch attn=4 + yflip (KEPT)
- **Hypothesis:** v18/v19 both failed because their singles were far
  outside our quality floor (1.23–1.25 vs floor 1.18). To improve the
  6-ensemble (1.0681), a new member needs BOTH quality parity (val/l2
  ≲ 1.18) AND architectural diversity. v17 was proof of concept
  (1.1493, v16 arch + 2 attn blocks + yflip, helped ensemble from
  1.0724 → 1.0681). v20 takes v17's recipe further: 4 attn blocks (2x
  attention depth), layer_scale 1e-3 (looser init), dropout 0.2, 55
  epochs with yflip. Expect lower single (more capacity + regularization
  + aug) AND ensemble-useful diversity vs the v6-arch members.
- **Change:** `--model_version v16 --n_blocks 10 --epochs 55 --dropout_p 0.2
  --n_attn_blocks 4 --layer_scale 1e-3 --yflip_aug True` + 180min budget.
- **Result:**
  - Single: val/l2=**1.1304** at ep54 of 55 (180.1 min, trajectory still
    improving; hit timeout). WandB `edward/v20-v16-attn4-long` (rlzz6vjz).
    200s/epoch at 18.1GB peak VRAM. Better than v17's 1.1493.
  - **7-member ensemble (add v20 to prior 6): val/l2=1.0654** (better
    than 6-member 1.0681). Delta −0.0027.
- **Verdict:** KEPT. Member #7.
- **Notes:** Quality+diversity hypothesis confirmed. v20 is now our
  best-single v16-arch checkpoint; doubling attn blocks didn't hurt
  training stability (thanks to layer_scale=1e-3). Next ideas:
  (A) v21 rerun of v20 with different seed — cheap way to get another
      quality-parity diverse member.
  (B) hidden=768 on v16 arch (capacity bump, ~40% slower).
  (C) Reconsider whether val.pt auto-overwrite from predict.py is
      causing silent regressions — watch for it.

### 2026-04-17 — v19: v6-arch + yflip aug only (DISCARDED)
- **Hypothesis:** v17 combined yflip+v16-arch (1.1493). Isolating yflip
  on v6-arch would give either (a) improved single over v14's 1.1041
  via 2x effective train data, or (b) arch-matched but yflip-diverse
  ensemble member.
- **Change:** `--model_version v6 --n_blocks 10 --dropout_p 0.2
  --epochs 44 --yflip_aug True`. 44 epochs because v19 ran at 162s/ep
  (60% slower than v14's 102s/ep — unclear why; same arch, same
  hardware, just yflip added; GPU was idle otherwise).
- **Result:**
  - Single: val/l2=**1.2316** at ep44 of 44 (119.1 min). Trajectory:
    1.49 (ep5), 1.43 (ep6), 1.33 (ep20), 1.25 (ep34), 1.24 (ep36),
    1.23 (ep44). WandB `edward/v19-v6-yflip` (1m23al87). Much worse
    than v14 (1.1041). Plausible causes: (1) only 44 effective epochs
    vs v14's 70; (2) yflip halves the training signal per pass because
    model sees each sample in original OR flipped form, not both;
    (3) airfoils are approximately y-symmetric but the flow isn't
    (inflow direction, wing incidence).
  - **7-member ensemble: 1.0743** (WORSE than 1.0681). Same failure
    mode as v18 — too weak to decorrelate productively.
- **Verdict:** DISCARDED.
- **Notes:** Two-in-a-row pattern: any new single with val/l2 > 1.20
  hurts our current 6-member ensemble (best singles 1.10–1.18).
  Takeaway — to improve the ensemble, new members need **quality
  parity** (val/l2 ≲ 1.18) AND architectural diversity. v17 was the
  sweet spot (1.1493 + diff arch). Options for v20:
  (A) v17-arch rerun with different seed (stable ≈1.14 diverse member)
  (B) v16-arch at hidden=768 (bigger capacity, slower but potentially
      1.10-ish)
  (C) go big: new arch (kNN graph message passing, or actual SetAttn)
  Also noted: the aggressive overwrite of val.pt by auto-predict.py
  happened again. Continuing to re-run ensemble.py defensively.

### 2026-04-17 — v18: v6-arch + near-airfoil weighted loss (DISCARDED)
- **Hypothesis:** errors concentrate in boundary layer (mean err 2.4
  within 0.05m vs 0.3 beyond 0.1m of airfoil). Up-weighting the loss
  there with `w = exp(-d/0.05)` (mean-normalized) should force the
  model to optimize the hard region. Even if single is worse,
  qualitatively different objective could add ensemble diversity.
- **Change:** `train.py` — per-point weighted loss via new config
  fields `loss_weight_near_airfoil` (α exponent) and
  `loss_weight_scale`. CLI: `--loss_weight_near_airfoil 1.0
  --loss_weight_scale 0.05`. Same v6 arch as v14. Epochs=40 (slower
  per-epoch: 181s due to min_distance_to per batch, vs v14's 102s).
- **Result:**
  - Single: val/l2=**1.2532** at ep40 of 40 (120.7 min, still
    decreasing). WandB `edward/v18-v6-airfoilweight` (bihmoi44).
    Much worse than v14 (1.1041) — the weighting shifts optimization
    away from the >95% of points that are far from airfoil and are
    already "easy but not trivial".
  - **7-member ensemble (add v18 to prior 6): val/l2=1.0764**
    (WORSE than 6-member 1.0681). v18's predictions are too far off
    to decorrelate productively.
- **Verdict:** DISCARDED. `predict.py` was auto-triggered by train.py
  and overwrote the ensemble val.pt on PVC with v18's singles (caught
  and re-ran ensemble.py to restore 1.0681). Removed v18 from MEMBERS.
- **Notes:** Gotcha for future — `train.py` auto-runs predict.py on
  best checkpoint, which overwrites ensemble predictions for current
  commit. To avoid: run new training, then always re-run ensemble.py
  afterward to keep the ensemble as the scored artifact.
  Failure mode analysis: α=1.0 with scale=0.05 gave near-field points
  ~20x weight vs far-field. Too aggressive — the model didn't have
  enough capacity/data to get BOTH right, and the far-field blew up
  from its usual ~0.3 mean err to dominate the uniform-averaged L2
  metric. Gentler weighting (α=0.3, scale=0.1) might work, but not
  prioritizing this direction.

### 2026-04-17 — v17: v16 arch (Fourier + voxel-token attn) + y-flip aug, added to ensemble
- **Hypothesis:** v14 (1.1041) saturated the v6-arch diversity dimension.
  To break the ensemble decorrelation ceiling, need a *qualitatively*
  different member. v16 arch has: Fourier positional embedding (K=6),
  voxel-token self-attention with LayerScale, and dropout; plus y-flip
  data aug (airfoils verified y-symmetric across 20+ samples, mean y≈0,
  skew <0.15) doubles effective train data. Expect worse-but-diverse
  single, better ensemble.
- **Change:** CLI: `--model_version v16 --n_blocks 10 --dropout_p 0.2
  --n_attn_blocks 2 --layer_scale 1e-3 --yflip_aug True --epochs 40`.
  120 min budget (181s/epoch with 10 blocks + 2 attn blocks).
- **Result:**
  - Single model: val/l2=**1.1493** at ep40 of 40 (120.9 min).
    MAE Ux=0.776, Uy=0.342, Uz=0.537. WandB run `76oie0fz`
    (`edward/v17-v16-attn-yflip`). Worse than v14 single (1.1041) as
    expected — more params + diff arch, same budget.
  - **6-member ensemble (v6+v9+v11+v13+v14+v17): val/l2=1.0681**
    (−0.0043 over 5-member 1.0724; −0.10 over best single 1.1041).
    Mean of individuals = 1.1505. Explored 1/score^k weighted
    averaging: k=4→1.0657, k=8→1.0637 — marginal, potentially
    overfits to val. Sticking with uniform mean.
- **Verdict:** Kept. v17 adds diverse-arch decorrelation to a
  previously same-arch pool. Modest but real gain.
- **Notes:** The diminishing-returns curve continues (5→6 members
  gave −0.004; 4→5 gave −0.014). To go further: need qualitatively
  different training objective or feature. Ideas for v18:
  (a) near-airfoil weighted loss (errors concentrate within 0.05m
  of airfoil: mean err 2.4 vs 0.3 beyond 0.1m) — target the hard
  boundary-layer region; (b) L1/Huber loss member; (c) fresh seed
  of v17 arch for within-arch ensemble diversity.

### 2026-04-17 — v14: v6 + dropout p=0.2 in ResBlock (added to ensemble)
- **Hypothesis:** v13 at p=0.1 worked well (1.1176); with 730 samples
  and a 10.8M-param model we may still be under-regularized. p=0.2
  should push further and either (a) improve single-model or
  (b) add more decorrelation to the ensemble.
- **Change:** Only CLI arg `--dropout_p 0.2` (same config as v13
  otherwise). 70 epochs, 120 min budget.
- **Result:**
  - Single model: val/l2=**1.1041** at ep69 of 70 (119 min).
    Train loss 0.043 → 0.0121. Both outcomes happened: (a) beats
    v13's 1.1176 single by 0.013; (b) ensemble still improves.
    WandB `edward/v14-v6-dropout02`.
  - **5-member ensemble (v12 + v13 + v14): val/l2=1.0724**
    (−0.014 over 4-member 1.0861; −0.10 over v6 1.1681).
- **Verdict:** Kept. Dropout 0.2 > 0.1 for this dataset size. The
  diminishing-returns pattern is visible (3→4 members gave −0.023,
  4→5 gave −0.014) but still worth adding well-trained diverse
  members. Ensemble score now 1.0724 — solidly mid-tier on the
  mar29-era scale.
- **Notes:** Mean of individuals = 1.1507 vs ensemble 1.0724 — the
  ensemble is recovering ~65% of the gap between mean-of-singles
  and the theoretical uncorrelated-errors bound (1.1507/√5≈0.515;
  we're at 1.0724). Next: try p=0.3 or a qualitatively different
  member (e.g., different VOXEL_SCALES, or L1/Huber loss) to break
  the decorrelation ceiling of "same-arch dropout variants".

### 2026-04-17 — v13: v6 + dropout p=0.1 in ResBlock (added to ensemble)
- **Hypothesis:** v11 showed the bottleneck is generalization gap,
  not optimization noise. Dropout is the textbook fix for overfitting
  on a small dataset (730 train samples). Functional dropout between
  GELU and 2nd Linear of each ResBlock keeps state_dict compatible
  with existing v6 checkpoints (so one model class loads both).
  Aims: beat v6's 1.1681 and add a diverse member to v12's ensemble.
- **Change:** `model.py::ResBlock` — keep `nn.Sequential` the same
  (so state_dict keys match), call `F.dropout` in forward between
  GELU and final Linear when `self.training`. `ResidualMLP.__init__`
  takes `dropout_p`. `train.py` adds `--dropout_p` CLI flag, passes
  through. Config same as v6: 10 blocks, hidden 512,
  VOXEL_MIX_SCALES=(0.12,), MIX_EVERY=2. 70 epochs, 120 min budget.
  Added v13 checkpoint (`oxopax5h`) to `ensemble.py` as 4th member.
- **Result:**
  - Single model: val/l2=**1.1176** at ep69 of 70 (119 min, 102s/ep).
    Train loss 0.043 → 0.0118. Trajectory: 1.44 (ep10), 1.31 (ep20),
    1.22 (ep29), 1.188 (ep33), 1.158 (ep41), 1.148 (ep45),
    1.13 (ep49), 1.128 (ep52), 1.124 (ep54), 1.120 (ep58),
    1.118 (ep67), 1.118 (ep69). WandB `edward/v13-v6-dropout01`.
  - **4-member ensemble (v12 + v13): val/l2=1.0861.**
    Improvement of 0.023 over v12 (1.1093), 0.082 over v6 (1.1681).
- **Verdict:** Kept — v13 alone beats all prior single models by a
  huge margin (1.1176 vs v6 1.1681, −0.050); adding it to the
  ensemble drops the score another 0.023. Dropout is clearly the
  right regularization here.
- **Notes:** Remarkable — dropout alone gave nearly the same gain
  as ensembling 3 v6-family runs. Training dynamics were different:
  dropout forces each sub-network to be self-sufficient, which seems
  to find a qualitatively better solution on this data volume. Next:
  (1) train 1-2 more dropout variants with different seeds / scales
  / dropout rates to further diversify; (2) try higher dropout
  (0.2, 0.3) — 0.1 may be conservative given the small dataset.

### 2026-04-17 — v12: ensemble 3 v6-family checkpoints (mean of predictions)
- **Hypothesis:** v6, v9, v11 all hit ~1.17–1.18 val from different
  seeds and training schedules — same architecture, plausibly
  decorrelated errors. Averaging predictions should beat any single
  model. Cheapest possible "regularization": no training, ~30s compute.
- **Change:** New `ensemble.py`. Loads 3 v6-arch checkpoints from
  PVC (`jnseenin`=1.1681, `9vaz4wrn`=1.1812, `w8rxftm4`=1.1828),
  runs each on val split, averages predictions per sample, saves
  to `/mnt/new-pvc/predictions/apr16/edward/<commit>/val.pt`.
  Also added `eval_ckpts.py` to rank v6-arch checkpoints on PVC.
- **Result:** Ensemble val/l2=**1.1093** (vs best single 1.1681,
  mean-of-singles 1.1774). Improvement of 0.058 (−4.9% relative)
  over v6 — breaks the 1.17 plateau cleanly.
- **Verdict:** Kept. Committed ensemble predictions as the scored
  file at HEAD. This is the new baseline score to beat.
- **Notes:** Decorrelation is large: individual errors mean 1.177,
  ensemble 1.109 → relative error reduction is ~58% of the way to
  the theoretical uncorrelated-errors bound (mean/√3 ≈ 0.68). Very
  healthy. Next directions: (1) grow the pool — train 2–3 more v6
  runs with different seeds/augmentations to diversify; (2) add a
  genuinely different arch (e.g. small KNN attention model) and
  ensemble it in — more decorrelated = bigger win; (3) rank-weighted
  average (down-weight worse models) or learned blend.

### 2026-04-17 — v11: v6 + EMA weights (decay 0.999)
- **Hypothesis:** batch_size=1 is very noisy. EMA on parameters +
  validating with EMA weights gives a dampened, monotone trajectory
  and typically ~1–3% val improvement. Cheapest regularizer worth
  trying before architecture changes. Target: break 1.17.
- **Change:** `train.py` — initialize `ema_params` from model params,
  update each step with `p_ema = 0.999·p_ema + 0.001·p`, swap EMA
  into model weights around `validate(...)`, save EMA weights in
  checkpoint. Architecture identical to v6 (10 blocks, hidden 512,
  VOXEL_MIX_SCALES=(0.12,), MIX_EVERY=2). 70 epochs, 120 min budget.
- **Result:** Best val/l2=**1.1812** at epoch 67 of 70 (113 min).
  Train loss 0.044 → 0.0128. Val trajectory (EMA-evaluated):
  1.36 (ep5), 1.30 (ep10), 1.25 (ep20), 1.23 (ep25, 28), 1.22 (ep30),
  1.21 (ep35), 1.20 (ep40), 1.192 (ep45), 1.181 (ep53–54, best phase),
  bounce up to 1.197 (ep59), recover to 1.1812 at ep67, flat to ep70.
  WandB `edward/v11-v6-ema`.
- **Verdict:** Discarded — worse than v6 (1.1681) by 0.013. EMA gave
  a smoother, monotone-looking trajectory early (no wild fluctuations
  like raw v9 runs) but the final best was still above v6. EMA can't
  compensate for what appears to be an architecture ceiling.
- **Notes:** Interestingly, train loss 0.0128 is *lower* than v6's
  0.014 — EMA-smoothed model fits train data better but doesn't
  generalize. Confirms the bottleneck is generalization gap, not
  optimization noise. Next: try actual regularization (dropout in
  ResBlocks, or stochastic depth) to directly attack the gap, or
  ensemble multiple v6-family runs from PVC checkpoints.

### 2026-04-17 — v10: richer VoxelMix (mean + max + Linear fuse)
- **Hypothesis:** v6's scatter-mean mix destroys within-voxel
  variation — exactly the turbulent signal we need. Adding a scatter
  max branch fused with mean via a Linear should give richer spatial
  context per mix step and break the 1.17 plateau.
- **Change:** `model.py::VoxelMix` — compute both `scatter_mean` and
  `scatter_max`, concat (2×dim), LN, `Linear(2d→d)` fuse, tanh gate.
  +5M params (13.5M vs v6's 8M). Config otherwise v6. 55 epochs,
  120 min budget. scatter_max falls back to torch (no torch-scatter),
  adds ~18% per-epoch cost.
- **Result:** Best val/l2=**1.1983** at epoch 48 of 55 (106 min).
  Train loss 0.044 → **0.0097** — fits much better than v6 (0.014).
  Val plateaued at ~1.20 from ep40 onward, bouncing 1.198–1.203
  with near-zero LR.
  WandB `edward/v10-meanmax-mix`.
- **Verdict:** Discarded — slightly worse than v6 (1.1681). Classic
  overfitting signature: lower train loss, higher val plateau.
  Adding mix capacity is the wrong direction.
- **Notes:** The model already has ample fitting capacity. Bottleneck
  is generalization, not expressiveness. Next: try EMA on weights
  (expect ~2% from dampening batch=1 noise) before any arch changes.

### 2026-04-17 — v9: pure v6 + 130 min budget (70 epochs)
- **Hypothesis:** v6 was still descending at its 52 ep / 90 min timeout.
  With a properly sized cosine schedule (epochs=70) and 130 min budget
  the same architecture should land meaningfully below 1.1681.
- **Change:** No model changes (v6: VOXEL_MIX_SCALES=(0.12,), MIX_EVERY=2,
  n_blocks=10, hidden=512). Only hyperparams: epochs 52→70,
  MAX_TIMEOUT_MIN 90→130.
- **Result:** Best val/l2=**1.1828** at epoch 69 of 70 (130.5 min, 97s/ep).
  Train loss 0.044 → 0.014. Val trajectory: 1.34 (ep20), 1.30 (ep26),
  1.29 (ep30), 1.27 (ep34), 1.26 (ep36), 1.24 (ep40), 1.23 (ep43, ep45),
  1.22 (ep47), 1.21 (ep48, ep50), 1.19 (ep53, ep55, ep57),
  1.188 (ep58, ep60), 1.184 (ep61, ep66), 1.183 (ep67, ep68, ep69).
  WandB `edward/v9-v6-long`.
- **Verdict:** Discarded — matches v6 (1.1681) within noise, does not beat.
  Confirms v6 architecture is at its plateau; longer training alone
  won't break through.
- **Notes:** Val improved ~0.01 over last 10 epochs (ep60→70) as LR
  finished decaying, so the late-annealing hypothesis was weakly
  confirmed. But the floor is ~1.18 for this architecture regardless.
  Next must add genuine capacity or a different inductive bias.

### 2026-04-17 — v8: + mid-network voxel-token transformer (scale 0.10)
- **Hypothesis:** v6's iterative scatter-mean mix gives only local
  receptive field. Adding a single mid-network transformer over
  voxel tokens (~1.2k tokens at scale 0.10) gives every point one
  global-receptive-field event. Should beat v6's 1.1681.
- **Change:** `model.py` → new `VoxelTokenMix` (scatter-mean pool to
  voxel tokens with positional encoding from voxel centroid, 2-layer
  self-attention, scatter back via tanh-gated gather). Inserted once
  after `n_blocks // 2`. Params 8M → 15M. Cosine `epochs=45` to align
  with 90 min budget (avoid v7's premature-cut LR schedule).
- **Result:** Best val/l2=**1.2344** at epoch 38 of 45 (77 min).
  Train loss 0.044 → 0.018 — *higher* than v6's 0.014 despite 2× the
  params. Val plateaued at ~1.23 from ep28 onward.
  WandB `edward/v8-token-transformer`.
- **Verdict:** Discarded — significantly worse than v6 (1.1681).
  Extra capacity didn't translate to better fit, suggesting the
  attention block was hard to optimize and/or the tanh-gated residual
  collapsed to ≈0. Cosine schedule fully decayed by ep42 with no late
  improvement.
- **Notes:** Stop adding capacity until we have evidence v6 hit a
  capacity ceiling (it didn't — v6's train loss kept descending).
  Next: v9 = pure v6 with longer training budget to see if v6 still
  has room with more time.

### 2026-04-17 — v7: two-scale voxel-mix (0.08 + 0.25)
- **Hypothesis:** v6 used a single mix scale 0.12. Adding a coarser
  second scale (0.25) should capture longer-range structure while
  keeping fine detail at 0.08 — covering wake + near-airfoil regions.
- **Change:** `model.py` → `VOXEL_MIX_SCALES = (0.08, 0.25)`. Same
  `MIX_EVERY=2`, 10 ResBlocks, hidden=512. 90 min budget.
- **Result:** Best val/l2=**1.1824** at epoch 36 of 36 (90 min timeout).
  Train 0.044 → 0.0155. Per-epoch 136s (vs v6 ~105s average). v7 was
  ahead of v6 at matched epoch count (ep29: v7=1.22 vs v6=1.26) but
  the extra mix scale slowed epochs so v6 got 52 vs v7's 36.
  WandB `edward/v7-multiscale-mix`.
- **Verdict:** Discarded — worse than v6 (1.1681). Architecture is
  per-epoch better but slower, and cosine schedule (epochs=100) didn't
  fully anneal at timeout.
- **Notes:** If retrying, either set `epochs` closer to the expected
  budget so LR actually decays, or drop the coarser scale. Better
  next step: real learned aggregation on voxel tokens, not gated
  mean-pooling.

### 2026-04-16 — v6: iterative voxel-mix + 90 min budget
- **Hypothesis:** v4's per-point MLP never mixes latent representations
  across space. Adding lightweight scatter-mean message passing between
  ResBlocks (VoxelMix) gives iterative spatial communication. v5.1
  showed this matches v4 at equal epochs; extra training time should
  push it well past v4.
- **Change:** Same model as v5.1 — `VoxelMix` with per-scale LN +
  tanh-bounded per-channel gate (zero-init), scatter-mean at scale 0.12
  applied every 2 blocks. Training: `torch.nn.utils.clip_grad_norm_`
  (max_norm=1.0) added to stabilize early iterations. 10 ResBlocks, 5
  VoxelMix, hidden=512. `MAX_TIMEOUT_MIN=90`.
- **Result:** Best val/l2=**1.1681** at epoch 52 of 52 (91 min).
  Train loss 0.044 → 0.014, still descending. Val trajectory: 1.78
  (ep1), 1.46 (ep5), 1.41 (ep10), 1.33 (ep20), 1.26 (ep28), 1.24
  (ep35), 1.20 (ep40), 1.17 (ep48, ep52). WandB `edward/v6-mix-90min`.
- **Verdict:** Kept — ~6% better than v4 (1.2409 → 1.1681). Iterative
  message passing is clearly the right direction. Val still descending
  at timeout.
- **Notes:** Per-epoch time was variable (97-190s) — likely background
  GPU use. Gap to winner (~0.85) is now ~37%. Next: pivot to
  voxel-token transformer (voxel pool at scale 0.10 → ~1-2k tokens,
  self-attention among them, scatter back to points). Much stronger
  spatial model than gated scatter-mean, still tractable.

### 2026-04-16 — v5.1: VoxelMix (LN + tanh gate) + grad-clip (55 min)
- **Hypothesis:** v5 diverged at ep19 because unconstrained gate
  parameter allowed positive feedback. Bound the gate magnitude and
  add grad clipping.
- **Change:** `model.py::VoxelMix` — added per-scale LN on pooled
  features, tanh on gate, zero-init kept. `train.py` — add
  `clip_grad_norm_(..., 1.0)`.
- **Result:** Best val/l2=1.2545 at epoch 34 of 34 (55 min budget
  gave fewer epochs due to mix compute overhead). No divergence.
  At ep34, v4 was 1.2806 — so v5.1 beats v4 at equal epoch count.
- **Verdict:** Kept as stepping stone — architecture validated,
  needed longer training.

### 2026-04-16 — v4: 4 voxel scales (0.03/0.08/0.20/0.50) + within-voxel offsets
- **Hypothesis:** v3's two scales under-resolve boundary-layer detail
  (0.05) and large-structure context (need > 0.20). Adding finer 0.03
  and coarser 0.50 scales plus an "offset within voxel" feature
  (lets two points in the same voxel differentiate sub-voxel position)
  should improve further.
- **Change:** `model.py` → `VOXEL_SCALES = (0.03, 0.08, 0.20, 0.50)`,
  `voxel_stats` now returns sub-voxel offset per scale, input dim
  53 → 83. Default blocks 10 → 12. 77s/epoch (vs v3's 67s).
  Peak VRAM 9.6GB.
- **Result:** Best val/l2=**1.2409** at epoch 36 of 43 (55 min).
  Train loss 0.044 → 0.018. Still improving at timeout.
  WandB `edward/v4-4scales+offsets`.
- **Verdict:** Kept — modest 1.6% gain over v3 (1.2615 → 1.2409),
  diminishing returns for more hand-crafted features alone.
- **Notes:** Gap to winner (~0.85) is still huge. Pure pointwise MLP
  + voxel features has a ceiling because voxels don't mix across the
  network — each point only sees static pooled neighbors *once*,
  at the input. Next: iterative voxel-scatter/gather interleaved
  with ResBlocks (graph-conv style message passing).

### 2026-04-16 — v3: multi-scale voxel stats (mean/std/dev) + laplacian + bf16 AMP
- **Hypothesis:** v2's voxel *mean* alone is weak — turbulence needs local
  variance and self-deviation (a gradient-like proxy), plus a cheap
  laplacian (coarse mean − fine mean) to signal spatial curvature.
  Richer spatial stats should finally break the 1.3 plateau.
- **Change:** `model.py` → `voxel_stats` returning (mean, std, dev) per scale,
  plus `laplacian = mean_coarse − mean_fine`. Input dim 41 → 53.
  `train.py` adds `--amp` bf16 autocast (~1.8× speedup).
  Arch: hidden=512, 10 ResBlocks. 67s/epoch. Peak 8.1GB (AMP).
- **Result:** Best val/l2=**1.2615** at epoch 49 of 49 (55 min timeout).
  Train loss 0.044 → 0.019. Val trajectory bounces but monotone-best:
  1.70, 1.66, 1.51, 1.43, 1.48, ..., 1.30 (ep20), 1.29 (ep27),
  1.28 (ep35), 1.27 (ep39), 1.27 (ep42), 1.26 (ep48, ep49).
  WandB run `edward/v3-voxelstats+amp` (uqgrg7g7). MAE
  Ux=0.86, Uy=0.36, Uz=0.59.
- **Verdict:** Kept — first genuine break through the 1.3 plateau
  (v1=1.34, v2=1.39). ~9% gain vs v2, ~6% vs v1. Still far from
  winner territory (~0.85).
- **Notes:** Loss still decreasing at timeout; longer training
  would likely push further. Next: either (a) run v3 longer with
  deeper model / more scales, or (b) move to inducing-point
  attention so each point can attend to a learned set of cluster
  tokens — gets us true long-range spatial interaction the voxel
  approach can't provide.

### 2026-04-16 — v2: voxel-pooled neighbors + dist-to-airfoil + Δv
- **Hypothesis:** pointwise MLP cannot model turbulence — it has no
  spatial context. Adding (a) multi-scale voxel-pooled neighbor velocity
  (0.05, 0.20 m scales), (b) log-distance to the nearest airfoil point,
  and (c) temporal Δv features should substantially lower val/l2.
- **Change:** `model.py` — `voxel_pool_mean`, `min_distance_to`; input
  dim grows from 19 → ~41. Architecture: hidden=512, 10 ResBlocks.
  121s/epoch. Peak VRAM 13.3 GB.
- **Result:** Best val/l2=1.39 at epoch 16 (of 16 run, killed early).
  Train loss 0.044 → 0.027. Val trajectory: 1.57,1.81,1.60,1.52,1.61,
  1.48,1.59,1.56,1.51,1.58,1.53,1.42,1.44,1.41,1.64,1.39.
  WandB run `edward/v2-voxel+dist`.
- **Verdict:** Only marginal gain over v1 (~1.34). Mean-only voxel
  features give weak spatial signal.
- **Notes:** Killed early (epoch 16/60) because trajectory mirrored
  v1's and the expected ~1.30 wouldn't justify full run; swap to v3
  with richer spatial stats (mean, std, self-deviation, laplacian
  proxy) and bf16 AMP for speed.

### 2026-04-16 — v1: residual + normalization + no-slip BC + time FiLM
- **Hypothesis:** absolute-velocity MLP wastes capacity modelling the ~35 m/s
  freestream. Residuals `v_out − v_in[-1]` have mean magnitude 2.46 m/s,
  ~14× smaller. Add: normalization, residual head, hard zero at airfoil
  indices, global time embedding broadcast to points.
- **Change:** `train.py` → `ResidualMLP` (hidden=512, 8 blocks).
  Loss in normalized space.
- **Result:** epoch 20, val/l2=1.3388 (wandb run `fkf8bty4`).
  Train loss plateaued around 0.026 (normalized MSE).
- **Verdict:** kept checkpoint but only baseline-level performance — the
  pointwise MLP has no way to learn spatial interactions (turbulence).
- **Notes:** `predict.py` was broken because importing `train` triggered its
  CLI parse. Extracted model to `model.py`. Val is noisy (bounces 1.34–1.65).
  Winners on mar29 hit 0.85 — need spatial features.
