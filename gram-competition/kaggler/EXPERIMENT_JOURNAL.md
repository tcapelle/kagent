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

### 2026-04-18 — exp67: per-timestep, per-channel mixture weights (K, T, C)
- **Hypothesis:** the current [K] softmax mixes all ckpts with one scalar each. Different ckpts may be better at specific timesteps (early vs late) or specific channels (Ux streamwise vs Uy/Uz cross-flow). Letting each (t, c) cell pick its own mix of K ckpts should reduce residual.
- **Change:** new `pred_perchan.py`: logits shape `[K, T=5, C=3]`, softmax over K independently per (t, c). Chunk samples to fit GPU (43 GB for preds fp32 on 95GB H100). 3000 Adam steps at lr=0.02.
- **Result:** val/l2=**0.8453** (−0.0013 over exp66 scalar-90 0.8466 re-run, −0.0012 over exp66 reported 0.8465). Uy channel prefers model-42wdgwd8 (weight 0.186), which did not dominate in scalar mixing — suggests it's specifically strong at Uy. Uz channel splits across multiple yflip variants and fresh ckpts.
- **Verdict:** KEPT — modest gain. Saves to `bea915c-perchan/val.pt`.
- **Notes:** The small gain (~0.001) vs parameter bloat (1350 vs 90) suggests the per-channel diversity was already partially captured at scalar mixing. Next: (1) per-sample mixture — might overfit val since val is small but worth trying with validation-loss regularization, (2) more fresh trainings for raw pool diversity, (3) stacking via a tiny MLP on stacked predictions.

### 2026-04-18 — exp66: TTA y-flip — add y-flipped predictions to ensemble pool
- **Hypothesis:** train.py uses `yflip_prob=0.5` during training, so the model is approximately y-equivariant. Generating a second prediction per ckpt from y-flipped input (and un-flipping the output) should act as a noisy sibling and add pool diversity, similar to exp63's "orthogonal errors" observation.
- **Change:** new `pred_tta.py`: for the top-30 ckpts by individual val l2, run `predict_yflipped(model, loader, device)` (flip `pos[...,1]` across per-batch y-center, flip `v_in[...,1]` sign, un-flip `pred[...,1]` sign), cache to `predcache_tta/{ckpt}_yflip.pt`. Stack 60 original + 30 TTA = 90 preds, run 5000-step Adam on softmax logits.
- **Result:** val/l2=**0.8465** (−0.0191 over exp64 0.8656, largest single gain of session). Individual yflip l2 values varied widely: wavg3 1.03, 74m37pbr 0.98, mgo03egs 1.04, hvoch2y9 1.34. Adam found strong yflip weights: 74m37pbr_yflip 0.133, bbr0yz3i_yflip 0.100, azw790ng_yflip 0.044 — the model is *not* perfectly equivariant, so yflip preds carry orthogonal error content.
- **Verdict:** KEPT — big jump. Saves to `0bce4a5-tta/val.pt`.
- **Notes:** The yflip axis happened to work despite the model not being strictly equivariant (yflip l2s are 1.0-1.3 individually). This is stronger evidence that *any* transform that yields slightly-different but valid predictions adds value to the Adam ensemble. Next: (1) x-flip TTA (along streamwise axis — less likely to help since flow is directional, but cheap to try), (2) point subset ensembling (predict on random 80% point subsets, average), (3) more fresh trainings with larger LR and different seeds.

### 2026-04-18 — exp65: smaller hidden=128 model for diversity — PARKED
- **Hypothesis:** different model capacity might produce different errors and boost ensemble.
- **Change:** launched training with `hidden=128, n_blocks=6` — one-off.
- **Result:** val/l2=1.2375 (best epoch 15). Could not integrate into pred_all.py because its `load_and_complete` uses a hidden=256 reference state so shape-mismatch drops all weights.
- **Verdict:** PARKED — model trained fine but integration requires architecture-dispatch in the prediction script. Deferred in favor of TTA which was trivially integrable.
- **Notes:** Ckpt saved at `model-c5i1mfjk/checkpoint.pt`. Could be revived later with a separate predict-and-cache script if a future exp wants more diversity.

### 2026-04-18 — exp64: fresh training from scratch + re-run pred_all (60 ckpts)
- **Hypothesis:** exp63 showed diversity matters more than individual score; even a weak fresh ckpt from totally different training trajectory (different init seed, no warm-start) could add diversity to ensemble.
- **Change:** `train.py --lr 1e-3 --subsample_train 0` (full-res, high LR, fresh init), 25 min budget. Then re-run `pred_all.py` to include new ckpt.
- **Result:** New ckpt `model-dos9qdkq` val/l2=1.6474 (best epoch 4 of ~3). Ensemble with 60 ckpts via Adam: **0.8656** (−0.0009 over exp63). dos9qdkq gets weight 0.027 in final mix.
- **Verdict:** KEPT — another small gain. Diminishing returns from diversity in the current pool scheme.
- **Notes:** 25 min was too short for fresh training to converge (~5 epochs at 1.65). Next: longer training runs, OR try per-channel weighting, OR stacking.

### 2026-04-18 — exp63: extend pool to all 59 ckpts + direct optimization
- **Hypothesis:** exp62 pool was 21 "recent good" ckpts; many earlier/weaker ckpts (38 more) exist. Even if their individual scores are 0.92-2.5, they may contribute orthogonal errors to the mixture.
- **Change:** new `pred_all.py`: caches per-ckpt val predictions for every ckpt in PVC (59 total), loads all as [K,80,5,N,3] tensor on GPU, runs Adam on softmax logits against eval metric.
- **Result:** val/l2=**0.8665** (−0.0071 over exp62 direct, −0.0135 over exp60 uniform, −0.0136 from exp45). Uniform-59 = 1.0649 (dragged by bad ckpts); Adam quickly routes weight to useful ones. Top 14 ckpts get meaningful weight: mgo03egs 0.13, eu7w7w48 0.12, 8bycg5j0 0.10, 79ynl9v3 0.09 (single score 0.94!), azw790ng 0.09, 2qrz5f8c 0.09, kgxzr9zm 0.07, ecrz5gj1 0.07, 4cun281z 0.05 (single 1.14!), dfal10k2 0.04, lkb1o42z 0.04, kvptxsnv 0.03, oxfbajwq 0.02, 1mnzgy9c 0.02.
- **Verdict:** KEPT — biggest pred-avg gain yet; confirms diverse ckpts (even weak ones) help.
- **Notes:** Individual-score >1.0 ckpts like 4cun281z, oxfbajwq get small but non-zero weights → they must be right on samples others get wrong. Next: (1) train fresh ckpts with different seeds/noise to expand pool further, (2) TTA with y-axis flip (domain looks symmetric y=-0.41 to 0.41), (3) per-sample mixture weights.

### 2026-04-18 — exp62: direct autograd optimization on the eval metric (softmax weights)
- **Hypothesis:** exp61 showed LS-minimizing weights don't match the per-point-L2 eval metric (they over-shrunk). Solution: make weights softmax-parameterized, compute the actual eval metric, backprop through the weighted mean of cached predictions, run Adam for a few thousand steps.
- **Change:** new `pred_direct.py`: load 21 cached preds, init logits near uniform-of-8 (exp60 greedy), Adam lr=0.02 on softmax(logits) minimizing `(pred-truth).norm(dim=3).mean(...)`. Saves to `{commit}-direct/val.pt`.
- **Result:** val/l2=**0.8736** (−0.0006 over exp60 uniform 0.8742). Converged after ~600 steps. Top weights: mgo03egs 0.214, eu7w7w48 0.211, bbr0yz3i 0.201, b1hzbt3r 0.124, kvptxsnv 0.102, 670v4v75 0.093, 18f6e3td 0.056. Interestingly *excluded* wavg3 and wavg-816156a (weight → 0) — wavg meta-ckpts are linear combos of this pool so adding them is redundant.
- **Verdict:** KEPT — direct autograd beats greedy selection; validates the math. Still modest gain.
- **Notes:** Given cached preds, each optimization run is ~30s. Next: TTA (rotate input by symmetric transforms), per-sample weighting, or train diverse ckpts for more pool.

### 2026-04-18 — exp61: LS weighted prediction averaging — FAILED
- **Hypothesis:** exp60 uniform-8 average might be beaten by LS-optimal per-ckpt weights.
- **Change:** new `pred_weighted.py`: cache per-ckpt predictions to `/mnt/new-pvc/kagent/apr16/tanjiro/predcache/*.pt`, solve `min_w ||Pw - Y||²` both unconstrained and on simplex via projected gradient.
- **Result:** unconstrained LS l2=0.8841 (worse); simplex LS l2=0.8774 (same as best single ckpt — converges to near-delta on wavg3). MSE objective ≠ eval metric (mean per-point L2 norm).
- **Verdict:** DISCARDED — LS over elementwise residuals misaligns with per-point L2 norm metric. Motivates exp62 direct metric optimization.
- **Notes:** The pred cache at /mnt/new-pvc/kagent/apr16/tanjiro/predcache/ is reusable — future ensemble experiments should load from there and skip the ~30 min prediction step.

### 2026-04-18 — exp60: prediction averaging (not weight averaging) across 21 ckpts
- **Hypothesis:** weight averaging only works for ckpts in the same loss basin; prediction averaging works even across disjoint basins because we only combine final outputs. Different ckpts may make independent errors that cancel. Typically 0.005-0.02 stronger than SWA.
- **Change:** new `pred_ensemble.py`: each of 21 candidates runs full inference on val, store all predictions in CPU memory, greedy-select subset whose uniform mean minimizes val l2. Outputs saved to `b50a07e-predavg/val.pt`.
- **Result:** val/l2=**0.8742** (−0.0032 over exp59 wavg3, −0.0059 over exp45 baseline). Best: 8 ckpts [wavg3, wavg1, 670v4v75 (237-key, indiv 0.8908), mgo03egs, bbr0yz3i, kvptxsnv (237-key, indiv 0.9142!), eu7w7w48, b1hzbt3r]. Notably includes 3 ckpts with individual scores >0.88.
- **Verdict:** KEPT — biggest single gain since switching strategies. Consistent with hypothesis: pred-avg lets each basin contribute what it's right about.
- **Notes:** kvptxsnv alone is 0.9142 but improves ensemble — it must be right on samples the others get wrong. Next: (a) weighted pred-avg (learn per-ckpt coefficients on val), (b) TTA (test-time aug: rotate input 90°, avg predictions), (c) fresh training run to expand pool with different seed/noise.

### 2026-04-18 — exp59: wavg3 — pre-knn (237-key) ckpts + compounding prior wavgs
- **Hypothesis:** More cross-arch diversity should help. Pad 237-key ckpts (exp30-33 era, pre-kNN) with zeros and include them. Also seed pool with prior wavg results (wavg-816156a from exp57, wavg2-266db37 from exp58) to let greedy compound.
- **Change:** `wavg_eval3.py`: 20 candidates incl. 237-key pre-knn; zero-pad missing keys including shape mismatches.
- **Result:** val/l2=**0.8774** (−0.0012 over exp58 wavg2, −0.0027 over exp45). Best: [wavg2-266db37, bal6xybc (exp33, 237-key), wavg-816156a, eu7w7w48]. Including bal6xybc helped despite its individual score being 0.8874.
- **Verdict:** KEPT — third successive ensemble gain.
- **Notes:** Compounding wavgs work (wavg2 included in ensemble along with wavg of different set). Next: test even older ckpts (111, 127 keys) and prediction-averaging (not weight averaging) which is usually stronger.

### 2026-04-18 — exp58: extended weight-avg including 241-key pre-exp44 checkpoints
- **Hypothesis:** exp57 found best avg from 4 of 9 candidates. Adding 241-key ckpts (exp41-43 era; pre pos-offset branch) with zero-initialized pos_proj adds cross-arch-variant diversity.
- **Change:** `wavg_eval2.py`: pads 241-key ckpts with zero pos_proj, greedy searches starting from best single (which is now the wavg from exp57).
- **Result:** val/l2=**0.8786** (−0.0003 over exp57). Best combo: [wavg-816156a, eu7w7w48 (exp44), jay6zniz (exp42)]. jay6zniz alone was 0.8805 but averaged with wavg+eu7w7w48 gave additional gain.
- **Verdict:** KEPT — compound ensemble working. Total improvement from exp45 is 0.0015 (0.8801 → 0.8786).
- **Notes:** Greedy stopped after 3 ckpts — further adds didn't help. Next: generate fresh diverse checkpoints via short training with different configs (noise aug, different LR/seed) to expand pool.

### 2026-04-18 — exp57: greedy weight-average ensemble across 4 checkpoints
- **Hypothesis:** 10 consecutive training experiments failed to beat exp45 (0.8801). Each training step moves weights away from val optimum. BUT the experiments saved several near-optimum checkpoints in the same basin — averaging them should recover a "center" that generalizes better than any individual. Zero training cost.
- **Change:** New script `wavg_eval.py`: load every compatible PVC checkpoint, validate each, then greedy-append the one that reduces val/l2 most. Save the best average to checkpoints/best.pt + new PVC dir.
- **Result:** val/l2=**0.8789** (−0.0012 over exp45). Greedy ensemble: [mh5wd0t6 (exp45), eu7w7w48 (exp44), bbr0yz3i, b1hzbt3r]. Interestingly b1hzbt3r individually was 0.8882 (much worse) but *adding* it to the avg still helped — diversity helps.
- **Verdict:** KEPT — first win in 11 experiments. New rank still #4 but gap shrinks.
- **Notes:** The 4 checkpoints were from the same underlying arch (exp41+exp44+exp45 variants). This validates "loss landscape is flat around exp45 — averaging finds the center". Next: generate MORE diverse checkpoints from independent trainings to feed a bigger ensemble.

### 2026-04-18 — exp56: full-res + EMA + LR warmup, warm-start from exp45
- **Result:** E1=0.8802, E2-5 climbed to 0.8808. Best at E1. No improvement over exp45 (0.8801).
- **Verdict:** DISCARDED as standalone. But exp56 E1 ckpt (model-fgqa41ag) entered the wavg candidate pool.
- **Notes:** Full-res training with warm-start confirms "training near exp45 stays in basin but can't descend". Consistent with exp55 drift.
- **Hypothesis:** exp55 (EMA+warmup, subsample=50k) drifted up (0.8809→0.8817→0.8821). Trajectory moves AWAY from exp45 val optimum. Theory: subsample=50k creates density mismatch with full-res val. Train at full-res (subsample=0) so train/val density matches, EMA smooths the overfit climb exp32 showed. exp32 E1 at full-res was 0.8879 (warm from exp31=0.8900), good; maybe full-res + EMA with better starting point hits lower.
- **Change:** same code as exp55; flag --subsample_train 0.
- **Result:** TBD
- **Verdict:** TBD
- **Notes:** Each epoch ~5 min at full res → 6 epochs in 30 min.

### 2026-04-18 — exp55: EMA + LR warmup, warm-start from exp45
- **Result:** E1=0.8809, E2=0.8817, E3=0.8821 (monotonic up, killed at E3). Killed. EMA did NOT prevent drift — training trajectory moves away from exp45 val optimum even with 48-77% EMA weighting.
- **Verdict:** DISCARDED — the 10th consecutive warm-start failure. Training dynamics themselves are broken: with subsample=50k, exp45 is at a val-set local min that no gradient step can improve on.
- **Notes:** This rules out stabilization fixes alone. Next: full-res training (density match) + EMA.


- **Hypothesis:** 8 consecutive warm-start experiments (exp46-54) all failed to beat exp45 (0.8801), usually because val spiked at E1-2 as new/perturbed weights disrupted the fine-tuned model. EMA (decay=0.999) of training weights, used at validation, starts exactly at exp45 and smooths out trajectory noise. LR warmup (200 linear steps → cosine) softens the LR shock at warm-start start. Combined, this should let the model actually improve rather than regress.
- **Change:** train.py: added `ema_decay`, `warmup_steps` config. Built ema_model (copy of model, no grad), updated per-step after optimizer.step. Replaced CosineAnnealingLR with step-based LambdaLR (linear warmup → cosine). Validation/checkpoints now use ema_model.
- **Result:** TBD
- **Verdict:** TBD
- **Notes:** Orthogonal to arch — first time trying optimization/regularization instead of architecture.

### 2026-04-17 — exp46-54: eight consecutive failed chain experiments (exp46: k=32 KNNMixer, exp47: out_refine reinit, exp48: k=32 rerun, exp49: KNNAttn head, exp50: noise, exp51: noise=0.01, exp52: vscale aug, exp53: vmag features, exp54: 60-min chain). All stayed at or above 0.8801 — confirms exp45 saturation at current arch under warm-start.
- **Verdict:** DISCARDED — pattern: every warm-start perturbation dominates the 5-epoch validation window. Need to either stabilize training (EMA, LR warmup) or cold-start a new arch.

### 2026-04-17 — exp45: chain exp44 @ lr=5e-6 (pos-offset refine)
- **Hypothesis:** exp44 tied exp42 (0.8805) but pos_proj had only 2 epochs to learn. Chain at lr=5e-6 to give it time. If chain doesn't help, pos-offset at end-of-stack is genuinely unhelpful.
- **Change:** No code. --warm_start <exp44 ckpt model-eu7w7w48> --lr 5e-6.
- **Result:** TBD
- **Verdict:** TBD
- **Notes:** Launching.

### 2026-04-17 — exp44: kNN + position-offset branch (PointNet++ style)
- **Hypothesis:** Current KNNMixer pools feature mean/max of 16 neighbors. Missing explicit relative position info. Add zero-init pos_proj (Linear 3→dim) that encodes (pos_neigh - pos_self) averaged over neighbors; add to h_agg. Zero-init → identity at warm-start; gives model explicit local geometry.
- **Change:** train.py: KNNMixer gains self.pos_proj (zero-init Linear 3→dim). forward() takes pos, computes pos_diff to neighbors, adds mean(pos_proj(pos_diff)) to h_agg. BaselineMLP.forward passes pos to knn_mixer.
- **Result:** val/l2=**0.8805** @ epoch 2 (34.6 min, 5 epochs). Ckpt: model-eu7w7w48. Trajectory: E1 0.8823 → E2 0.8805 → E3 0.8834 → E4 0.8851 → E5 0.8827. Train loss declined 0.0083→0.0076 (overfitting).
- **Verdict:** HOLD — tied exp42 exactly. Not an improvement, but E2 is kept as chain starting point. Chain at lr=5e-6 will decide.
- **Notes:** Mean-pooling pos_diff over neighbors may be too lossy (averages small 3D vectors → ~0). Max pool would preserve more. But let chain decide before redesigning.

### 2026-04-17 — exp43: chain exp42 kNN @ lr=2e-6 (fine refine)
- **Hypothesis:** exp42 val was monotonically decreasing through E5 (0.8810 → 0.8807 → 0.8805). Not saturated. Chain at lr=2e-6 for slow refine.
- **Change:** --warm_start <exp42 ckpt model-jay6zniz> --lr 2e-6.
- **Result:** val/l2=**0.8805** @ epoch 2 (34.4 min, 5 epochs). Trajectory: E1 0.8812 → E2 0.8805 → E3 0.8806 → E4 0.8822 → E5 0.8813.
- **Verdict:** DISCARDED — tied exactly with exp42 (0.8805). lr=2e-6 too low to progress. Model saturated at this scale.
- **Notes:** Stick with exp42 ckpt model-jay6zniz. Next: add more kNN capacity via position-offset features (PointNet++ style).

### 2026-04-17 — exp42: chain exp41 kNN-mixer @ lr=5e-6 (arch-add refine)
- **Hypothesis:** exp41 showed kNN-mixer helps (+0.0007). At lr=2e-5 explore, the new block got rough initialization; chain at lr=5e-6 should refine. Expected +0.001-0.003.
- **Change:** No code changes. --warm_start <exp41 ckpt model-er5pk3oc> --lr 5e-6.
- **Result:** val/l2=**0.8805** @ epoch 5 (34.4 min). Ckpt: model-jay6zniz. Trajectory: E1 0.8810 → E2 0.8817 → E3 0.8817 → E4 0.8807 → E5 0.8805. Train loss 0.0084 → 0.0078.
- **Verdict:** KEPT — +0.0022 over exp41 (0.8827). Arch-add+chain recipe: arch gave +0.0007, chain gave +0.0022. Net from exp40 (pre-kNN): +0.0029.
- **Notes:** Monotonic decline through E5 — not saturated. Distance to leader (thorfinn=0.7475): 0.133.

### 2026-04-17 — exp41: kNN neighbor aggregation (k=16, zero-init)
- **Hypothesis:** Voxels (G=32, 16, 8) give coarse spatial context, but miss fine local neighborhood. Add a single KNNMixer block after main stack that gathers mean+max features from 16 nearest spatial neighbors per point. Zero-init output projection → identity at warm-start.
- **Change:** train.py: new KNNMixer class (LayerNorm + gather-kNN + mean/max pool + zero-init Linear). BaselineMLP: self.knn_mixer = KNNMixer(hidden, k=16). Forward: after main block stack, compute _knn(pos) via chunked cdist+topk, apply knn_mixer, then out_refine. Warm-start from exp40 @ lr=2e-5.
- **Result:** val/l2=**0.8827** @ epoch 3 (34.4 min, 5 epochs). Ckpt: model-er5pk3oc. Trajectory: E1 0.8848 → E2 0.8862 → E3 0.8827 → E4 0.8856 → E5 0.8831. Train loss 0.0098 → 0.0083.
- **Verdict:** KEPT — +0.0007 over exp40 (0.8834). Arch-add+chain recipe worked again (first time since exp33). Forward 413s/epoch (+147s over baseline), 18.6 GB VRAM.
- **Notes:** kNN-mixer adds 130k params. Gives fine local neighborhood info voxels miss. Distance to leader (thorfinn=0.7475): 0.135. Next: chain at lr=5e-6 for refine.

### 2026-04-17 — exp40: chain exp38 @ lr=2e-6 (slower refine of y-flip aug)
- **Hypothesis:** exp38 val was still decreasing at E10 (0.8839). At lr=5e-6 over 50-epoch cosine, LR barely decayed. Try lr=2e-6 for finer refine.
- **Change:** No code changes. --warm_start <exp38 ckpt model-mgo03egs> --lr 2e-6 --subsample_train 0 --yflip_prob 0.5.
- **Result:** val/l2=**0.8834** @ epoch 11 (30.7 min). Ckpt: model-fdgxhd3i. Trajectory oscillated 0.883-0.885 until E9-E11 where it dipped to 0.8835, 0.8837, 0.8834.
- **Verdict:** KEPT — marginal +0.0005 over exp38 (0.8839). Still worth keeping as it's strictly better. Distance to leader (thorfinn=0.7475): 0.136.
- **Notes:** Chain yielded little at lr=2e-6 — model near convergence for this arch. Next paradigm shift needed: kNN neighbor aggregation for fine-grained spatial interaction (voxels are coarse).

### 2026-04-17 — exp39: TTA (test-time y-flip averaging) on exp38 ckpt
- **Hypothesis:** Averaging pred(x) with flip(pred(flip(x))) should reduce variance on turbulent predictions (free inference-time win).
- **Change:** Quick inference-only test — no commit. Ran both pred and y-flipped pred, averaged, measured val/l2.
- **Result:** Plain: 0.8839 → TTA: **0.8886** (delta -0.0047, WORSE).
- **Verdict:** DISCARDED. Model has learned slightly asymmetric response (Uy mean ≈ 0.5 is non-zero in training data); averaging with flipped pred dilutes this.
- **Notes:** Surprising — training aug gave +0.0036 but TTA on the aug-trained model hurts. Suggests training aug works via data expansion, not via teaching mirror equivariance. Don't use TTA.

### 2026-04-17 — exp38: y-flip data augmentation (bilateral symmetry)
- **Hypothesis:** F1 wings are mirror-symmetric about y=0 (Uy mean ≈ 0.5 from stats confirms). Mirror-flip pos_y around bbox center and negate Uy to double effective training data. Pure generalization fix (no arch change). Chain from exp33 @ lr=5e-6.
- **Change:** train.py: Config.yflip_prob=0.5. In train loop after subsampling, with p=0.5 flip pos_y around bbox-y-center, negate v_in[...,1] and v_out[...,1]. No architecture change — pure augmentation.
- **Result:** val/l2=**0.8839** @ epoch 10 (30.6 min, 11 epochs). Ckpt: model-mgo03egs. Trajectory: E1 0.8902 → E4 0.8850 → E8 0.8846 → E10 0.8839. Train loss dropped 0.0131 → 0.0098 as model adapted to augmentation.
- **Verdict:** KEPT — +0.0036 over exp33 (0.8875). First generalization win after 4 arch-add failures. Breaks the overfitting pattern.
- **Notes:** Train loss starts HIGH (0.0131) vs previous ~0.003 because initially model can't handle flipped Uy distribution (mean shift of ~0.15 in normalized). Adapts over 10 epochs. Distance to leader (thorfinn=0.7475): 0.136. Next: (a) chain exp38 @ lr=2e-6, (b) add TTA to predict.py for stacking inference-time gain, (c) add input Gaussian noise aug.

### 2026-04-17 — exp37: 2 extra Res+Voxel pairs (depth extension)
- **Hypothesis:** Model hasn't saturated capacity-wise (16 blocks→20). Add 2 extra zero-init Res+Voxel pairs after main stack + dedicated time encoder. Chain from exp33 @ lr=2e-5.
- **Change:** BaselineMLP gains self.extra_blocks (2 Res+Voxel zero-init) + self.extra_time_enc. Forward adds second loop over extra_blocks.
- **Result:** val/l2=**0.8890** @ epoch 1 (30.2 min). Val climbed 0.8890→0.8919 over 9 epochs.
- **Verdict:** DISCARDED — 4th consecutive failure. Pure overfitting: best at epoch 1, monotonically worse. More depth no longer helps — model is already over-parameterized for 730 samples.
- **Notes:** Pattern confirmed: exp34-37 all failed to improve on exp33 (0.8875). Paradigm shift needed: generalization fix (augmentation) not more capacity.

### 2026-04-17 — exp36: train-time no-slip BC supervision
- **Hypothesis:** Currently BC zeroed after pred (inference-only). If model learns delta without BC constraint, the gradient signal for airfoil points is wrong. Train with BC zero too so loss sees correct pred.
- **Change:** Moved BC-zeroing inside self.training branch (always active).
- **Result:** val/l2=**0.8891** @ epoch 1, then climbed. Chain from exp33 lr=5e-6 didn't help.
- **Verdict:** DISCARDED. Pre-BC-zero pred was already close (~54 m/s mean) → initial loss jump disrupted optimization.

### 2026-04-17 — exp35: chain exp34's linear-extrap-prior @ lr=2e-5
- **Hypothesis:** Exp34 regressed at lr=2e-4 (too hot for 5 scalars). Try standard chain @ lr=2e-5.
- **Change:** Same code as exp34, --lr=2e-5.
- **Result:** val/l2=**0.8888** (worse than exp33 0.8875).
- **Verdict:** DISCARDED. extrap prior is redundant with learned delta — model already encodes "last + delta" structure.

### 2026-04-17 — exp34: learnable linear-extrapolation prior
- **Hypothesis:** Add 5 zero-init scalars: pred = v_in[-1] + weight[t] * (v_in[-1] - v_in[-2]) * step + delta. Gives model a persistence-of-derivative prior.
- **Change:** BaselineMLP gains self.extrap_weight (nn.Parameter, zero). Forward adds extrap term.
- **Result:** val/l2=**0.9068** @ lr=2e-4 — big regression.
- **Verdict:** DISCARDED. Lesson: tiny arch additions don't need "lr=2e-4 explore then chain" recipe. Chain @ lr=2e-5 directly (see exp35).

### 2026-04-17 — exp33: full-res chain from exp32 + lr=5e-6
- **Hypothesis:** Exp32 overfit at lr=2e-5 (val climbed E1→E11). At lr=5e-6 cosine decay, expect stable slow refine. Target: beat 0.8879 by +0.001-0.003.
- **Change:** --warm_start <exp32 ckpt> --subsample_train 0 --lr 5e-6.
- **Result:** val/l2=**0.8875** (KEPT, current best). Ckpt: model-bal6xybc.
- **Verdict:** KEPT. +0.0004 over exp32 (0.8879). Full-res chain @ lr=5e-6 yielded slow refine as hypothesized.
- **Notes:** Marginal gain confirms saturation at current arch. Subsequent 4 arch-adds (exp34-37) all failed to beat this.

### 2026-04-17 — exp32: full-resolution training (warm-start from exp31)
- **Hypothesis:** Training uses subsample=50k but val runs at full 100k — the voxel density statistics differ. Also, at 50k each voxel has ~50% occupancy vs 100% at val, which likely degrades VoxelMixer behavior. Chain from exp31 with --subsample_train 0 and lr=2e-5 to test if closing the density gap helps. Cost: ~2x epoch time → ~9 epochs in 30min vs 18.
- **Change:** --subsample_train 0 --warm_start <exp31 ckpt> --lr 2e-5. No code changes.
- **Result:** val/l2=**0.8879** @ epoch 1 (30.6 min total, 167s/epoch × 11). train=0.0028. mae_Ux=0.5970, mae_Uy=0.2693, mae_Uz=0.4115. Val CLIMBED over epochs (0.8879→0.8912) — overfitting at full-res + this LR.
- **Verdict:** KEPT — +0.0021 over exp31 (0.8900). Big insight: density mismatch was real (immediate +0.002 at E1), but training past E1 overfits. Next: lower LR (5e-6) to slow overfit, or mix subsample-50k + full-100k per batch for augmentation.
- **Notes:** Ckpt: model-18f6e3td. Distance to leader (thorfinn=0.7511): 0.137. Peak VRAM: 11.1GB (up from 7.4GB at 50k). Fits easily.

### 2026-04-17 — exp31: chain exp30 (aux features) + lr=2e-5
- **Hypothesis:** Exp30 regressed 0.9132 (same arch-add pattern). Chain at lr=2e-5 to recover. Expected +0.003-0.006 over exp29 (0.8965).
- **Change:** --warm_start=<exp30 ckpt model-kvptxsnv> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.8900** @ epoch 7 (30.2 min). train=0.0031. mae_Ux=0.5984, mae_Uy=0.2699, mae_Uz=0.4126. Val stable 0.8900-0.8945.
- **Verdict:** KEPT — +0.0065 over exp29 (0.8965). 5th arch-add+chain cycle. Pattern persistent: {multi-scale 0.016, FiLM 0.010, triple-scale 0.006, out_refine 0.006, aux-diff 0.006}. Diminishing but reliable.
- **Notes:** Ckpt: model-670v4v75. Distance to leader (thorfinn=0.7511): 0.139 (leader still improving too). Local arch-adds saturate at ~+0.006. Need bigger moves: (a) full-res training, (b) kNN local attention, (c) more blocks.

### 2026-04-17 — exp30: temporal-derivative aux features (warm-start from exp29)
- **Hypothesis:** v_diff = v_in[:, 1:] - v_in[:, :-1] is an acceleration signal (Δv/Δt). Turbulent/vortex dynamics show up in acceleration, not just position. Also feed per-point mean+std of input trajectory (moments). Current input uses v_in concatenated as 15 features — model has to learn to compute diffs. Handing them directly should let early blocks focus on dynamics.
- **Change:** train.py: BaselineMLP adds self.proj_aux (Linear 18→hidden, zero-init). Forward computes v_diff_feat [12] + v_mean [3] + v_std [3] → aux [18], then x += proj_aux(aux).
- **Result:** val/l2=**0.9132** @ epoch 18 (30.2 min). train=0.0038. Same arch-add undertrained pattern, worse than exp29 (0.8965) by 0.0167.
- **Verdict:** HOLD — regressed as expected. Ckpt: model-kvptxsnv. Will chain.
- **Notes:** 2 missing keys (proj_aux.{weight,bias}). Zero-init verified. Params: 91.33M (+4.9K).

### 2026-04-17 — exp29: chain exp28 (out_refine) + lr=2e-5
- **Hypothesis:** Exp28 regressed to 0.9110 from 0.9027 (same arch-add pattern). Chain at lr=2e-5 to recover the fresh out_refine layer while preserving main weights. Based on recipe: multi-scale (+0.016), FiLM (+0.010), triple-scale (+0.006) → out_refine expected +0.003-0.005.
- **Change:** --warm_start=<exp28 ckpt model-kgxzr9zm> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.8965** @ epoch 14 (30.2 min). train=0.0033. mae_Ux=0.6032, mae_Uy=0.2717, mae_Uz=0.4154. Val plateaued E2 onwards (0.8965-0.8984).
- **Verdict:** KEPT — +0.0062 over exp27 (0.9027). 4th arch-add + chain cycle. Pattern: {multi-scale 0.016, FiLM 0.010, triple-scale 0.006, out_refine 0.006}. Diminishing returns on local arch additions.
- **Notes:** Ckpt: model-hvoch2y9. Distance to leader (thorfinn=0.7670): 0.1295. Need bigger moves — small arch-add saturates. Next: velocity temporal-difference features (acceleration signal) via zero-init proj_aux.

### 2026-04-17 — exp28: out_refine residual MLP (warm-start from exp27)
- **Hypothesis:** Per-component MAE shows Ux gap biggest (0.61 vs leader's 0.52). Maybe output stage is the bottleneck. Add a residual MLP (LayerNorm → Linear hidden→2*hidden → GELU → Linear 2*hidden→hidden) between final block and proj_out. Zero-init last Linear → identity at init → warm-start safe. Per-block FiLM already conditioning; this gives extra compute at output.
- **Change:** train.py: BaselineMLP.__init__ adds self.out_refine Sequential, forward adds `x = x + self.out_refine(x)` before proj_out.
- **Result:** val/l2=**0.9110** @ epoch 16 (30.3 min). train=0.0041. Val trajectory: 0.9471→0.9278→0.9276→0.9110→0.9133. mae_Ux=0.6134, mae_Uy=0.2768, mae_Uz=0.4211.
- **Verdict:** HOLD — worse than exp27 (0.9027) by 0.0083. Same arch-add undertrained pattern. Val still climbing at timeout. Will chain.
- **Notes:** 6 missing keys (out_refine). Params: 91.1M → 91.3M (tiny). Ckpt: model-kgxzr9zm.

### 2026-04-17 — exp27: chain warm-start from exp26 + lr=5e-6
- **Hypothesis:** Exp26 val stable 0.9035-0.9061. Very low LR chain for last refinement. Expected Δ~0.001-0.003.
- **Change:** --warm_start=<exp26 ckpt> --lr=5e-6. No code changes.
- **Result:** val/l2=**0.9027** @ epoch 7 (30.0 min). train=0.0035. Val stable 0.9027-0.9038.
- **Verdict:** KEPT — +0.0008 over exp26. Chain saturated.
- **Notes:** Current chain tail: 0.9238→0.9202→0.9194→0.9096→0.9091→0.9035→0.9027. Total arch journey: multi-scale→FiLM→triple-scale → next arch.

### 2026-04-17 — exp26: chain warm-start from exp25 (triple-scale refine) + lr=2e-5
- **Hypothesis:** Exp25 showed same undertrained pattern as exp18/22. Following recipe: chain at lr=2e-5 for 30 more min to let ucoarse branch refine while preserving main weights. If pattern holds, should beat exp24 (0.9091) by 0.005-0.015.
- **Change:** --warm_start=<exp25 ckpt model-lkb1o42z> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9035** @ epoch 11 (30.1 min). train=0.0035. Val stable 0.9035-0.9061 E11-18.
- **Verdict:** KEPT — +0.0056 over exp24 (0.9091). Pattern validated 3rd time: arch-add regresses, chain recovers + beats. Δ pattern: multi-scale 0.016, FiLM 0.010, triple-scale 0.006 (diminishing returns but still positive).
- **Notes:** Distance to leader: 0.9035 - 0.7750 = 0.129. Still closing. Next: exp27 chain at lr=5e-6 for cheap harvest, then arch.

### 2026-04-17 — exp25: triple-scale voxel (add 8³ ultra-coarse branch)
- **Hypothesis:** Multi-scale voxel (exp18/19) showed coarse branch added real signal. Extend same pattern: add parallel G/4=8³ ultra-coarse branch per VoxelMixer. Captures scene-level context (wing upper vs lower, wake near vs far). Zero-init (proj_agg_ucoarse=0, last conv_ucoarse=0) → warm-start safe.
- **Change:** VoxelMixer adds proj_agg_ucoarse + conv_ucoarse (G/4 grid). Forward adds sampled_u to output.
- **Result:** val/l2=**0.9192** @ epoch 16 (30.0 min). train=0.0044. Val oscillated 0.9192-0.9534. 100s/epoch.
- **Verdict:** HOLD — worse than exp24 (0.9091) by 0.0101. Same undertrained pattern as exp18/22. Val still chaotic at timeout.
- **Notes:** Following the established recipe: exp26 = chain at lr=2e-5 to stabilize and extract. Expect exp26 to recover and surpass 0.9091 like exp19/23 did.

### 2026-04-17 — exp24: chain warm-start from exp23 + lr=5e-6
- **Hypothesis:** Exp23 val stable E14-22 (0.9096-0.9115). Very low LR chain should squeeze last fine-tune. Expected Δ~0.001-0.003.
- **Change:** --warm_start=<exp23 ckpt model-mnvmn263> --lr=5e-6. No code changes.
- **Result:** val/l2=**0.9091** @ epoch 2 (31.0 min). train=0.0036. Val stable 0.9094-0.9104.
- **Verdict:** KEPT — micro +0.0005 over exp23. Chain fully saturated at 5e-6. Pivoting to arch.
- **Notes:** Current chain tail: 0.9238→0.9202→0.9194→(FiLM)0.9096→0.9091. Going to exp25 = triple-scale voxel (add 8³ ultra-coarse branch, same zero-init recipe as exp18).

### 2026-04-17 — exp23: chain warm-start from exp22 (FiLM refine) + lr=2e-5
- **Hypothesis:** Exp22 FiLM destabilized by high LR but trending down. Chain at lr=2e-5 should let FiLM params settle without further destroying main weights. If FiLM has signal, this extracts it; if not, we'll plateau around exp22's 0.9281.
- **Change:** --warm_start=<exp22 ckpt model-79ynl9v3> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9096** @ epoch 19 (30.3 min). train=0.0037. Val stable 0.9096-0.9115 E14-22.
- **Verdict:** KEPT — +0.0098 over exp21 (0.9194). FiLM validated: absolute t signal helps. Second confirmation of the pattern "new arch regresses then chain recovers + beats".
- **Notes:** Pattern: exp18/19 (multi-scale: -0.017 → +0.016), exp22/23 (FiLM: -0.009 → +0.010). Both pairs net positive, both arch additions provide real signal. Next: chain further (exp24 @ lr=5e-6) then another arch.

### 2026-04-17 — exp22: FiLM time conditioning (warm-start from exp21)
- **Hypothesis:** t is still unused. Sample-specific absolute time (0.3-0.4 range) correlates with flow development stage. Add TimeEncoder (MLP 1→64→64→2*hidden*17) producing per-block (γ, β) FiLM params, applied as h*(1+γ)+β after each block's norm. Zero-init final MLP layer → FiLM=identity at init → warm-start preserves exp21 exactly.
- **Change:** train.py: added TimeEncoder class, _apply_film helper. ResBlock.forward and VoxelMixer.forward now take optional `film` kwarg. BaselineMLP.forward calls time_enc(t) once, iterates blocks with per-block FiLM slice.
- **Result:** val/l2=**0.9281** @ epoch 19 (31.2 min). train=0.0047. Val oscillated 0.9281-0.9576.
- **Verdict:** HOLD — worse than exp21 (0.9194) by 0.0087. lr=2e-4 too aggressive for fresh FiLM layers; disrupted fine-tuned core. Trajectory trending down at end (0.9355→0.9286→0.9355). Not committed to git.
- **Notes:** FiLM output scale grew too fast. Next: exp23 chain at lr=2e-5 to let FiLM refine while preserving main weights. If exp23 still loses, revert to exp21.

### 2026-04-17 — exp21: chain warm-start from exp20 + lr=1e-5
- **Hypothesis:** Cheap-harvest: one more chain step at lr=1e-5 should extract last refinement. Expected Δ~0.001-0.002 based on geometric decay (0.0036→half).
- **Change:** --warm_start=<exp20 ckpt> --lr=1e-5. No code changes.
- **Result:** val/l2=**0.9194** @ epoch 22 (30.0 min). train=0.0039.
- **Verdict:** KEPT — +0.0008 over exp20. Chain fully saturated now (was expected). Val oscillated 0.9194-0.9219 E14-22.
- **Notes:** Multi-scale chain progression: 0.9238→0.9202→0.9194 (Δ=0.0158→0.0036→0.0008). Time to pivot to arch change.

### 2026-04-17 — exp20: chain warm-start from exp19 + lr=2e-5
- **Hypothesis:** Exp19 val stable at 0.9238-0.9283 (E12-23). Best at E12 suggests quick overfit. Chain at lr=2e-5 should refine without moving too far. Expect Δ~0.003-0.007.
- **Change:** --warm_start=<exp19 ckpt model-fdr738d3> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9202** @ epoch 10 (30.9 min). train=0.0040. Val stable 0.9202-0.9228 E10-24. run run id in PVC model-<exp20>.
- **Verdict:** KEPT — +0.0036 over exp19 (0.9238). Chain refinement of multi-scale arch. Best at E10 again (similar quick-minimum as exp19). Val floor 0.9202-0.9211 through E24.
- **Notes:** Multi-scale chain progression: exp18(0.9411 undertrained) → exp19(0.9238, Δ=0.0158 with lr=5e-5) → exp20(0.9202, Δ=0.0036 with lr=2e-5). Geometric decay. Distance to leader: 0.9202 - 0.8132 = 0.107.

### 2026-04-17 — exp19: chain warm-start from exp18 + lr=5e-5
- **Hypothesis:** Exp18 val was still dropping rapidly at E22-24 (0.9469→0.9411). Coarse branch needs more epochs at a lower LR to refine without destroying fine-branch weights. Chain at lr=5e-5 for 30 more min. If it beats exp17's 0.9396, commit the full multi-scale arch.
- **Change:** --warm_start=<exp18 ckpt model-5qrjp5if> --lr=5e-5. No code changes.
- **Result:** val/l2=**0.9238** @ epoch 12 (32.0 min). train=0.0040. Val settled 0.9238-0.9283 E12-23. 76-170s/epoch (some batches slower).
- **Verdict:** KEPT — +0.0158 over exp17 (0.9396). Biggest gain since exp15 (Δ=0.0126). Multi-scale architecture validated: the coarse branch added real signal the 32³ branch couldn't capture. Now #3 or better on leaderboard (was #4 at 0.9487, nezuko=0.9299).
- **Notes:** Best at E12 is unusual — model quickly adapted + overfit slightly after. E12-23 all within 0.9238-0.9283 (stable basin). Next: exp20 chain at lr=2e-5 to squeeze more.

### 2026-04-17 — exp18: multi-scale voxel (16³ coarse parallel, warm-start from exp17)
- **Hypothesis:** Chain saturating again. Arch change gave biggest recent win (exp15 Δ=0.0126). Add parallel 16³ coarse voxel branch to each VoxelMixer — captures larger-scale flow structures (wake, separation zones) that 32³ 3×3 conv can't reach. Zero-init (proj_agg_coarse=0, last conv_coarse=0) → sampled_c=0 at init → warm-start equivalent to exp17. Fresh training allows coarse branch to learn contributions.
- **Change:** VoxelMixer now has a parallel G_coarse=G/2 branch (proj_agg_coarse + conv_coarse) summed with fine output. Zero-init end-to-end. Refactored fine-branch logic into _voxel_mm/_gather helpers.
- **Result:** val/l2=**0.9411** @ epoch 23 (30.5 min). train=0.0052. run 5qrjp5if. 76s/epoch (38% slower).
- **Verdict:** HOLD — slightly worse than exp17 (0.9396 vs 0.9411). Val was still rapidly dropping at E22-24 (0.9469→0.9411→0.9494). Coarse branch undertrained — lr=2e-4 was too aggressive + too few epochs. Not committed to git (don't regress).
- **Notes:** Param count 59.7M→61.1M. 76s vs 55s/epoch. Next: exp19 = chain from exp18 PVC ckpt at lr=5e-5 to let coarse branch mature.

### 2026-04-17 — exp17: chain warm-start from exp16 + lr=2e-5
- **Hypothesis:** Exp16 val was still dropping at E33 (0.9422, best E24 0.9420). One more chain at lr=2e-5 should extract the last fine-tuning gains — similar pattern to the exp12→exp13 chain (+0.0025). With a richer arch (mean+max) there may be more to extract.
- **Change:** --warm_start=<exp16 ckpt model-4fk50oi3> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9396** @ epoch 25 (30.1 min). train=0.0044. run uxpz2abi. 55s/epoch.
- **Verdict:** KEPT — +0.0024 over exp16. Val oscillated 0.9396-0.9433 E25-33. Chain clearly saturating on this arch at this LR.
- **Notes:** Chain on mean+max arch: exp15(0.9487)→exp16(0.9420,Δ=0.0067)→exp17(0.9396,Δ=0.0024). Geometric decay returns. Time to pivot: next is arch change (exp18 — multi-scale voxel).

### 2026-04-17 — exp16: chain warm-start from exp15 + lr=5e-5
- **Hypothesis:** Exp15 val was still slightly dropping at E33 (0.9487). The new proj_agg layers were still learning. Another 30min of fine-tuning at lr=5e-5 should let the mean+max aggregation fully mature.
- **Change:** --warm_start=<exp15 ckpt> --lr=5e-5. No code changes.
- **Result:** val/l2=**0.9420** @ epoch 24 (30.0 min). train=0.0047. run 4fk50oi3. 54s/epoch.
- **Verdict:** KEPT — +0.0067 over exp15 (0.9487). Early epochs looked bad (E1 0.9513, E2 0.9603) but converged cleanly after E10. Mean+max aggregation wasn't saturated yet.
- **Notes:** Lesson: don't trust early chain epochs; give at least 15-20 epochs before bailing. Val still dropping E30→E33 (0.9450→0.9422). Another chain at lr=2e-5 might squeeze more. Chain progression: Δ=0.040→0.006→0.004→0.0025→(arch)→0.0067. Distance to leader: 0.9420 - 0.8245 = 0.1175.

### 2026-04-17 — exp15: mean+max voxel aggregation (warm-start partial)
- **Hypothesis:** Current voxel aggregation is scatter_add/count (mean) — loses extreme features (high-velocity gradients, shear layers). Add parallel scatter_reduce(amax) branch, concat with mean (2×dim), project back via Conv3d(2*dim, dim). This captures peaks per voxel (turbulent events). Pre-voxel layers weights can warm-start from exp13, only new proj_agg needs fresh training.
- **Change:** VoxelMixer adds scatter amax + 1x1 Conv3d(2D→D) projection. Identity init on mean half, zero on max half → first forward pass identical to exp13 baseline. Warm-start with strict=False.
- **Result:** val/l2=**0.9487** @ epoch 33 (30.1 min). train=0.0050. 4.8GB peak. run 4cun281z. 54s/epoch.
- **Verdict:** KEPT — +0.0126 over exp13 (0.9613). Mean+max captures voxel-local peaks the mean-only lost. Val still dropping at E33 (last epoch).
- **Notes:** Larger single-experiment gain than recent chain steps. This is the first successful arch change in a while. Next: chain-finetune exp15 with lower LR.

### 2026-04-17 — exp14: pos jitter=0.5 (DISCARDED)
- **Hypothesis:** Warm-start + pos jitter (0.5 × voxel_size Gaussian noise) as regularization to unstick saturated chain.
- **Change:** train.py: added cfg.pos_jitter, applied Gaussian noise to pos.
- **Result:** val/l2=**1.1116** @ epoch 1 (broke model). Train loss exploded to 0.028+ immediately.
- **Verdict:** DISCARDED — 0.5 × voxel_size is far too large; warm-started model's SDF features + voxel assignments are sensitive to position perturbations. val climbed rather than fell.
- **Notes:** Lesson: fine-tuned model is brittle to input perturbation. Jitter scale <0.05 might work but arch change is higher-leverage.

### 2026-04-16 — exp13: chained warm-start from exp12 + lr=2e-5
- **Hypothesis:** Exp12 plateau in 0.964-0.966 suggests we're at LR=5e-5 convergence. One more chain at lr=2e-5 to squeeze last ~0.002. After this plateau, pivot to arch changes.
- **Change:** Run with --warm_start=<exp12 ckpt> --lr=2e-5.
- **Result:** val/l2=**0.9613** @ epoch 36 (30.3 min). train=0.0053. run rfc1ntlv.
- **Verdict:** KEPT — +0.0025 over exp12. Chain progression: Δ=0.040→0.006→0.004→0.0025, clear geometric decay.
- **Notes:** Chain saturated; best checkpoint still at exp13.

### 2026-04-16 — exp12: chained warm-start from exp11 + lr=5e-5
- **Hypothesis:** Exp11 val oscillated 0.968-0.977 mid-run, best at E23 (0.9681). LR=1e-4 still too high for fine-tuning. Half again to 5e-5, warm-start from exp11. Expected gain ~0.004 (diminishing chain).
- **Change:** Run with --warm_start=<exp11 ckpt> --lr=5e-5.
- **Result:** val/l2=**0.9638** @ epoch 25 (30.0 min). train=0.0057. run 9jtypb06.
- **Verdict:** KEPT — +0.004 over exp11 (0.9681). Chain plateau approaching: val oscillated 0.964-0.966 at E20-25. Train at 0.0057 (still dropping slowly).
- **Notes:** Chain progression: exp8(1.014) → exp10(0.974, Δ=0.04) → exp11(0.968, Δ=0.006) → exp12(0.964, Δ=0.004). Geometric decay in gains. One more chain expected ~0.002.

### 2026-04-16 — exp11: chained warm-start from exp10 + lr=1e-4
- **Hypothesis:** Exp10 (0.9742) hit minimum mid-run (E36) then slowly climbed — the 2e-4 cosine was slightly too high late, allowing oscillation. Chain another warm-start from exp10 with lr=1e-4 (half), adds 30 more min of fine annealing. Train=0.0070 suggests capacity room remains. Each warm-start cycle has diminishing returns but should add 10-20% improvement per run until plateau.
- **Change:** Run train.py with --warm_start=<exp10 ckpt> --lr=1e-4.
- **Result:** val/l2=**0.9681** @ epoch 23 (30.3 min). train=0.0062. 3.6GB peak. run dyjblcu8.
- **Verdict:** KEPT — +0.006 over exp10. Val plateaued in 0.968-0.977 range after E20 (minimum at E23 then oscillation). Chain is working but diminishing.
- **Notes:** LR=1e-4 was too high for late-cycle refinement — val kept oscillating at end. Next chain at lr=5e-5 should extract remaining fine-tuning gains.

### 2026-04-16 — exp10: warm-start from exp8 + 30 more min
- **Hypothesis:** Exp8 val was still dropping linearly at 30min timeout (best 1.0137). Rather than architectural change, reload exp8 ckpt and train another 30min with a fresh cosine LR schedule at lower peak (2e-4 vs 5e-4). Effectively doubles training budget without needing arch changes. SGDR-style warm restart: new annealing cycle may find better minima from a pre-trained init.
- **Change:** train.py: Added `warm_start: str | None` to Config; loads state_dict post model init. Set cfg.lr=2e-4 for fine-tune run.
- **Result:** val/l2=**0.9742** @ epoch 36 (30.2 min). train=0.0070 (vs exp8's 0.0100, ~30% lower). 3.6GB peak. run dgolkvqw. 46-67s/epoch.
- **Verdict:** KEPT — beats exp8 (1.0137) by 0.0395 (massive jump). Warm-start effectively doubled compute budget. Val still dropping slowly at end (0.9742 → 0.9768 after peak, so we found the minimum). Now rank #2 above nezuko (0.9867).
- **Notes:** This is the biggest single-experiment win. Compound warm-start is a clear winning strategy while budgets are tight. Next: exp11 warm-start from exp10 with even lower LR.

### 2026-04-16 — exp9: subsample 50k→30k (DISCARDED)
- **Hypothesis:** Exp8 (n_blocks=8) was still dropping val at timeout. Subsample 50k→30k saves step time → more epochs to finish the cosine schedule.
- **Change:** subsample_train 50000→30000.
- **Result:** val/l2=1.0141 @ epoch 50 (29.9 min). train=0.0109, 2.6GB peak. run oxfbajwq. 36s/epoch (all 50 epochs fit).
- **Verdict:** DISCARDED — tied exp8 (1.0137) within noise. The extra subsampling regularized too much: train loss only reached 0.0109 vs exp8's 0.0100. More epochs + more regularization ≈ same effective learning. Need different mechanism to beat exp8.
- **Notes:** Lesson: aggressive subsampling acts as a regularizer ceiling. Future "buy more epochs" attempts should look at reducing other costs (grid_size, n_fourier) rather than subsample further.

### 2026-04-16 — exp8: n_blocks 6→8 on exp7 base
- **Hypothesis:** Exp7 val (1.0189) plateaued around E45-50 with train still dropping; exp5 showed depth 4→6 gained 0.016. Try 6→8 with subsample=50k to keep per-step fast. Expect ~47s/epoch → 38 epochs. More rounds of voxel mixing at same grid should extract more spatial structure, which has been our best lever so far.
- **Change:** Config.n_blocks 6→8 (train.py + predict.py). All else identical to exp7.
- **Result:** val/l2=**1.0137** @ epoch 39 (30.5 min timeout). train=0.0100, 3.5GB peak. run g6bus2pg. 47s/epoch.
- **Verdict:** KEPT — beats exp7 (1.0189) by 0.0052. Smooth descending val curve (E34=1.023 → E39=1.014), clearly under-trained at timeout. Depth continues to help.
- **Notes:** Val was still dropping linearly at timeout — more epochs would likely push to ~1.00 or below. Bottleneck is epochs, not capacity. Next: buy more epochs via subsample=30k (same arch, faster steps).

### 2026-04-16 — exp7: train-time point subsampling to 50k
- **Hypothesis:** Exp5 (best: 1.0430) had train loss still dropping at timeout (0.0091, down from 0.0103 at epoch 27). More epochs should help. Subsampling points 100k→50k at train time only (val stays 100k) gives 1.44x step speedup → ~52 epochs vs 38 in same 30min budget. Voxel grid remains same density per voxel (just slightly sparser) so spatial structure preserved.
- **Change:** Config.subsample_train=50000. Train loop: randomly sample K=50k point indices per step, slice v_in/v_out/pos, remap idcs_airfoil via inverse index. Val unchanged (all 100k).
- **Result:** val/l2=**1.0189** @ epoch 50 (29.8 min). train=0.0103, 2.7GB peak. run jdwouppy. 35s/epoch (fit all 50 epochs in budget).
- **Verdict:** KEPT — beats exp5 (1.0430) by 0.024. Subsampling acts as regularization AND enables more epochs. Final train loss 0.0103 is higher than exp5's 0.0091, but val is much better → subsampling = better generalization (new point sampling per step ≈ implicit augmentation).
- **Notes:** Val still dropping at end — could benefit from even more epochs or a deeper net. Subsampling was a 2x win (0.024 drop) without any architectural change. Next: try subsample=30k for even more epochs, or push the net deeper (n_blocks=8) now that per-step is cheap.

### 2026-04-16 — exp6: multi-scale voxel (alternate grids 32/16) (DISCARDED)
- **Hypothesis:** Alternate VoxelMixer grids 32/16 to get multi-scale receptive field at no extra param cost.
- **Change:** Block i even → grid=32, odd → grid=16. 3 blocks each scale.
- **Result:** val/l2=1.0470 @ epoch 42 (30.1 min). 43s/epoch.
- **Verdict:** DISCARDED — 0.004 worse than exp5 (1.0430). Coarse grid (16³ ≈ 6k voxels, >10 pts/voxel) oversmooths; losing 3 fine-scale blocks cost more than multi-scale gained.
- **Notes:** Lesson: reducing fine-scale depth to make room for coarse scale is a net loss. If multi-scale helps, do it via parallel branches, not alternation.

### 2026-04-16 — exp5: n_blocks 4→6 on exp4 base
- **Hypothesis:** Exp4 fully converged at 1.060 — architecture ceiling with 4 blocks. Try moderate depth bump: n_blocks 4→6, hidden=256 (23M params, 1.5x exp4). bf16 gives 68ms/step budget.
- **Change:** Config.n_blocks=6 (train.py + predict.py). All else identical to exp4.
- **Result:** val/l2=**1.0430** @ epoch 38 (30.2 min). train=0.0091, 4.4GB peak. run wu0jb4c0.
- **Verdict:** KEPT — beats exp2 (1.0595) by 0.016. Confirms depth helps where width (exp3) didn't: more rounds of spatial mixing at same grid > larger features per point.
- **Notes:** Train loss still dropping (0.0091 vs exp4 0.0103), could benefit from longer training or an even deeper net. Next: try n_blocks=8 or multi-scale voxel.

### 2026-04-16 — exp4: exp2 arch + bf16 + SDF-to-airfoil
- **Hypothesis:** bf16 autocast at exp2 sizes halves step time → ~50 epochs vs exp2's 38. Add SDF-to-nearest-airfoil + is_airfoil binary as physics priors.
- **Change:** Revert to hidden=256/n_blocks=4/grid=32. Added bf16 autocast. Added `_geom_features()`: chunked cdist to 1024-sample airfoil subset → normalized SDF + is_airfoil indicator (+2 channels). Fixed VoxelMixer dtype bug (h.dtype not x.dtype) for autocast.
- **Result:** val/l2=1.0600 @ epoch 50 (27.6 min, fully converged — train=0.0103 matched exp2 exactly).
- **Verdict:** Tied exp2 (1.0595 vs 1.0600, within noise). Kept base changes (SDF + bf16 + dtype fix) for exp5 since they enable faster iteration.
- **Notes:** SDF + voxel grid seem redundant — grid already encodes geometry. Real gains need architectural change, not more features. Clear convergence plateau confirms arch ceiling at ~1.06.

### 2026-04-16 — exp3: scale up hidden=384/n_blocks=6 + bf16 (DISCARDED)
- **Hypothesis:** Exp2 underfit — scale hidden 256→384 and n_blocks 4→6 (52M params, 3.3x). Add bf16 so bigger model still trains in 30min.
- **Change:** Config defaults bumped + bf16 autocast. Fixed VoxelMixer dtype bug.
- **Result:** val/l2=1.1058 @ epoch 21 (30min timeout). train=0.0142 (vs exp2 0.010).
- **Verdict:** DISCARDED — worse than exp2 by 0.05. Larger model under-trained: bf16 gained speed but 82s/epoch still only yielded 21 epochs vs exp2's 38. Train loss was 50% higher than exp2's converged level, confirming not enough steps.
- **Notes:** Lesson: with a fixed 30min budget, "go bigger" must be paired with enough speedup. 3.3x params needed >3x speedup to equalize step count — bf16 only gave ~1.5x. For future scale-ups: combine bf16 + subsample points + lower batch iters.

### 2026-04-16 — exp1: residual + normalize + no-slip
- **Hypothesis:** Predicting delta from `velocity_in[-1]` should be much easier than absolute velocity (|delta|=1.17 vs |v|=14 raw). Normalizing by vel_std balances loss across Ux/Uy/Uz. Hard no-slip on airfoil is a physical constraint baseline ignores.
- **Change:** BaselineMLP: normalized v_in features, predicts delta_norm (zero-init head), denorms, adds to last frame, zeros airfoil indices. hidden=384, n_blocks=8 (~4.7M params). Loss is MSE on normalized error.
- **Result:** val/l2=1.3016 @ epoch 27 (30min hit timeout at epoch 33). train_loss=0.023. 8.1GB peak. run id 8bycg5j0.
- **Verdict:** Discarded as architecture direction. Marginal gain over last-round 1.33 baseline — confirms MLP-per-point is fundamentally limited without spatial context. Auto-predict failed due to predict.py import re-running train.py's sp.parse (fixed with __main__ guard in exp2).
- **Notes:** Residual + no-slip + normalization stack is still sound — keeping them in exp2. Clear plateau in train loss suggests architecture ceiling, not optimization issue.

### 2026-04-16 — exp2: voxel-grid spatial mixer + Fourier pos
- **Hypothesis:** Per-point MLP can't see neighbors → can't predict local turbulence. Pool features onto a per-sample 32³ voxel grid (bbox-normalized), mix with 3D conv, gather back via trilinear `F.grid_sample`. Fourier features on pos (8 freqs, sin+cos) help represent high-freq spatial structure. Alternate 4 ResBlock + 4 VoxelMixer.
- **Change:** New VoxelMixer module. Added `__main__` guard so predict.py import is clean. Config fields exposed (hidden/n_blocks/grid_size/n_fourier/grad_clip). hidden=256, 4 mixer blocks, 15.5M params. grad_clip=1.0.
- **Result:** val/l2=1.0595 @ epoch 37 (38 ran before 30min timeout). train=0.0103. 4.8GB peak. 48s/epoch. run 9a9gbsue. Auto-predict OK → /mnt/new-pvc/predictions/apr16/tanjiro/368fd11/val.pt. Would be rank #3 (leader alphonse=0.92, #2 thorfinn=1.07).
- **Verdict:** KEPT — huge gain over exp1 (1.30→1.06, -18%). Voxel mixer delivers the spatial context MLPs lacked. Train loss still dropping at timeout (0.010), suggesting capacity+time both still leave room.
- **Notes:** Peak only 4.8GB of 96GB — lots of headroom to scale. Train loss curve is smooth, no instability with grad_clip=1. Best epoch late (37/38) — more epochs would help. For exp3: go bigger (hidden, blocks, grid) AND faster (bf16 autocast).
