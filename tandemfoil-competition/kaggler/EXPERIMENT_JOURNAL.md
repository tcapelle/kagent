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

### 2026-04-27 — Ensemble + diversity-seed runs (iter 13–17)
- **Hypothesis:** A handful of moderately-similar checkpoints (iter 8 + variants) averaged should produce a tighter prediction by cancelling independent errors. Tried: 2-way (iter 8+11), 3-way (iter 8+11+13), weighted (iter 8 ×3 + iter 13 + iter 15).
- **Diversity members trained:**
  - iter 13 (surf_w=30): val 39.56 → test 35.77 — different bias, slightly worse alone.
  - iter 15 (seed=1234, otherwise iter 8 recipe): val 40.43 → test 35.98 — different init/data trajectory, also worse alone.
  - iter 17 (lr=5e-4, seed=42): in progress.
- **Result:** iter 8 alone (test 32.07) remains my top scoring submission. Ensemble scores (b5c6c32, f4e504d, e3a4bdc) still pending in scores.json after >1 hour.
- **Verdict:** Inconclusive on ensembles until scored. Iter 8 alone is the proven champion.
- **Notes:** Setting torch.manual_seed materially affects iter 8-style training: the canonical iter 8 (no explicit seed) reached val 35.88; an explicit seed=1234 reached only 40.43. Suggests iter 8's optimum was somewhat lucky — explains why bigger-model attempts (iter 7, iter 10) couldn't match it. Future runs should run iter 8 with multiple seeds and pick the best, or properly average them.

### 2026-04-27 — 2-checkpoint ensemble iter 8 + iter 11 (iter 12) — submitted
- **Hypothesis:** iter 8 (huber_delta=0.1) and iter 11 (huber_delta=0.05) have the same architecture and Cp recipe but different loss-shape — iter 11's val_single is slightly better (31.19 vs 33.13) while iter 8 is better on geom_rc. Averaging predictions should give a tighter consensus on common errors and partial cancellation of independent ones.
- **Change:** Added predict_ensemble.py — loads multiple checkpoints with their own runtime.yaml, runs each, averages predictions before saving. No retraining.
- **Result:** Predictions saved to commit b5c6c32. Score pending.
- **Verdict:** Pending; will keep if score < 32.07, otherwise revert.
- **Notes:** No diminishing returns risk — pure post-hoc averaging at zero training cost. Worth trying since I had multiple viable checkpoints from the iter 8/11 search.

### 2026-04-27 — huber_delta 0.05 (iter 11) — DISCARDED
- **Hypothesis:** If 1.0→0.1 was a step toward L1, going 0.1→0.05 should sharpen the loss further toward MAE-matching behavior.
- **Change:** Only huber_delta 0.1 → 0.05.
- **Result:** Best val avg_surf_p=37.31 at epoch 78 — worse than iter 8's 35.88. val_single improved (31.19 vs 33.13) but val_geom_rc regressed (51.19 vs 48.25).
- **Verdict:** Discarded. huber_delta=0.1 is the sweet spot.
- **Notes:** Below ~0.1, the L1-like region dominates training and the model learns slightly different solutions per split — generalization on the harder geom_rc track suffers. Useful as an ensemble member though (different solution).

### 2026-04-27 — Bigger model h144-l7 with iter 8 recipe (iter 10) — DISCARDED
- **Hypothesis:** Iter 7 (h160-l8-s48) overfit, but the gentler bump h144-l7-s32 (1.69M params) under iter 8's full recipe (Cp + huber_delta=0.1) might find a better optimum without overfitting.
- **Change:** train.py — n_hidden=128→144, n_layers=6→7; epochs 80→70 to fit in budget.
- **Result:** 64 epochs in 30 min. Best val avg_surf_p=39.22 at epoch 64 — worse than iter 8's 35.88. Each epoch ~28s vs iter 8's 21s, so we trained fewer epochs at higher per-epoch cost.
- **Verdict:** Discarded — h128-l6 is the sweet spot for this 30-min budget.
- **Notes:** Even with iter 8's loss recipe, the bigger model reaches a worse val. May simply be that larger models need more total compute to generalize as well. Future bigger-model attempt would need either dropout or much longer training (which we don't have).

### 2026-04-27 — Velocity Cp normalization too (iter 9) — DISCARDED
- **Hypothesis:** Velocity scales linearly with Re. Extending the Cp recipe to also divide Ux/Uy by exp(log_re - log_re_ref) gives nondimensional velocity targets, mirroring the pressure rescaling that gave iter 6's win.
- **Change:** train.py — added velocity_norm flag, _scale_y/_unscale_pred helpers; recompute stats over rescaled (Ux/re_v, Uy/re_v, p/re_p) targets. predict.py — same logic in inference.
- **Result:** Best val avg_surf_p=37.59 at epoch 64 — worse than iter 8's 35.88 across every split.
- **Verdict:** Discarded — disabled velocity_norm flag (kept code path for future).
- **Notes:** Velocity errors are ~50× smaller than pressure errors anyway. Forcing this rescaling probably destabilizes training without giving the model anything to gain. Pressure has wide regime-dependent dynamic range (>100×) — velocity does not.

### 2026-04-27 — Sharper Huber (delta=0.1) on iter 6 architecture (iter 8) — KEPT — RANK 1
- **Hypothesis:** Iter 6's training surf loss is ~0.005 in normalized space — typical errors are well below huber_delta=1, so the loss is essentially MSE. Lowering huber_delta to 0.1 makes the loss almost-pure-L1 for typical errors, which directly matches the MAE test metric and reduces sensitivity to outlier nodes.
- **Change:** train.py — only huber_delta=1.0 → 0.1. Architecture, Cp norm, surf_weight, epochs all unchanged from iter 6.
- **Result:** 80 epochs in 27.4 min. Best val avg_surf_p=35.88 at epoch 73. **Test avg_surf_p=32.07 — RANK 1 by 6 points.** Per-split test: single=29.93, geom_rc=45.02, geom_cruise=19.69, re_rand=33.65 — beating every other agent on all splits.
- **Verdict:** **Kept.** Loss-shape match to metric was a clean compounding win over Cp-norm (39.16 → 32.07).
- **Notes:** Train surf loss roughly halved (0.005 → 0.002). The L1-like behavior keeps gradients constant at small errors, preventing the model from over-fitting tiny residuals at the expense of harder-to-fit nodes.

### 2026-04-27 — Bigger model h160-l8-s48 + Cp norm (iter 7) — DISCARDED
- **Hypothesis:** Iter 6 still has memory headroom (4.5GB peak). Bumping to h160-l8-s48-mr4 (2.37M params, 2x iter 6) under the same Cp recipe should push further.
- **Change:** train.py — n_hidden=128→160, n_layers=6→8, slice_num=32→48; epochs 80→70 (compensate for slower epochs).
- **Result:** Hit 30-min timeout at epoch 46. Best val avg_surf_p=54.62 at epoch 44 — much worse than iter 6's 42.42. Val started rising at epoch 44; model overfit.
- **Verdict:** Discarded — bigger model overfits faster AND only completes ~half the epochs. Iter 6's 1.16M-param architecture is the sweet spot for this budget.
- **Notes:** With 39s/epoch (vs iter 6's 21s), we lose ~30 epochs of training. The smaller model also clearly fits the geometry well enough — adding capacity without adding regularization was a mistake.

### 2026-04-27 — Cp normalization (Re² scaling on pressure) (iter 6) — KEPT — RANK 1
- **Hypothesis:** Pressure scales as ρU² ∝ Re² in kinematic units. Test sets re_rand and geom_cruise involve out-of-distribution Re — globally normalizing pressure mixes vastly different scales (std 17 to 304 across regimes). Per-sample dividing y_p by exp(2*(log_re - log_re_ref)) decouples scale from shape: model predicts a roughly-O(1) Cp; physical p reconstructed at output via re_factor multiplication.
- **Change:** train.py — added cp_normalize flag, computes y_p/re_factor before global normalization, recomputes pressure-channel stats on rescaled targets (over 200 train samples). predict.py — saves Cp-rescaled stats in `runtime.yaml`, applies the same scaling at inference.
- **Result:** 80 epochs in 27.4 min. Best val avg_surf_p=42.42 at epoch 74. **Test avg_surf_p=39.16 — RANK 1**, beating frieren (42.11) and thorfinn (42.90). Per-split: single=34.79 (best), geom_rc=51.90 (best), geom_cruise=27.54 (3rd), re_rand=42.43 (3rd).
- **Verdict:** **Kept.** Single best improvement so far — closed the entire gap to leaders. Cruise/re_rand were the weakest splits in iter 5 and benefited most from Re-aware scaling.
- **Notes:** With Re² rescaling, residual Cp std variation across samples is ~4x (43–186) vs ~100x raw. Memory still spare (4.5GB peak), so next experiment scales the model up — Iter 7 tries h160-l8 + same Cp recipe.

### 2026-04-27 — vol_subsample 20K + 80 epochs (iter 5) — KEPT
- **Hypothesis:** With iter 4 overfitting, give the model more data variety per batch (vol_subsample 12K→20K) while keeping iter 3's 1.16M-param architecture. Use freed time for longer training (80 epochs).
- **Change:** train.py — vol_subsample=20000, epochs=80, reverted model to h128-l6-s32-mr4. Same Huber loss, channel weights, surf_weight=20.
- **Result:** 80 epochs in 27.3 min. Best val avg_surf_p=55.22 at epoch 78. **Test avg_surf_p=50.87** — rank 6 on leaderboard (was rank 7).
- **Verdict:** **Kept.** Bigger improvement than iter 3→4 attempts. Best splits: single=48.55, geom_rc=61.27, geom_cruise=37.61, re_rand=56.03.
- **Notes:** Cruise (37.61) and re_rand (56.03) are weakest splits — both test Re generalization. Leader frieren is at 25.27/39.88 there → 12 and 16 points behind. Next experiment (iter 6): Cp-style Re² pressure normalization to handle scale variation across regimes.

### 2026-04-27 — Bigger model h192-l8 + subsampling (iter 4) — DISCARDED
- **Hypothesis:** With subsampling freeing up budget (only 17 min used in iter 3), a bigger model (h192-l8, slice_num=64, mlp_ratio=4 → 3.4M params) should reach a better optimum given the same training loop.
- **Change:** Bumped n_hidden=128→192, n_layers=6→8, slice_num=32→64; warmup_epochs=1→2; epochs=60→50.
- **Result:** Best val avg_surf_p=78.40 at epoch 35, but overfit hard afterwards (val ~100+ by epoch 50). Worse than iter 3's 70.96.
- **Verdict:** Discarded — bigger model overfits without more regularization. Reverting to iter 3 architecture.
- **Notes:** Train loss went much lower (0.012 vs iter 3's 0.013) but val drifted up — classic overfitting on the surface points (which are upweighted 20x). Need either more regularization (dropout) or more data variety (bigger vol_subsample) before scaling up the model.

### 2026-04-27 — Surface-aware subsampling (iter 3) — KEPT
- **Hypothesis:** Most compute is wasted on volume nodes (~95% of mesh) while metric is 100% on surface (~3% of mesh). Keep all surface, subsample volume to 12K → ~7x faster per batch + 10x effective surface upweighting → far more training epochs converging on the right thing.
- **Change:** train.py — added `subsample_collate` (keeps surface, samples K=12000 volume points) for train loader; val loaders still use full-mesh `pad_collate` for fair comparison. Bumped epochs to 60.
- **Result:** 60 epochs in 17 min (vs 15 epochs in iter 2). Best val avg_surf_p=70.96 at epoch 55. **Test avg_surf_p=64.12** (single=61.35, geom_rc=79.87, geom_cruise=46.30, re_rand=68.94). Down from baseline 79.12 — about 19% improvement.
- **Verdict:** **Kept.** Single biggest unlock so far; freed ~13 min of budget (used in iter 4).
- **Notes:** Memory dropped to 3.7GB peak (vs 45GB before) — plenty of room for bigger model now. Cosine LR finishes by epoch 60; later epochs flat. Cruise track still cleanest (val_geom_c=53), single_in_dist hardest (val_single=78). Top of leaderboard is ~42; still a ~22 gap.

### 2026-04-27 — h128-l6 huber + AMP + channel-weighted loss (iter 2)
- **Hypothesis:** Smaller, faster-converging Transolver (n_hidden=128, n_layers=6, slice_num=32, mlp_ratio=4) with bf16 AMP, Huber loss, channel weight 2x on pressure, surf_weight=20 should outperform iter 1.
- **Change:** train.py — smaller model (1.16M params), Huber loss with channel weights, lr=1e-3 + 1-epoch warmup + cosine; predict.py wired up via shared model.py.
- **Result:** 15 epochs in 30 min, best val avg_surf_p=105.85 (epoch 14). Test avg_surf_p=94.66 — still worse than baseline 79.12. Trajectory still descending at end of budget.
- **Verdict:** Discarded — undertrained. Best path forward: subsample volume so we get more epochs.
- **Notes:** With AMP and bs=4, ran 15 epochs vs 8 in iter 1. Cruise track always best (val_geom_c=82); single_in_dist worst (val_single=119). Train loss still falling (0.166 → 0.035), so model isn't saturated. Next: surface-aware subsampling (keep all surface, subsample volume to 12K) for 5–10x speedup per batch.

### 2026-04-27 — h192-l8 channel-weighted + warmup (iter 1)
- **Hypothesis:** Bigger Transolver (n_hidden=192, n_layers=8) with channel-weighted MSE loss (3x on pressure), surf_weight=25, LR warmup, gradient clipping should beat baseline 79.12 by exploiting more capacity and better loss alignment.
- **Change:** train.py — bigger model (2.23M params), channel-weighted MSE, surf_weight=25, lr=1e-3 with 2-epoch warmup; predict.py wired up to load Transolver from yaml config; checkpoint selection by avg_surf_p instead of val/loss.
- **Result:** 8 epochs in 30 min (each ~3.8 min), best val avg_surf_p=132.18 (epoch 7). Test avg_surf_p=incomplete-but-likely-worse than baseline.
- **Verdict:** Discarded — too slow per epoch, undertrained.
- **Notes:** Bigger model needs more time per epoch (3.8 min) so only completes 8 epochs in budget. Validation trajectory still trending down. Trying smaller faster-converging model in iter 2 with same loss structure.
