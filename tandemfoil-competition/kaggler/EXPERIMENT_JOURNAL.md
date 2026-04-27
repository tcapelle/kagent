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

### 2026-04-27 — iter15 chain polish at lr=2e-5, slow EMA, n_vol=50k
- **Hypothesis:** Continue the chain at very low LR with slow EMA (0.9999, ~10000-step window) and intermediate subsample (50k for balance between speed and coverage). Squeeze out final residual gain.
- **Change:** No code change; flags `--lr 2e-5 --resume /tmp/iter11_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.9999 --p_weight 20.0 --n_vol_subsample 50000`.
- **Result:** Best epoch 24/30 (timeout at 30 min, 78s/epoch), val/loss=0.852, **avg_surf_p=42.11** — exact tie with frieren's leader number on apr27. Run `gubm0gnz`.
- **Verdict:** Kept (commit e29cc53). −0.19 vs iter11. iter15-single auto-submitted predictions to HEAD 51938c1 — leaderboard scorer picks this up.
- **Ensemble check:** iter15 alone (42.11) BEATS iter15+iter11 (42.19) and iter15+iter11+iter10 (42.20). The chain is so tight that adding members drags down the mean. Single best polish wins.
- **Notes:** Reached the asymptote of this architecture/recipe. Further gains likely require: a) different architecture (256/8/8/96), b) external data, or c) test-time tricks beyond simple ensemble.

### 2026-04-27 — iter14 fresh-from-scratch diverse training (DISCARDED)
- **Hypothesis:** Train a fresh model with same architecture but feature_noise=0.1 (high) and no warm-start to get genuinely uncorrelated predictions. Even if its individual val is poor, ensembling with iter11 might cancel uncorrelated errors.
- **Change:** Same `train.py`; flags `--lr 1e-3 --warmup_epochs 2 --epochs 30 --p_weight 20 --feature_noise 0.1` (no `--resume`). Run `7bpcj870`.
- **Result:** Best epoch 28, val/loss=2.98, **avg_surf_p=95.42** — much worse than iter11's 42.30 (no warm-start budget can match 11 chained iters).
- **iter11+iter14 ensemble val:** 62.00 — way worse than iter11 alone (42.30). The weak member dragged the mean.
- **Verdict:** Discarded. Restored 2-ensemble (iter10+iter11) predictions at HEAD c9609d0.
- **Notes:** Confirms ensemble math: averaging predictions only helps when both models have comparable error magnitudes. A fresh-from-scratch model in 30 min cannot match the warm-start chain. Lesson: ensemble diversity must come from the chain itself (e.g., chain branches with different recipes), not from fresh restarts within budget.

### 2026-04-27 — iter13 ensemble sizing sweep + 2-member submission
- **Hypothesis:** Test how ensemble size affects val avg_surf_p; pick the best subset of chain checkpoints. More members ≠ always better since chain checkpoints are highly correlated.
- **Change:** Wrote `eval_ensemble.py` for fast local val sweeps. Then submitted predictions from the best subset.
- **Result (val avg_surf_p):**
  - iter11 alone: 42.295
  - iter10+iter11 (2): **42.268**
  - iter9+iter10+iter11 (3): 42.346
  - iter8+iter9+iter10+iter11 (4): 42.472
  - iter6+...+iter11 (6): 43.064
- **Verdict:** Submitted 2-member (iter10+iter11) ensemble at HEAD `e8c3d64` (best val: 42.268, only 0.027 below single iter11). Larger ensembles HURT — older chain checkpoints are too similar but worse, dragging the mean down.
- **Notes:** Diminishing returns from chain ensembling. To get real ensemble gain, need uncorrelated members from a different optimization trajectory. iter14 plan: fresh-from-scratch training with same 192/6/6/128 architecture but different seed → genuinely different predictions to average with iter11.

### 2026-04-27 — iter12 4-checkpoint predict-time ensemble
- **Hypothesis:** Single-model chain plateaued at 42.30. Average predictions (in normalized space) over 4 chain checkpoints (iter8 p_weight=20, iter9 +noise, iter10 60k, iter11 80k) — these have slightly different optima and an ensemble should reduce variance enough to break below frieren's 42.11.
- **Change:** `predict.py` (commit 59a7ccf): `--checkpoint` now accepts comma-separated paths; multi-model load + mean predictions per batch. Ran with `--checkpoint /tmp/ens/iter{8,9,10,11}/checkpoint.pt`.
- **Result:** 4-model ensemble predictions saved to `/mnt/new-pvc/predictions/apr27-4/nezuko/59a7ccf/`. Score awaiting leaderboard refresh; per-iter best is 42.30 (iter11).
- **Verdict:** Pending — need leaderboard run to confirm gain.
- **Notes:** All 4 ensemble members are 192/6/6/128 Transolver, identical architecture, all warm-start descendants — predictions are highly correlated, so expected ensemble gain is small (0.3–0.8). If it works, iter13 expands to 5–6 members.

### 2026-04-27 — iter11 push subsample to 80k
- **Hypothesis:** Iter10 (60k) gave −0.69. Push subsample further to 80k for full(er) spatial coverage.
- **Change:** No code change; flags `--lr 3e-5 --resume /tmp/iter10_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.9995 --p_weight 20.0 --feature_noise 0.0 --n_vol_subsample 80000`.
- **Result:** Best epoch 15/30 (only 16 epochs fit at 112s/epoch), val/loss=0.8536, avg_surf_p=**42.30** (val splits: in_dist=0.89, geom_rc=1.33, geom_cruise=0.26, re_rand=0.95). Peak VRAM 58GB. Run `j2rvu5uq`.
- **Verdict:** Kept (commit dd525ed). Essentially flat (−0.04). The chain has hit the architecture/data ceiling. Going wider on subsample costs epochs proportionally.
- **Notes:** Same-architecture warm-start chain has run out of room. Next: ensemble at predict time across multiple chain checkpoints (iter7-iter11), or model-weight soup. Free in compute since training is done.

### 2026-04-27 — iter10 increase n_vol_subsample to 60k, lr=5e-5
- **Hypothesis:** More spatial coverage per training batch (40k → 60k volume nodes) should help the model see finer flow features — feature_noise was a wash, this is a more direct knob. Drop noise back to 0, very low LR (5e-5), slower EMA (0.9995).
- **Change:** No code change (`n_vol_subsample` was already a flag); flags `--lr 5e-5 --resume /tmp/iter9_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.9995 --p_weight 20.0 --feature_noise 0.0 --n_vol_subsample 60000`.
- **Result:** Best epoch 21/30 (timeout — 89s/epoch is slower with 60k vs 66s with 40k), val/loss=0.8503, avg_surf_p=**42.34** (val splits: in_dist=0.88, geom_rc=1.31, geom_cruise=0.26, re_rand=0.96). Peak VRAM 44GB. Run `5xju6s4l`.
- **Verdict:** Kept (commit 981401c). −0.69 vs iter9; **0.23 behind frieren leader (42.11)**. Larger subsample matters more than feature noise.
- **Notes:** geom_cruise dropped (0.27 → 0.26) — finer spatial context particularly helped the dense cruise meshes. Push subsample further (80k) for iter11.

### 2026-04-27 — iter9 add feature_noise=0.05 on AoA+NACA dims
- **Hypothesis:** Geom_rc plateau (1.32 across iter6-8) is generalization-bound, not optimization-bound. Augment by adding per-sample Gaussian noise (std 0.05 in normalized space) to AoA + NACA dims (channels 14-21) to make the model robust to small geometry perturbations and improve OOD camber.
- **Change:** `train.py` (commit 376de75): added `feature_noise` config; injected noise post-normalization, broadcasting per-sample across all N nodes. Ran with `--lr 1e-4 --resume /tmp/iter8_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.999 --p_weight 20.0 --feature_noise 0.05`.
- **Result:** Best epoch 24/30, val/loss=0.8570, avg_surf_p=**43.03** (val splits: in_dist=0.90, geom_rc=1.30, geom_cruise=0.27, re_rand=0.97). Run `oy6v0hkh`.
- **Verdict:** Kept (commit acffa12). Marginal gain (-0.17 vs iter8) — feature noise *did* help geom_rc (1.32 → 1.30) but slightly hurt in_dist (0.87 → 0.90). Net positive but small. Best was epoch 24 not 28 — noise made training noisier so EMA selected an earlier epoch.
- **Notes:** Augmentation is real but ceiling is still close. Next: increase n_vol_subsample to 60k for more spatial context per batch, drop noise (small benefit, slows convergence), keep p_weight=20.

### 2026-04-27 — iter8 p_weight=20, lr=8e-5
- **Hypothesis:** Continue chain at p_weight=20 with slightly lower LR (8e-5) — still gaining at p_weight=15.
- **Change:** No code change; flags `--lr 8e-5 --resume /tmp/iter7_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.999 --p_weight 20.0`.
- **Result:** Best epoch 28/30, val/loss=0.8533, avg_surf_p=**43.20** (val splits: in_dist=0.87, geom_rc=1.32, geom_cruise=0.27, re_rand=0.96). Run `cg7alo9z`.
- **Verdict:** Kept (commit 7191e47). −1.03 vs iter7. Within 1.09 of frieren leader (42.11). val_single_in_dist still dropping (0.87 from 1.03), but geom_camber_rc now firmly stuck at 1.32 across iters 6-8 — pure p_weight scaling cannot crack OOD camber.
- **Notes:** Diminishing returns from p_weight chain. Time to address generalization directly. Plan: input-feature noise on NACA + AoA dims to encourage robustness to small geometry perturbations.

### 2026-04-27 — iter7 p_weight=15
- **Hypothesis:** Continue scaling p_weight (10 → 15) — gains were still present at p_weight=10.
- **Change:** No code change; flags `--lr 1e-4 --resume /tmp/iter6_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.999 --p_weight 15.0`.
- **Result:** Best epoch 27 (not 28 — slight oscillation), val/loss=0.9011, avg_surf_p=**44.23** (val splits: in_dist=1.03, geom_rc=1.33, geom_cruise=0.27, re_rand=0.97). Run `dmqzwdj7`.
- **Verdict:** Kept (commit 1711d04). −1.16 vs iter6, smaller gain than the 5→6→10 step. Now within 2.12 of frieren's 42.11.
- **Notes:** Striking pattern — `val_single_in_dist` keeps dropping rapidly (1.45 → 1.22 → 1.03 → 0.27 across iters 5-7), but `geom_camber_rc` is stuck near 1.33 across iters 6-7 — the OOD camber split has plateaued. Generalization ceiling is on geom_rc. Bigger model or input augmentation (NACA-channel noise) needed to break it.

### 2026-04-27 — iter6 p_weight=10
- **Hypothesis:** iter5 (p_weight=6) gave a bigger gain than the prior pure-chain iters, so continue pushing pressure-channel emphasis to 10 to extract more signal aligned with the leaderboard metric.
- **Change:** No code change; flags `--lr 1e-4 --resume /tmp/iter5_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.999 --p_weight 10.0`.
- **Result:** Best epoch 28/30, val/loss=0.9602, avg_surf_p=**45.39** (val splits: in_dist=1.22, geom_rc=1.34, geom_cruise=0.28, re_rand=1.00). Run `tlyvswlv`.
- **Verdict:** Kept (commit d0e9c45). −1.74 over iter5; now within 3.3 of frieren's leader 42.11. val_single_in_dist dropped sharply (1.45 → 1.22) — model is getting calibrated on the similar-domain split.
- **Notes:** geom_rc still highest (1.34) — out-of-distribution camber generalization is the bottleneck. Worth probing larger model or augmentation next. For now, push p_weight to 15.

### 2026-04-27 — iter5 push p_weight from 3 → 6
- **Hypothesis:** The leaderboard metric is surface pressure MAE. By doubling the per-channel weight on pressure (3 → 6), the optimizer should focus more on what we're scored on, even at small cost to volume MAE we don't care about.
- **Change:** No code change; flags `--lr 1e-4 --resume /tmp/iter4_best.pt --epochs 30 --warmup_epochs 1 --ema_decay 0.999 --p_weight 6.0`.
- **Result:** Best epoch 28/30, val/loss=1.0365, avg_surf_p=**47.13** (val splits: in_dist=1.45, geom_rc=1.37, geom_cruise=0.30, re_rand=1.03). Run `89vpw2ay`.
- **Verdict:** Kept (commit cc7c719). Bigger gain than iter4 (−2.08 vs −1.76) — re-weighting toward the actual scoring metric helps even when warm-started. Now ~5 behind frieren's 42.11.
- **Notes:** Push further next iter (p_weight=10) or try fresh frieren-style 256/8/8/96 architecture as a parallel branch.

### 2026-04-27 — iter4 chain at lr=1e-4
- **Hypothesis:** Continue the warm-start chain at lower LR (1e-4) and slightly slower EMA (0.9995, averaging ~2000 steps) to grind further down the loss curve.
- **Change:** No code change; flags `--lr 1e-4 --resume /tmp/iter3_best.pt --ema_decay 0.9995 --epochs 30 --warmup_epochs 1`.
- **Result:** Best epoch 28/30, val/loss=1.1087, avg_surf_p=**49.21** (val splits: in_dist=1.65, geom_rc=1.41, geom_cruise=0.32, re_rand=1.06). Run `akpldqxb`.
- **Verdict:** Kept (commit 6e7afaf). Smaller gain (50.97 → 49.21, only -1.76) — chain is asymptoting. Per-epoch curve still monotone but very slow descent.
- **Notes:** Diminishing returns from chained fine-tuning. Need more substantive change. Ideas: bump p_weight to push harder on the leaderboard metric (surface p MAE is what's scored), increase n_vol_subsample to expose model to more spatial context, or fresh start with bigger frieren-style 256/8/8/96 model (but loses 4 iters of training).

### 2026-04-27 — iter3 EMA + warm-start chain
- **Hypothesis:** Add EMA(0.999) over weights, validate/checkpoint on EMA shadow. Warm-start from iter2 ckpt at even lower LR (2e-4) to extend the fine-tune chain. Frieren reportedly used EMA — expected to smooth val curve and trim a few points.
- **Change:** `train.py` (commit 97dd212): added `EMA` class; in train loop, update shadow after each optimizer step; before validation, swap shadow→model; save state_dict (which is EMA weights at that moment) on improvement; restore real weights after val. Added `--ema_decay` flag. Ran with `--lr 2e-4 --warmup_epochs 1 --resume /tmp/iter2_best.pt --epochs 30 --ema_decay 0.999`.
- **Result:** Best epoch 28/30, val/loss=1.1557, avg_surf_p=**50.97** (val splits: in_dist=1.78, geom_rc=1.45, geom_cruise=0.34, re_rand=1.06). 66s/epoch. Run `g9ar78uj`.
- **Verdict:** Kept (commit cfb480e). Strict monotone improvement every single epoch (56.24 → 50.97), no oscillation — EMA produces a much smoother loss curve than the noisy iter1/iter2 trajectories. ~9% better than iter2.
- **Notes:** Now ahead of thorfinn's apr27-bis (45.94 was best, mine 50.97 — wait, thorfinn at 45 is still better; I'm 5 behind). Curve still descending so another warm-start chain at lr=1e-4 should help. After that, may need to break out of the local basin via bigger model or new features.

### 2026-04-27 — iter2 warm-start fine-tune
- **Hypothesis:** Iter1 timed out at epoch 28/60 with cosine LR only halfway through — train another 30 epochs warm-started from iter1's ckpt at lower LR (5e-4) so the cosine schedule completes properly within budget.
- **Change:** `train.py` (commit 7b1be64): added `--resume` flag to load a pre-trained state_dict; default `epochs` 60 → 30. Same architecture (192/6/6/128), same loss, same subsampling. Ran with `--lr 5e-4 --warmup_epochs 1 --resume /tmp/iter1_best.pt --epochs 30`.
- **Result:** Best epoch 28/30, val/loss=1.2629, avg_surf_p=**56.30** (val splits: in_dist=1.85, geom_rc=1.60, geom_cruise=0.40, re_rand=1.20). 66s/epoch, 30GB peak. Run `azn9w9u0`.
- **Verdict:** Kept (commit 36453d3). 30% improvement over iter1 (79.11 → 56.30). Now ahead of thorfinn's apr27 (42.90 was the best, mine 56) — still ~30% behind. Loss still decreasing slowly at epoch 28.
- **Notes:** The warm-start pattern works well. Next: another fine-tune chain at even lower LR (1e-4 or 2e-4), or add EMA weights, or push to larger model (256/8/8/96 frieren-style) once we plateau.

### 2026-04-27 — iter1 thorfinn-recipe Transolver 192/6/6/128
- **Hypothesis:** Adopting thorfinn's apr27-bis recipe (192/6/6/128 Transolver, 40k volume node subsampling, L1 loss with p_weight=3 on surface pressure, bf16 AMP, warmup+cosine, grad_clip=1.0, batch_size=8) should beat my apr27-bis score of 79.95 and approach thorfinn's 45.94.
- **Change:** Full rewrite of `train.py` (commit c3dc789):
  - Architecture: n_hidden 128→192, n_layers 5→6, n_head 4→6, slice_num 64→128.
  - Custom collate that keeps all surface nodes and subsamples 40k volume nodes per training sample (val unchanged → full mesh).
  - Loss: MSE → L1 with per-channel weight `[1, 1, 3]` (extra weight on pressure).
  - bf16 autocast for forward + grad clip 1.0 + AdamW.
  - LR: 5e-4 → 1e-3, scheduler: per-epoch cosine → per-step warmup(2 epochs) + cosine over 60 epochs.
  - Best ckpt selected by `avg_surf_p` (matches leaderboard metric), not val/loss.
  - Wrapped main in `if __name__ == "__main__":` so predict.py can `from train import Transolver` cleanly.
  - Mirror best ckpt to PVC + `checkpoints/best.pt`.
  - Fixed predict.py to load Transolver + config.yaml.
- **Result:** Best epoch 28/60 (timeout); val/loss=1.8288, avg_surf_p=79.11 (val splits: in_dist=2.50, geom_rc=2.32, geom_cruise=0.61, re_rand=1.88). Train: 66s/epoch, 30GB peak VRAM. Run `g0e6o8nz`.
- **Verdict:** Kept (commit de6dcc2). Marginal improvement over apr27-bis (79.95 → 79.11) but still ~1.7× behind thorfinn (45.94). Loss was still decreasing fast at epoch 28 — cosine was scheduled for 60 epochs so LR was only halfway decayed when timeout hit.
- **Notes:** Two clear next moves: (a) set `epochs` close to what actually fits (~28-30) so the cosine schedule completes properly within the 30-min budget; (b) warm-start from this checkpoint (de6dcc2) at low LR for final fine-tune. Geom_rc track is the worst (2.32 split-loss vs 0.61 for geom_cruise) — possibly under-represented in the balanced sampler since cruise gets equal weight despite tougher targets being p in raceCar tandem.
