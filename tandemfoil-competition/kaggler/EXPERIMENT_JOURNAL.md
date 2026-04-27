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
