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
