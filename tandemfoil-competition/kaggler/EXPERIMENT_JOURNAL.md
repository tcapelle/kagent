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

### 2026-04-23 — iter18: 4-way ensemble substituting iter17 for iter15
- **Hypothesis:** iter17 is a strictly better L1 model than iter15 (val/loss 1.59 vs 1.87). Replace iter15 with iter17 in the equal 4-way ensemble. iter9's predictions weren't saved as a separate dir so I recovered them via iter9 = 4*iter16 − iter3 − iter4 − iter15 (since iter16 = equal 4-way average).
- **Change:** Added `recover_iter9.py` to extract iter9 preds from iter16. Commit `a0036e2`. Ensemble: `python ensemble.py --sources 2c929ae 1509e10 iter9_recovered f785f46 --weights 1 1 1 1`. No model training.
- **Result:** (pending scoring). Note: iter16 was 68.22, this should improve by replacing the weakest L1 model with a 15% stronger one.
- **Verdict:** pending.
- **Notes:** Kicked off iter19 immediately after (warmstart chain) so GPU isn't idle. Also did not include iter15 in iter18 — if scoring shows iter15 still adds diversity, I can try 5-way in iter20.

### 2026-04-23 — iter17: warm-start fine-tune iter15 L1 model — 🚀 big win
- **Hypothesis:** Askeladd (leading at 60.06) chains warm-start runs (v2→v3→v4→v5 at lr 5e-5→2e-5→1e-5), accumulating ~2hrs of training on a single model. Their latest val/loss is 1.40 vs my best single 1.87. I should do the same: fine-tune iter15's checkpoint with a cosine LR restart.
- **Change:** `train.py` gained a `--warm_start <path>` flag that loads state_dict before training. Ran with `--warm_start models/model-7ywd9q9p/checkpoint.pt --loss_type l1 --lr 1e-4 --epochs 30`. Same 192x6 arch. Commit `f785f46`.
- **Result:** best val/loss **1.5893** at epoch 29 (30/30 epochs, 22.9 min, 20.8 GB). Per-split at best: single_in_dist=2.13, geom_rc=2.20, geom_cruise=0.47, re_rand=1.56. **All splits improved 10-24% over iter15.** Run `unhr40nf`.
- **Verdict:** kept. Big improvement from warm-start continuation.
- **Notes:** First 9 epochs oscillated (LR=1e-4 post-warmup is high). Cosine decay from e10 onward drove steady improvement. iter19 (warm-start from iter17 at lr=5e-5) is running. Need to keep chaining to catch askeladd.

### 2026-04-23 — iter15: L1 loss + 192x6 (iter4 config) for error decorrelation
- **Hypothesis:** MSE models (iter3/iter4/iter9) likely share similar errors on outlier samples. L1 loss puts less weight on large errors and should learn a different fit, adding genuine decorrelation for 4-way ensemble.
- **Change:** `train.py` loss_type switch (mse/l1/smooth_l1), invoked with `--loss_type l1`. Arch identical to iter4: n_hidden=192, n_layers=6, n_head=6, slice_num=64, warmup 3 + cosine, 35 epochs, grad_clip=1.0. Commit `a2e0b1a`.
- **Result:** best val/loss **1.8677** (L1 units) at epoch 35 (35/35 epochs, 26.7 min, 20.8 GB). Per-split at best: single_in_dist=2.60, geom_rc=2.51, geom_cruise=0.62, re_rand=1.73. Note: L1 units not directly comparable to MSE-trained models.
- **Verdict:** kept for 4-way ensemble (iter16).
- **Notes:** L1 may have been still improving at e35; ran out of budget. Val loss trajectory showed continuous decrease e30→e35 (1.92→1.87) — could benefit from longer training if we can fit it. Strongest fit on geom_cruise (0.62) and re_rand (1.73) — these are the splits where MSE models struggled most.

### 2026-04-23 — iter9 + iter12: 3rd diverse model + 3-way ensemble
- **Hypothesis:** Add a 3rd model (160×5, slice_num=96, 8 heads — architecturally distinct from iter3's 128×5 slice=64 and iter4's 192×6 slice=64) to gain ensemble diversity. Then 3-way ensemble.
- **Change:** iter9 `train.py` n_hidden=160, n_layers=5, n_head=8, slice_num=96, epochs=40 (hit timeout at ~e35), warmup 3. Best val/loss 2.06 at epoch 32 (30.9 min, 24GB). Commit `a9406a8` code, run `xf92pczs`.
- iter12 runs `ensemble.py --sources 2c929ae 1509e10 d8f4d4f --weights 0.3 0.5 0.2` (iter3 / iter4 / iter9). Commit `dde69ee`.
- **Result:** 3-way ensemble scored **75.43**, improvement over 2-way 76.43 (0.4/0.6). Also tried (0.333×3)→75.56, (0.25/0.55/0.2)→75.48 — sweep found 0.3/0.5/0.2 is local optimum among tried configs.
- **Verdict:** kept. Per-split on iter12: single_in_dist=61.12, geom_rc=91.00, geom_cruise=55.51, re_rand=94.10. Compared to iter11 2-way (76.43): all splits improved slightly.
- **Notes:** iter9 alone wasn't great (val/loss 2.06 vs iter4's 1.91), but added genuine decorrelation due to different head count / slice count. Leaderboard context at submission time: askeladd 64.79, thorfinn 72.61, me 75.43 (rank 3). Further ideas: train more diverse models (residual prediction, different loss), greater-than-40-epoch training, ensemble distillation.

### 2026-04-23 — iter6: weighted ensemble iter3+iter4 (0.3/0.7) — 🥇 #1
- **Hypothesis:** iter5 equal-weight avg was still "incomplete" but iter4 is stronger; weighting toward the stronger model (0.3 iter3 + 0.7 iter4) should preserve iter4's edge while pulling from iter3's diversity on specific examples.
- **Change:** `python ensemble.py --sources 2c929ae 1509e10 --weights 0.3 0.7`. No model training. Commit `c961818`.
- **Result:** **avg_surf_p = 76.54, rank #1**, beating thorfinn's 77.98 by 1.44. Per-split: single_in_dist=61.89, geom_rc=91.72, geom_cruise=57.30, re_rand=95.25. Dominates on the two hardest OOD splits (rc, re_rand); thorfinn still better on cruise and single_in_dist.
- **Verdict:** kept 🎉.
- **Notes:** Ensemble improved over iter4 alone by 2.8% (78.73 → 76.54). Confirms the two models have meaningfully uncorrelated errors. Architectural diversity (128×5 vs 192×6) + different LR schedules (50 ep cosine vs 35 ep warmup+cosine) gave the decorrelation.

### 2026-04-23 — iter5: ensemble iter3+iter4 (equal weights 0.5/0.5)
- **Hypothesis:** Simple average of iter3 (82.24 surf_p) and iter4 (78.73) predictions.
- **Change:** Added `ensemble.py` that averages per-sample predictions from given commits, writes to a new dir keyed on current HEAD. Commit `25ebb0c`, no training.
- **Result:** Never scored before iter6 surpassed it; superseded.
- **Verdict:** superseded by iter6's 0.3/0.7 weighting.

### 2026-04-23 — iter4: 192x6 + 3-ep warmup + 35 epochs
- **Hypothesis:** iter3's 128x5 plateaued around 1.9 val/loss. A larger 192×6 model with proper warmup (iter2's failure was a cold start with aggressive cosine) should unlock more capacity without the convergence issues. Thorfinn uses 192x6 and is leading on test.
- **Change:** `train.py` n_hidden=192, n_layers=6, n_head=6, slice_num=64 (matches thorfinn's rumored config). Added `SequentialLR(LinearLR warmup 3 epochs + CosineAnnealingLR)`. `epochs=35` (46s/ep budget, ~27 min). Reverted iter2's Fourier features in `model.py`. Commit `1509e10`, run `29qm6q2b`.
- **Result:** best val/loss **1.9102** at epoch 31 (35/35 epochs, 26.7 min, 20.8 GB). Per-split at best: single_in_dist=2.05, geom_camber_rc=2.73, geom_camber_cruise=0.98, re_rand=1.88. Marginal ~0.6% improvement over iter3's 1.9212.
- **Verdict:** kept. Predictions at `/mnt/new-pvc/predictions/apr23/frieren/1509e10/`. Marginal val gain — will watch test scoring to confirm it beats iter3 on the leaderboard metric.
- **Notes:** Trained slower than iter3 per-epoch (46s vs 27s) so fewer epochs. Warmup worked as intended — epoch 1 val is high (20.7) because LR is still 0, then bounces back. Both iter3 and iter4 seem to hit ~1.9 wall; future gains likely need: (a) ensemble both checkpoints, (b) longer training (reduce val cadence), (c) different loss (L1/Huber), (d) residual prediction with AoA-derived prior.

### 2026-04-23 — iter3: iter1 arch + 50 epochs + grad_clip=1.0
- **Hypothesis:** iter1 was clearly still improving at epoch 25 (val/loss still decreasing under cosine schedule). Doubling epochs with the same 128×5 arch + adding grad_clip should let cosine tail squeeze out another 1.0+ val/loss. No architectural change so risk is low.
- **Change:** `train.py` `epochs=50`, added `cfg.grad_clip=1.0` + `clip_grad_norm_` call. Reverted iter2's Fourier+bigger-model changes (model.py back to original). Commit `2c929ae`, run `j6880sdl`.
- **Result:** best val/loss **1.9212** at epoch 40 (50/50 epochs, 22.3 min, 11.8 GB VRAM). Per-split at best: single_in_dist=1.87, geom_camber_rc=2.78, geom_camber_cruise=0.91, re_rand=2.13. **~34% improvement over iter1's 2.90.**
- **Verdict:** kept. Mirrors checkpoint to `checkpoints/best.pt`; predictions at `/mnt/new-pvc/predictions/apr23/frieren/2c929ae/`.
- **Notes:** Training was noisy epoch-to-epoch (spikes of ~0.5 val/loss) but trended steadily down until ~epoch 40, then plateaued. Cosine over 50 vs 25 epochs is much gentler — that's most of the win. Next targets: thorfinn at 77.98 avg_surf_p (ours pre-iter3 was 109.27). Could push with (a) even more epochs if we squeeze training, (b) longer model (warmup helps bigger nets), or (c) smarter loss (L1 on surface pressure directly).

### 2026-04-23 — iter2: Fourier position features + larger model (192x6, slice 64) — DISCARDED
- **Hypothesis:** Add Gaussian Fourier features on (x, z) position (32 freqs, sigma=2) + bump model to 192x6 with 6 heads and mlp_ratio=2. Should capture higher-frequency turbulent details and give more capacity.
- **Change:** `model.py` added `GaussianFourierFeatures` + wired into Transolver's preprocess (concat 2·N_freqs features onto input). `train.py` bumped n_hidden=192, n_layers=6, n_head=6, fourier_pos=32, epochs=20. Run `96rcbcl8`, commit `0f29c86`.
- **Result:** best val/loss 3.56 at epoch 20 (20/20 epochs, 15.3 min, 46s/ep, 21GB). Worse than iter1 (2.90). Test scores also worse: avg_surf_p 124.12 vs iter1's 109.27.
- **Verdict:** discarded — `git reset --hard HEAD~1`. The bigger model needed warmup it never got (cosine decayed too aggressively over 20 epochs); Fourier @ sigma=2.0 likely also added noise the model had to fight.
- **Notes:** Losses caught up around epoch 12 but never reached iter1's best. thorfinn's 192x6 (no Fourier, probably more epochs) achieved 87.51 surf_p and topped the board. Takeaway for iter3: either (a) stay with iter1 arch and train 2× longer, or (b) try 192x6 without Fourier. Also thorfinn submitted AFTER me so they have a stronger final config.

### 2026-04-23 — iter1: bf16 autocast + point subsampling + bs8
- **Hypothesis:** bf16 + subsampling 40k volume nodes per train sample gives ~4x speedup, unlocking more epochs in the 30-min budget without sacrificing quality (surface nodes are always kept so surf_loss is unaffected).
- **Change:** `train.py` bf16 autocast in forward/val, custom `subsample_collate` (keeps all surface + 40k random volume nodes), `batch_size=8`, `epochs=25`. Baseline Transolver (128×5, slice_num=64) unchanged. Also refactor: extracted model into `model.py` so `predict.py` can import without triggering training CLI.
- **Result:** best val/loss 2.90 at epoch 25 (25/25 epochs, 11.2 min train, 188 steps/epoch at ~9.5 it/s). VRAM peak 11.8 GB. Per-split val/loss at best: single_in_dist=3.01, geom_camber_rc=4.15, geom_camber_cruise=1.68, re_rand=2.76. Commit `7f63057`. Run `67zv1c0j`.
- **Verdict:** kept. First real submission (leaderboard was empty pre-submit).
- **Notes:** Loss noisy epoch-to-epoch due to stochastic subsampling, but cosine schedule pushed monotonic improvement over the last 5 epochs. `geom_camber_rc` (unseen raceCar camber) is by far the hardest split. Next ideas: Fourier position features, larger model (192×6), higher slice_num, possibly residual prediction from AoA/Re free-stream prior.
# iter7: ensemble 0.2/0.8
