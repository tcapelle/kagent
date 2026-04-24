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

### 2026-04-24 — iter93: bs=2 + no-subsample + warm-start = 🚀🚀 **BREAKTHROUGH** 🥇 #1 at 35.27
- **Hypothesis:** Askeladd uses `batch_size=2` and no volume subsampling. My `batch_size=8` + subsample-to-40K-nodes was cheap but information-lossy. Try bs=2 no-subsample to match their setup. With 1499 samples bs=2 gives 750 steps/epoch (4x my normal 188) so 10 epochs with cosine fits the 30-min budget.
- **Change:** `train.py --warm_start /tmp/iter79_best.pt --slice_num 64 --loss_type l1 --lr 2e-5 --p_weight 3.0 --epochs 10 --batch_size 2 --train_subsample 0`. Commit `3b06e7b` (placeholder). Run `jbidn8vs`.
- **Result:** best val/loss **1.0158** at epoch 10 (25.2 min, 30.2GB). Per-split val: single=1.80, rc=1.17, cruise=0.20, re_rand=0.89 — all MUCH lower than iter79 (1.84, 2.02, 0.37, 1.40). **Scored 35.27 avg_surf_p** — **#1 by 12.66 pts over edward (47.93), 14.68 over askeladd (49.95)**. Per-split test: single=40.26, rc=48.87, cruise=18.50, re_rand=**33.43** (was 73 before). Re-generalization FIXED.
- **Verdict:** KEPT. Single biggest improvement of the competition. All previous ensembles superseded.
- **Notes:** Subsampling was the root cause of my re_rand weakness — dropping 60% of volume nodes left the model unable to learn Re-dependent field structure. bs=2 gives 4x more gradient updates per epoch, and with no subsampling the model sees the full 240K-node grid. Askeladd's edge was entirely this config difference (slice=128 was a minor factor). Starting iter101 = warm-start iter93 with lr=5e-6 to continue chain.

### 2026-04-24 — iter79+iter81: slice=64 chain-3 + 3-way ensemble PB 52.36
- **Hypothesis:** Continue warm-start chain on best slice=64 model (iter37). Then replace iter37 with iter79 in the winning 3-way.
- **Change:** iter79 warm-start iter37 L1 lr=5e-6 p_weight=3 30ep. iter81 ensemble iter29+iter79+iter47 0.5/0.3/0.2.
- **Result:** iter79 val/loss 1.3951. iter81 scored **52.36** (down from 52.82). Per-split: single=41.65, rc=67.94, cruise=26.74, re_rand=73.11.
- **Verdict:** kept iter79 for ensembles. iter81 was my best before bs=2 breakthrough.

### 2026-04-23 — iter63/iter73-77: slice=128 chain-2 + 4-way ensemble experiments
- **Hypothesis:** Further warm-start iter47 at lr=5e-5 gets slice=128 model past val/loss 1.8. Then replace iter47 with iter63 in the best 3-way, or add iter63 as 4th model.
- **Change:** iter63 warm-start iter47 slice=128 L1 p_weight=3 lr=5e-5 30ep. Run `y3i6aw3f` killed at ep27 by timeout (slice=128 takes 66s/ep). Ran predict manually. Ensembles iter73 (3-way 0.5/0.3/0.2 with iter63 swapped for iter47), iter75/77 (4-way variants).
- **Result:** iter63 val/loss **1.72** (down from iter47's 1.80). Per-split val: single=2.09, rc=2.54, cruise=0.64, re_rand=1.60. Ensemble scores pending.
- **Verdict:** pending. iter63 is a better slice=128 model; ensemble should improve.
- **Notes:** Askeladd pushed to 50.52 (ad132a5). Gap now 2.30. Also noticed: slice=128 training exceeds 30-min Linux-timeout; internal MAX_TIMEOUT=30 min is supposed to stop cleanly but apparently epoch 28 passes the check before being reached. Workaround is to manually cp the checkpoint and run predict.py separately.

### 2026-04-23 — iter59: 3-way ensemble iter29+iter37+iter47 → 🥈 new PB 52.82
- **Hypothesis:** Combining diverse-arch models (iter29 slice=64 chain, iter37 slice=64 chain p_weight=10, iter47 slice=128 warm-start chain) in a 3-way weighted average would beat 2-way since each covers different failure modes. Weight iter29 highest (strongest), iter37 medium (complementary), iter47 lowest (weakest val but different arch).
- **Change:** `python ensemble.py --sources 3b76dc7 1c8a6c4 a2ad957 --weights 0.5 0.3 0.2`. No model training. Commit `fe5703f`. Also swept weights (0.4/0.4/0.2, 0.4/0.3/0.3, 0.45/0.35/0.2, 0.6/0.2/0.2).
- **Result:** **52.82 avg_surf_p** — new personal best. Per-split: single=41.52, rc=68.24, cruise=27.72, re_rand=73.81. Askeladd leads at 50.76 (gap 2.06, down from 2.41).
- **Verdict:** kept — best ensemble so far. Weight sweeps (iter65/67/69/61) pending scoring.
- **Notes:** Ensemble sweep from earlier (iter35/39/41/43/45) tops out at 53.38 with 0.85/0.15 (iter29+iter33). Adding iter47 (stronger slice=128) substantially improves over iter33 (undertrained slice=128). Starting iter63 = warm-start iter47 at lr=5e-5 to further strengthen the slice=128 model.

### 2026-04-23 — iter33 + iter47: slice_num=128 architecture for diversity
- **Hypothesis:** Askeladd's config uses slice_num=128 (mine was 64). Their re_rand advantage is 19-25 pts on surf_p vs mine. A fresh slice=128 model (different physics-attention capacity) adds diversity to my warm-chain models.
- **Change:** Added `slice_num` CLI arg to train.py. iter33: `--slice_num 128` fresh 30 epochs L1 p_weight=3 lr=5e-4. iter47: warm-start iter33 with lr=1e-4 for another 30ep. Commits `1bc19d3` / `a2ad957`.
- **Result:** iter33 val/loss 2.24 (undertrained but still helpful in ensembles). iter47 val/loss 1.80 (much improved). iter47 alone scoring pending. iter35 = iter29(0.85) + iter33(0.15) scored 53.38 (beat single iter29). iter53-57 (iter29 + iter47) scored ~53.04-53.28, iter59 (3-way) 52.82 (best).
- **Verdict:** kept. slice=128 diversity is valuable when mixed at 15-25%.
- **Notes:** iter33 was killed by timeout at epoch 28 (fewer epochs/slower iters due to slice=128 compute cost). Still useful in ensemble. Also lesson: iter37 single (p_weight=10) scored 54.49 while iter29 would score similarly; ensemble does all the work.

### 2026-04-23 — iter29/iter31: p_weight=5 + SWA of best 3 checkpoints
- **Hypothesis:** iter27's p_weight=3 seemed to help val. Push further to p_weight=5 and continue warm-start chain. Also do SWA of only the top-3 checkpoints (iter21, iter23, iter27 at weights 1/1/2) since SWA of all 4 (iter25) was worse than iter21 alone.
- **Change:** iter29: `train.py --warm_start /tmp/iter27_best.pt --lr 5e-6 --p_weight 5.0 --epochs 30`. Predictions at `apr23/frieren/3b76dc7/`. iter31: `swa.py --checkpoints iter21 iter23 iter27 --weights 1 1 2`. Run iter29 `543g9e1e`.
- **Result:** iter29 val/loss **1.4171** at epoch 7 — marginal gain over iter27's 1.42. Per-split val (final): single=1.90, rc=2.06, cruise=0.38, re_rand=1.42. iter27/iter29 scoring stuck in queue.
- **Verdict:** iter29 kept (slight val gain). SWA3 pending scoring.
- **Notes:** Askeladd jumped to **52.90 (ffd8b04)** — they're ahead by 2.4 pts. Their weakness is single_in_dist (they're 57 vs my 45); my weakness is re_rand (76 vs their 53). Contemplating fresh slice_num=128 model to match their architecture, since my 64 can't capture their Re-generalization capacity. Also chain is plateauing (1.47→1.45→1.44→1.42 over 4 iterations).

### 2026-04-23 — iter25: SWA of 4 warm-start checkpoints
- **Hypothesis:** Averaging state dicts from iter17/iter19/iter21/iter23 (all 192x6 L1 models on the same warm-start chain) smooths the loss landscape and should generalize better than any single checkpoint.
- **Change:** Added `swa.py` that averages state dicts. Weights 1/1/2/2 (more weight on the better-converged iter21/iter23). Wrote to `models/model-swa4/`, then `predict.py` generated test predictions at `apr23/frieren/66b342c/`. No training.
- **Result:** **55.68 avg_surf_p** — worse than iter21's 55.32 (best single). Per-split: single=45.70, rc=72.48, cruise=28.15, re_rand=76.38.
- **Verdict:** discarded — simple SWA averaging all 4 didn't help. Earlier (weaker) checkpoints pulled the mean toward their inferior region.
- **Notes:** Could try SWA of only the best 2 (iter21+iter23) or just skip — chain is plateauing anyway. iter27 (p_weight=3 warm-start) is running next.

### 2026-04-23 — iter21/iter23: continuing warm-start chain (lr=2e-5, lr=1e-5)
- **Hypothesis:** Askeladd chains 4 warm-starts at decreasing LRs. Continue my chain: iter21 = warm iter19 at 2e-5, iter23 = warm iter21 at 1e-5 (matching askeladd's endpoint).
- **Change:** `train.py --warm_start <path> --loss_type l1 --lr 2e-5|1e-5 --epochs 30`. iter21 commit `e7bea18`, run `yxj4y1an`. iter23 reused the same commit (no intervening git commit) → its predictions overwrote iter21's at `apr23/frieren/e7bea18/`; run `xrkzylm8`.
- **Result:** iter21 val/loss 1.4461 at e19 → scored **55.32 avg_surf_p, 🥇 #1** (askeladd 56.07). iter23 val/loss 1.4351 at e12, predictions overwrote iter21 at `e7bea18`. Per-split iter21 test surf_p: single=44.81, rc=72.42, cruise=27.66, re_rand=76.40.
- **Verdict:** kept both for SWA downstream. iter21 currently sits at #1.
- **Notes:** Diminishing returns — val/loss went 1.47→1.45→1.44 across iter19/21/23. Chain has largely plateaued. Weakness vs askeladd is **re_rand** (76 vs 57, 19-point gap) — their architecture (slice_num=128 vs my 64) or more training must generalize better in Re. Should have used separate commits per run to keep iter21 and iter23 predictions distinct.

### 2026-04-23 — iter19: 2nd warm-start chain → 🥇 #1 on the leaderboard
- **Hypothesis:** Continue the warm-start chain — fine-tune iter17's checkpoint at yet-lower LR (5e-5, half of iter17's 1e-4). Each chain link accumulates more training time on the same model, mimicking askeladd's v2→v5 chain.
- **Change:** `train.py --warm_start /tmp/iter17_best.pt --loss_type l1 --lr 5e-5 --epochs 30`. Commit `bcbcb52` (journal commit that auto-triggered predictions). Run `ogycayte`.
- **Result:** best val/loss **1.4741** at epoch 19 (30/30 epochs, 22.9 min, 20.8 GB). Per-split: single_in_dist=1.94, geom_rc=2.07, geom_cruise=0.42, re_rand=1.46. **Scored 56.35 avg_surf_p — rank 1** (askeladd 57.48, +1.13 ahead). Per-split test surf_p: single=47.42, rc=72.88, cruise=29.42, re_rand=75.69.
- **Verdict:** kept. Took #1 spot.
- **Notes:** Val loss converged around e19-21 and stayed flat. LR 5e-5 post-warmup hit the sweet spot. My weakness is still re_rand (75.69 vs askeladd's 58.04) — 17.6 pt gap. Iter21 started immediately (warm-start from iter19 at lr=2e-5). Also iter18 (4-way ensemble) underperformed at 65.92 vs iter17 single's 59.14 — weaker MSE models dilute iter17's good predictions. Lesson: only ensemble strong models.

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
