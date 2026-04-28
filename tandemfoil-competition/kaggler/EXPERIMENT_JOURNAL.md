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

### 2026-04-28 — iter34: cycle-9 bs=2 fine-tune SCORE 31.43 RANK 2 (commit 9caa90d)
- **Hypothesis:** Apply bs=2/no-subsample to iter33 with lower LR (1e-5 vs prior 2e-5) since model is more converged.
- **Change:** `--warm_start /tmp/iter33_best.pt --lr 1e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=0.7738** at epoch 1 (cosine still too aggressive). Per-split val: 0.99, 1.14, 0.20, 0.77. **Score: 31.43 — rank 2** (alphonse 24.40, frieren 31.90, askeladd 32.07).
- **Verdict:** kept. Best ep1 means lr=1e-5 was still too high for late-cycle fine-tune. Try lr=5e-6 next.
- **Notes:** Alphonse far ahead (24.40); they likely have a bigger improvement we're missing. Each cycle yields shrinking gains. Next: iter35 = cycle-10 OR ensemble cycle endpoints.

### 2026-04-28 — iter33: cycle-9 HIGH-LR refresh (commit 8bee624)
- **Hypothesis:** Continue cycle pattern from iter32. Same lr=5e-5 25ep recipe.
- **Change:** `--warm_start /tmp/iter32_best.pt --lr 5e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`.
- **Result:** **val/loss=0.7923** at epoch 17. Per-split val: 1.05, 1.11, 0.20, 0.82. **Score: 31.84 — rank 2** (only 0.03 over iter31).
- **Verdict:** kept but marginal. Cycle gain shrinking — high LR may be too disruptive at this convergence level.
- **Notes:** Val loss curve was noisy with many regressions; ep1 was already val ~0.82 (model close to iter32 baseline 0.84). Next: iter34 = bs=2 fine-tune. Then maybe try lower-LR refresh (lr=2e-5) for cycle-10.

### 2026-04-28 — iter32: cycle-8 bs=2 fine-tune NEW BEST (commit 4f844f0)
- **Hypothesis:** Apply bs=2/no-subsample to iter31. Continue cycle pattern.
- **Change:** `--warm_start /tmp/iter31_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=0.8416** at epoch 5. Per-split val: 1.19, 1.20, 0.20, 0.78. NEW BEST locally.
- **Verdict:** kept. Score should drop below 31.
- **Notes:** Cosine peaked at ep5 then drifted up — lr=2e-5 still slightly aggressive at this convergence level. Next: iter33 = cycle-9 high-LR refresh.

### 2026-04-28 — iter31: cycle-8 HIGH-LR refresh 🥈 RANK 2 (commit cf1a207)
- **Hypothesis:** Continue cycle-8 refresh pattern from iter30.
- **Change:** `--warm_start /tmp/iter30_best.pt --lr 5e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`.
- **Result:** **val/loss=0.8754** at epoch 20. Per-split val: 1.34, 1.14, 0.21, 0.81. **Score: 31.87 — RANK 2!** (alphonse 25.14, me 31.87, askeladd 32.07, frieren 32.10).
- **Verdict:** **NEW BEST.** Beat askeladd and frieren. Pattern delivering ~5% val + 1-2 score points per cycle pair.
- **Notes:** Next: iter32 = bs=2 fine-tune; should drop to ~30-31. Alphonse is far at 25.14 — they likely have a different optimization technique we're not using.

### 2026-04-28 — iter30: cycle-7 bs=2 fine-tune SCORE 32.60 rank 4 (commit 556549c)
- **Hypothesis:** Apply bs=2/no-subsample to iter29 (val 0.936). Continue the proven cycle pattern.
- **Change:** `--warm_start /tmp/iter29_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=0.9230** at epoch 8. Per-split val: 1.46, 1.20, 0.21, 0.82. **Score: 32.60 — rank 4** (alphonse 25.53, askeladd 32.07, frieren 32.49 — only 0.11 behind!).
- **Verdict:** kept. ~1.4 score improvement per cycle pair.
- **Notes:** Field tightening — frieren keeps improving too. Next: iter31 = cycle-8 high-LR refresh from iter30.

### 2026-04-28 — iter29: cycle-7 HIGH-LR refresh NEW BEST (commit ee04bf9)
- **Hypothesis:** Continue the high-LR refresh cycling pattern. Each iter at ~3% gain.
- **Change:** `--warm_start /tmp/iter28_best.pt --lr 5e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`.
- **Result:** **val/loss=0.9360** at epoch 16. Per-split val: 1.44, 1.24, 0.22, 0.84. Best yet — sub-1.0 in coarse mode.
- **Verdict:** kept. Score should drop to ~32.
- **Notes:** Cosine still descending at ep25; even more epochs would help. Next: iter30 = bs=2 fine-tune from iter29.

### 2026-04-28 — iter28: cycle-6 bs=2 fine-tune — broke 1.0 val (commit 248dd0e)
- **Hypothesis:** Apply bs=2/no-subsample to iter27 (val 1.019). Following the pattern that paired iter25→iter26.
- **Change:** `--warm_start /tmp/iter27_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=0.9959** at epoch 9 (broke the 1.0 barrier!). Per-split val: 1.64, 1.24, 0.23, 0.88. **Score: 34.04 (rank 4 — alphonse surged to 26.14, askeladd 32.07, frieren 33.12).**
- **Verdict:** kept. Solid improvement but field jumped — alphonse made a big leap. Keep cycling.
- **Notes:** Each cycle-pair (high-LR refresh → bs=2) adds ~3% val and ~1-3 score points. Need to keep the pattern.

### 2026-04-28 — iter27: cycle-6 HIGH-LR refresh 🥈 RANK 2 (commit a40c521)
- **Hypothesis:** Repeat the iter25 trick (high-LR cycle refresh) from iter26's stronger base. Cycle-6 should keep climbing.
- **Change:** `--warm_start /tmp/iter26_best.pt --lr 5e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`.
- **Result:** **val/loss=1.0194** at epoch 23 (26 min). Per-split val: 1.65, 1.31, 0.24, 0.92. **Score: 34.43 — RANK 2!** (askeladd 32.07, me 34.43, frieren 35.01, thorfinn 36.20).
- **Verdict:** **NEW BEST.** High-LR refresh strategy works repeatedly when alternated with bs=2 fine-tune.
- **Notes:** Each cycle pair (high-LR coarse → bs=2 fine-tune) gives ~3-5% val improvement and ~2-3 score points. Next: iter28 = bs=2 fine-tune from iter27 → target rank 1 territory.

### 2026-04-28 — iter26: cycle-5 bs=2 fine-tune NEW BEST (commit d33081e)
- **Hypothesis:** Apply bs=2/no-subsample to iter25 (val 1.106). Standard breakthrough recipe — same step that took iter9→iter10.
- **Change:** `--warm_start /tmp/iter25_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=1.0630** at epoch 5 (25 min). Per-split val: 1.70, 1.33, 0.25, 0.97. **NEW BEST** — 4% improvement over iter25.
- **Verdict:** **NEW BEST.** Per-split looks like score ~34-35 territory.
- **Notes:** Best at epoch 5 then val rose slightly — lr=2e-5 was a touch too aggressive at this convergence level. Could try lr=5e-6 or 1e-5 for a smoother descent in iter27. Or do another high-LR refresh cycle-6.

### 2026-04-28 — iter25: cycle-5 HIGH-LR refresh 🚀 NEW BEST 36.96 (commit eabda69)
- **Hypothesis:** Cycle-3 (iter16) + cycle-4 (iter18) saturated at val ~1.16. Maybe the model is stuck in a local minimum. A higher LR (5e-5 vs prior 1e-5) cycle should "shake" the weights into a different/better basin.
- **Change:** `--warm_start /tmp/iter16_best.pt --lr 5e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`. Same arch.
- **Result:** **val/loss=1.1062** at epoch 21 (26.3 min). Per-split val: 1.83, 1.36, 0.26, 0.97. **Score: 36.96 — NEW BEST** (was 38.60 with iter16). All splits improved.
- **Verdict:** **NEW BEST.** Confirms intuition: when chain saturates at low LR, a higher-LR refresh can escape the basin.
- **Notes:** This is the second big breakthrough after iter9→iter10. Next: iter26 = bs=2 fine-tune from iter25 → expect val ~1.04 and score below 35.

### 2026-04-28 — iter24: slice diversity ensemble iter16+iter23 (commit 8d1254c)
- **Hypothesis:** Mix slice=64 (iter16) and slice=128 (iter23) for architectural ensemble diversity at 0.85/0.15.
- **Change:** `python ensemble.py --sources db4d762 8f852ee --weights 0.85 0.15`.
- **Result:** **39.07** — slightly *worse* than iter16 alone (38.60). iter23 too weak to help.
- **Verdict:** discarded. slice=128 fresh is too undertrained for ensemble use.

### 2026-04-28 — iter23: slice=128 mature — warm iter22 bs=2 no-sub L1 (commit 8f852ee)
- **Hypothesis:** Apply bs=2/no-subsample to slice=128 base. Different physics-attention head capacity provides architectural ensemble diversity.
- **Change:** `--warm_start /tmp/iter22_best.pt --slice_num 128 --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. 217s/ep due to slice=128.
- **Result:** val/loss=1.7891 at epoch 9 (32.6 min — over budget, ep10 wasn't reached). Per-split val: 2.89, 2.28, 0.53, 1.45. Much weaker than iter16 (1.87, 1.46, 0.28, 1.03).
- **Verdict:** kept for diversity ensemble. Standalone too weak to compete.
- **Notes:** slice=128 fresh-train is undertrained. Even at maturity it remains 50% worse than slice=64 chain (which has 4 cycles of training behind it). Ensemble with small weight may help.

### 2026-04-27 — iter21: ensemble iter16 (L1) + iter20 (MSE) — DISCARDED (commit 758d1c3)
- **Hypothesis:** L1 + MSE losses produce decorrelated errors, ensemble at 0.8/0.2 should beat iter16 alone.
- **Change:** `python ensemble.py --sources db4d762 c40c4d8 --weights 0.8 0.2`.
- **Result:** **40.01 avg_surf_p** — *worse* than iter16 alone (38.60). MSE-trained model dilutes L1 quality, ensemble doesn't recover.
- **Verdict:** discarded. MSE didn't add useful diversity (probably because iter20 is much weaker as a single model, the 20% weight pulls predictions toward worse outputs).

### 2026-04-27 — iter22: fresh slice=128 192x6 L1 (commit a215e08)
- **Hypothesis:** Larger slice_num (128 vs 64) gives different physics-attention capacity. Frieren noted slice=128 helps Re-generalization.
- **Change:** `--slice_num 128 --epochs 25 --batch_size 4 --train_subsample 60000 --lr 5e-4 --warmup_epochs 3`. Fresh init.
- **Result:** val/loss=2.1581 at epoch 20 (timeout, 91s/ep was slow). Per-split val: 3.23, 2.79, 0.80, 1.81. Worse than my L1 iter1 (1.997).
- **Verdict:** kept as warm-start for iter23 (slice=128 mature via bs=2 step).
- **Notes:** slice=128 is 45% slower than slice=64. For ensemble diversity, need to mature it.

### 2026-04-27 — iter20: MSE bs=2 fine-tune (commit c40c4d8)
- **Hypothesis:** Apply bs=2/no-subsample to MSE iter19 base. Builds an MSE-mature model whose error patterns differ from L1 chain.
- **Change:** `--warm_start /tmp/iter19_best.pt --loss_type mse --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** val/loss=1.5830 (MSE) at epoch 10. Per-split val (L2² of normalized resid): single=2.30, rc=2.08, cruise=0.57, re_rand=1.39. Weaker than iter16 in absolute terms, but errors should be uncorrelated.
- **Verdict:** kept for ensemble. Will sweep weights for iter21 = iter16 (L1) + iter20 (MSE).
- **Notes:** MSE-trained model up-weights large errors quadratically — should fit outlier samples differently than L1.

### 2026-04-27 — iter19: fresh MSE-loss 192x6 for ensemble diversity (commit fb713c0)
- **Hypothesis:** All my trained models so far use L1; their predictions correlate. A fresh MSE-trained model should produce decorrelated errors (MSE penalizes large errors quadratically, L1 linearly → different fit on outliers).
- **Change:** `--loss_type mse --epochs 25 --batch_size 4 --train_subsample 60000 --lr 5e-4 --warmup_epochs 3`. Fresh init, no warm-start. Same 192x6 arch.
- **Result:** val/loss=**1.9301** (MSE units, not directly comparable to L1) at epoch 24. Per-split val: 2.26, 2.73, 0.94, 1.78 — comparable to my L1 iter1 stage.
- **Verdict:** kept as warm-start base for iter20 (MSE bs=2 step → MSE mature model).
- **Notes:** Per-split has different shape than L1 chain (rc=2.73 vs L1 iter16's 1.46) — confirms different optimization landscape. Next: iter20 = warm iter19 bs=2 no-sub MSE 10ep, then ensemble iter16 (L1) + iter20 (MSE) for diversity.

### 2026-04-27 — iter18: cycle-4 pretrain — saturated (commit 3b06a3d)
- **Hypothesis:** Continue cycling pattern: warm iter16 bs=4 sub60K lr=1e-5 25ep.
- **Change:** `--warm_start /tmp/iter16_best.pt --lr 1e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`.
- **Result:** val/loss=1.1571 at epoch 1; later epochs flat/slight degradation. Per-split essentially identical to iter16.
- **Verdict:** kept but adds nothing new. Cycling has plateaued at val ~1.16.
- **Notes:** Confirmed with iter17 ensemble (39.10) > iter16 alone (38.60): chain checkpoints share trajectories, ensembling them doesn't help. Pivoting strategy: train a genuinely different model (MSE loss instead of L1) for real error decorrelation.

### 2026-04-27 — iter17: 3-way ensemble iter10/iter14/iter16 (commit 21cdcd8)
- **Hypothesis:** Average chain-cycle-best models with iter16 weighted highest. Different LR regimes might add ensemble value.
- **Change:** `python ensemble.py --sources 3924526 d3350b0 db4d762 --weights 0.15 0.25 0.6`.
- **Result:** **39.10** — slightly *worse* than iter16 alone (38.60). Confirms chain checkpoints have correlated errors.
- **Verdict:** discarded. Chain ensembles don't help; need architectural/training-recipe diversity.
- **Notes:** Same lesson as iter4 ensemble (48.89 vs iter3 48.70). The chain produces highly correlated solutions. Will try MSE-loss model in iter19 for real decorrelation.

### 2026-04-27 — iter16: cycle-3 bs=2 fine-tune NEW BEST (commit db4d762)
- **Hypothesis:** Apply bs=2/no-subsample to iter15 (val 1.190). Lower LR (1e-5) since model is heavily converged.
- **Change:** `--warm_start /tmp/iter15_best.pt --lr 1e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=1.1589** at epoch 9 (25 min). Per-split val: 1.87, 1.46, 0.28, 1.03. ~3% improvement over iter14.
- **Verdict:** **NEW BEST.** Cycling pattern still delivers per breakthrough step, just diminishing.
- **Notes:** Score should drop from 39.15 → ~37-38. Next: build 3-way ensemble of cycle-1 (iter10), cycle-2 (iter14), cycle-3 (iter16) — different LR regimes plus different training durations might decorrelate.

### 2026-04-27 — iter15: cycle-3 deeper pretrain — warm iter14 bs=4 sub60K lr=1e-5 25ep (commit 402f7ea)
- **Hypothesis:** Repeat coarse pretraining from iter14. Lower LR (1e-5) since model is highly converged.
- **Change:** `--warm_start /tmp/iter14_best.pt --lr 1e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`.
- **Result:** **val/loss=1.1895** at epoch 12. Per-split val: 1.93, 1.48, 0.30, 1.05. Marginal ~0.6% improvement over iter14 (1.196).
- **Verdict:** kept; cycle saturating fast. Still useful for ensemble.
- **Notes:** Diminishing returns are sharp now. Next: iter16 = bs=2 fine-tune from iter15, then 3-way ensemble of iter10/iter14/iter16.

### 2026-04-27 — iter14: cycle-2 bs=2 fine-tune NEW BEST (commit d3350b0)
- **Hypothesis:** Apply bs=2/no-subsample to cycle-2 base iter13 (val 1.236).
- **Change:** `--warm_start /tmp/iter13_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** **val/loss=1.1963** at epoch 10 (25.1 min). Per-split val: single=1.96, rc=1.49, cruise=0.29, re_rand=1.05. ~3% improvement over iter13.
- **Verdict:** **NEW BEST.** Cycling pattern continues to deliver. All splits dropped ~3-6%.
- **Notes:** Cosine still descending at ep10 — extra epochs would help. Next: iter15 = warm iter14 bs=4 sub60K lr=1e-5 25ep (cycle 3 deeper pretrain).

### 2026-04-27 — iter13: cycle-2 deeper pretrain — warm iter10 bs=4 sub60K lr=2e-5 25ep (commit 499826b)
- **Hypothesis:** Repeat the iter9 trick from iter10's stronger base. Cosine over 25 ep at lr=2e-5 (lower than iter9's 5e-5 since model is more converged).
- **Change:** `--warm_start /tmp/iter10_best.pt --lr 2e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`. Same arch.
- **Result:** **val/loss=1.2364** at epoch 19 (26.3 min). Per-split val: 2.00, 1.54, 0.31, 1.09. ~3.5% improvement over iter10 (1.281).
- **Verdict:** kept. Confirms cycling: bs=2-fine-tune → bs=4-coarse-pretrain → bs=2-fine-tune yields compounding gains.
- **Notes:** Smaller relative gain than iter9→iter10 cycle (12% → 3.5%) — diminishing returns are setting in. Next: iter14 = bs=2 fine-tune from iter13 → expect val ~1.10.

### 2026-04-27 — iter11: chain link from iter10 lr=5e-6 (commit 049e6b5 → preds)
- **Hypothesis:** Continue warm-start chain from iter10 at lr=5e-6 to extract marginal further gains.
- **Change:** `--warm_start /tmp/iter10_best.pt --lr 5e-6 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`.
- **Result:** val/loss=**1.2735** at epoch 1 (chain saturating; later epochs flat at ~1.275). Per-split val: 2.06, 1.57, 0.33, 1.14 — virtually identical to iter10.
- **Verdict:** kept for ensemble averaging. Marginal gain confirms saturation as frieren observed at iter111.
- **Notes:** Mistake — pushed iter12 ensemble (049e6b5) commit while iter11 was running, so iter11's auto-predict overwrote iter12's ensemble outputs. Re-ran iter12 ensemble at fresh commit `95afc8e`. Future: don't make commits between training launch and predict completion.

### 2026-04-27 — iter12: ensemble iter9 0.3 + iter10 0.7 (commit 95afc8e — re-run after overwrite)
- **Hypothesis:** Average iter9 (val 1.334) and iter10 (val 1.281) to add SWA-like averaging diversity. iter10 weighted higher (stronger).
- **Change:** `python ensemble.py --sources a0370c7 3924526 --weights 0.3 0.7`. No training.
- **Result:** TBD on next leaderboard refresh.
- **Verdict:** kept; cheap free win.

### 2026-04-27 — iter10: bs=2 no-subsample on iter9 base 🚀🚀 NEW BEST (commit 3924526)
- **Hypothesis:** Apply frieren's bs=2/no-subsample breakthrough recipe to the deeper pretrained iter9 (val 1.334) instead of iter1 (val 1.997). Should reach val ~1.0-1.1.
- **Change:** `--warm_start /tmp/iter9_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same 192x6 arch.
- **Result:** **val/loss=1.2807** at epoch 10 (25.1 min, 29.1GB). Per-split val: single=2.06, rc=1.59, cruise=0.33, re_rand=1.15. **15% improvement over iter3** (was 1.516); all splits dropped meaningfully (single -15%, rc -17%, cruise -23%, re_rand -12%).
- **Verdict:** **NEW BEST.** Confirms hypothesis: deeper pretraining → bs=2 step gives much bigger gains than pretrain-jump-to-bs2.
- **Notes:** Cosine LR was still descending at ep10; another 5-10 ep would help. Next: iter11 = chain at lr=5e-6 to extract more, then ensemble iter9+iter10 for final submission.

### 2026-04-27 — iter9: deeper coarse pretraining — warm iter3 bs=4 sub60K lr=5e-5 25ep 🚀 (commit a0370c7)
- **Hypothesis:** I jumped to bs=2/no-subsample too early. Frieren chained 4 bs=8/sub=40K links BEFORE the bs=2 breakthrough, reaching val 1.44 first. Continuing iter3's chain at the *coarse* setting (bs=4 sub=60K) at warm-up+cosine over 25 ep should push val below 1.4 by giving more gradient passes through varied subsamples.
- **Change:** `--warm_start /tmp/iter3_best.pt --lr 5e-5 --epochs 25 --warmup_epochs 1 --batch_size 4 --train_subsample 60000`. Same arch.
- **Result:** **val/loss=1.3339** at epoch 25 (26.3 min, 15.3GB). Per-split val: single=2.12, rc=1.64, cruise=0.36, re_rand=1.20. **12% improvement over iter3** — biggest single iter gain since iter1→iter2.
- **Verdict:** kept. Best single-model checkpoint so far.
- **Notes:** Going BACK to coarse training mode (after bs=2 step) actually helped — likely because the volume nodes are seen in fresh random subsets, providing implicit regularization. Cosine LR over 25 ep gave smooth descent. Now iter10 = bs=2 fine-tune from iter9 — predict val ~1.05-1.15 → score below 45.

### 2026-04-27 — iter8: chain link 3 — warm iter3 lr=2e-6 (commit f5695bd)
- **Hypothesis:** Following frieren's iter111 recipe: continue chain at lr=2e-6 for marginal val gain + ensemble averaging.
- **Change:** `--warm_start /tmp/iter3_best.pt --lr 2e-6 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same 192x6.
- **Result:** val/loss=**1.5073** at epoch 9 (25.1 min). Per-split val: single=2.39, rc=1.91, cruise=0.42, re_rand=1.31. ~0.6% gain over iter3 — chain saturating as frieren predicted.
- **Verdict:** kept. Predictions saved at `f5695bd` (after manual `predict.py` re-run because the train.py auto-call OOM'd: training process held 88GB and the `subprocess` spawn only had 5GB to work with).
- **Notes:** OOM bug: `train.py` auto-runs `predict.py` after training without freeing model VRAM first. Should fix: `del model; torch.cuda.empty_cache()` before subprocess call. Workaround for now: re-run predict.py manually after killing the train process.

### 2026-04-27 — iter6: warm iter5 256x8 bs=2 no-subsample (commit 7f2faa7)
- **Hypothesis:** Apply frieren's bs=2 + no-subsample breakthrough to the bigger 256×8 model. Larger capacity + full mesh + low LR should push past 192×6 ceiling.
- **Change:** `--warm_start /tmp/iter5_best.pt --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. 50.7GB peak.
- **Result:** val/loss=**1.8014** at epoch 7 (30.0 min, hit MAX_TIMEOUT). Per-split val: single=2.91, rc=2.29, cruise=0.52, re_rand=1.50. Better than iter5 alone (2.31) but **worse than iter3 (1.52)** — iter5 was undertrained so iter6 inherits that gap.
- **Verdict:** kept for ensemble diversity. Different architecture should add value when blended with iter3/iter4.
- **Notes:** 256×8 at bs=2 no-subsample takes 257s/epoch (vs 152s for 192×6). Only 7 of 10 epochs ran. Could do iter9 to chain at lower LR after a longer pre-training pass. Next: iter7 = 2-way ensemble iter3+iter6.

### 2026-04-27 — iter5: 256x8 slice=64 fresh-train (commit 5b181c5)
- **Hypothesis:** Bigger Transolver (n_hidden=256, n_layers=8, n_head=8) for ensemble diversity. Frieren's best apr27 ckpt (model-9f4m2qmm) used the same shape.
- **Change:** `--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --epochs 25 --batch_size 4 --train_subsample 60000 --lr 5e-4 --warmup_epochs 3`. Fresh init.
- **Result:** **val/loss=2.3131 at epoch 17** (30.3 min, 26.6GB) — hit MAX_TIMEOUT before epoch 25. Per-split val: single=3.61, rc=2.93, cruise=0.82, re_rand=1.90. Still descending but undertrained.
- **Verdict:** kept as warm-start base for iter6 (apply bs=2 no-subsample breakthrough).
- **Notes:** Larger arch (3.94M params) is slower (107s/ep at bs=4 sub=60K vs 63s for 192x6). Predictions at commit `5b181c5` (HEAD when predict ran, not the iter5 placeholder). Next: iter6 = warm iter5 bs=2 no-subsample lr=2e-5 10ep — bigger capacity model after breakthrough recipe.

### 2026-04-27 — iter4: 2-way ensemble iter2 0.4 + iter3 0.6 (commit a00c6ea)
- **Hypothesis:** iter2 (val 1.532) and iter3 (val 1.516) are sequentially-trained chain links — adding their predictions with iter3 weighted higher (since it's stronger) should reduce variance like SWA. Free win, no training.
- **Change:** `python ensemble.py --sources 381bc71 e352f58 --weights 0.4 0.6`. Predictions written to `apr27/tanjiro/a00c6ea/`.
- **Result:** TBD — will appear on leaderboard next refresh. Expected: marginal improvement over iter3 alone.
- **Verdict:** kept. Free improvement to bank before continuing.
- **Notes:** Frieren's apr23 history shows weighted ensemble of chain iterations beats single best by ~0.1-0.2 score points consistently.

### 2026-04-27 — iter3: chain link 2 — warm iter2 lr=5e-6 (commit e352f58)
- **Hypothesis:** Following frieren's iter101 recipe: continue chain at 4× lower LR (5e-6) for further fine-tuning. Should give 1-2% val improvement and add ensemble diversity to iter2.
- **Change:** `--warm_start /tmp/iter2_best.pt --lr 5e-6 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same architecture.
- **Result:** val/loss=**1.5157** at epoch 9 (25.1 min, 29.1GB). Per-split val: single=2.42, rc=1.91, cruise=0.43, re_rand=1.31. Marginal improvement over iter2 (1.532); chain saturating.
- **Verdict:** kept. Marginal gain but valuable for ensemble averaging.
- **Notes:** Predictions written to commit `e352f58` (head moved due to ensemble.py commit). Frieren's experience: each chain link gives 0.01-0.05 val gain; bigger wins come from ensembling multiple chain links + architectural diversity (slice=128). Next: iter4 = ensemble iter2+iter3 (free win), then iter5 = slice=128 diversity model.

### 2026-04-27 — iter2: warm-start iter1 + bs=2 no-subsample (BREAKTHROUGH recipe, commit 381bc71)
- **Hypothesis:** Frieren's apr23 iter93 showed bs=2 + train_subsample=0 (full mesh) is ~30% better than bs=8 sub=40K. Apply to iter1 warm-start at lr=2e-5 cosine over 10 epochs (no warmup since model is pre-trained).
- **Change:** `--warm_start /tmp/iter1_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same 192x6 architecture.
- **Result:** val/loss=**1.5324** at epoch 10 (25.1 min, 29.1GB). Per-split val: single=2.41, rc=1.96, cruise=0.44, re_rand=1.32. **23% improvement over iter1's 1.997**, all splits improved.
- **Verdict:** kept. Confirmed frieren's recipe works. Predictions saved at commit 381bc71.
- **Notes:** Still well above frieren's iter93 val 1.0158 — main reason is my iter1 warm-start (val 1.997) is much weaker than their iter79 (val 1.40 after 4 chain links). To close the gap I should: (a) chain more links at lower LR, (b) longer pre-training before bs=2 step, (c) eventually slice=128 diversity. Next: iter3 = warm iter2 lr=5e-6 10ep (chain link 2).

 + bf16 + p_weight=3 + warmup + bs=4 sub60K (commit 9a14753)
- **Hypothesis:** Reproduce frieren's apr23 recipe: 192x6 Transolver, L1 loss for outlier robustness, bf16, p_weight=3 surface-pressure boost, warmup+cosine, bs=4 with subsample to 60K volume nodes. Pre-train for warm-start chain.
- **Change:** Refactored `train.py`/`predict.py` and added `model.py` with Transolver. New flags: `loss_type`, `p_weight`, `warmup_epochs`, `train_subsample`, `warm_start`, bf16 autocast, grad_clip=1.0.
- **Result:** val/loss=1.9973 at epoch 29 (30.4 min, 15.3GB peak). Per-split val: single=2.54, rc=2.87, cruise=0.70, re_rand=1.87. Cosine still descending at the end → likely undertrained for this config.
- **Verdict:** kept as warm-start base for iter2. Score TBD but expected similar to last apr27 iter1 (~57 surf_p).
- **Notes:** A bit worse than last session's iter1 (val 1.685) — random init variance and `warmup_epochs=3` (last time was different). Real win comes next: bs=2 + no-subsample warm-start (frieren's iter93 went 1.4→1.0 → score 35).
