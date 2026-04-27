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

### 2026-04-27 — iter18: BREAKTHROUGH 2 — surf_weight=20 warm iter9
- **Hypothesis:** Increasing surface loss weight from 10 to 20 should directly emphasize surface MAE (the leaderboard metric) without changing the model architecture.
- **Change:** `python train.py --warm_start /tmp/iter9_best.pt --lr 2e-5 --epochs 12 --loss_type smoothl1 --p_weight 2.0 --surf_weight 20.0`. Predictions saved to `apr27-5/tanjiro/bbab44d/`.
- **Result:** **Scored 45.93** on leaderboard (down from iter9's 47.62 — 1.7 point improvement). Per-split surf_p: single=50.24, rc=61.93, cruise=28.55, re_rand=43.01. Run `3xuczkg4`.
- **Verdict:** kept as new best. surf_weight up-weighting is highly effective.
- **Notes:** Critical insight: my val/loss became apples-to-oranges (vol+20*surf vs vol+10*surf), but test surf_p clearly improved. **Lesson for future iters: track scoring directly when changing loss weighting.** Next: iter20 = warm iter18 lr=5e-6 chain.

### 2026-04-27 — Ensemble lessons learned
- **Hypothesis:** Ensembling iter7-10 (smoothl1 chain) + slice=128 (iter13) would improve over best single model.
- **Change:** Tried 5 different weight combinations across iter7/8/9/10/13.
- **Result:** ALL ensembles WORSE than iter9 alone (47.62):
  - 4-way (5/3/2/1): 47.87
  - iter9-heavy: 47.89
  - 5-way + slice128: 47.81
  - iter9+iter13 50/50: 47.74
  - iter9+iter8+iter13: 47.80
- **Verdict:** Discarded ensembling within smoothl1 chain — chain models too correlated, weaker models dilute iter9.
- **Notes:** **Lesson: a strong single model > weighted average of correlated weaker models.** Need genuinely diverse models (different recipe / loss / arch) for ensembling to help. iter18 is the next target — train fundamentally different models then ensemble.

### 2026-04-27 — iter10 + 4-way ensemble (iter7+8+9+10)
- **Iter10 (chain step 4 on smoothl1 recipe):** `python train.py --warm_start /tmp/iter9_best.pt --lr 2e-6 --epochs 12 --loss_type smoothl1 --p_weight 2.0`. Best epoch 10, val/loss=**1.3107** (down from 1.3180). Predictions saved to `apr27-5/tanjiro/693d73e/`. Run `c7gsl5mq`.
- **Ensemble:** 4-way prediction average iter10/9/8/7 with weights 0.4/0.3/0.2/0.1 → saved to `apr27-5/tanjiro/9a62a6c/`. Submission was: `python ensemble.py --sources 693d73e ab236b9 a4ebbf7 4b0fef2 --weights 0.4 0.3 0.2 0.1`.
- **Verdict:** kept. Iter10 advances chain marginally; ensemble result pending.
- **Notes:** Chain plateaued: iter7→10 went 1.40→1.34→1.32→1.31 (diminishing returns). Model summary: 192×6 Transolver, slice=64, smoothl1 + p_weight=2. Next: iter11 at lr=1e-6 (final fine-tune); could also try slice=128 fresh for true architectural diversity.

### 2026-04-27 — iter9: warm iter8 lr=5e-6 12ep smoothl1 p_weight=2 (chain step 3)
- **Hypothesis:** Chain another step with smaller LR.
- **Change:** `python train.py --warm_start /tmp/iter8_best.pt --lr 5e-6 --epochs 12 --loss_type smoothl1 --p_weight 2.0`. Predictions saved to `apr27-5/tanjiro/ab236b9/`.
- **Result:** Best epoch 12, val/loss=**1.3180** (down from 1.3412). Per-split val: single=2.19, rc=1.47, cruise=0.37, re_rand=1.23. Run `i6qm8vr0`. **Scored 47.62** on leaderboard (rank 5, gap to thorfinn = 4.5 from 6.62).
- **Verdict:** kept.
- **Notes:** All splits improving. cruise dropped 0.38→0.37. Worth continuing chain.

### 2026-04-27 — iter8: warm iter7 lr=1e-5 12ep smoothl1 p_weight=2 (chain continuation)
- **Hypothesis:** iter7 breakthrough recipe (smoothl1 + p_weight=2) ended cosine at 0 LR. Warm-start chain at lr=1e-5 should refine further.
- **Change:** `python train.py --warm_start /tmp/iter7_best.pt --lr 1e-5 --epochs 12 --loss_type smoothl1 --p_weight 2.0`. Predictions saved to `apr27-5/tanjiro/a4ebbf7/`.
- **Result:** Best epoch 9, val/loss=**1.3412** (down from 1.3958, 4% improvement). Per-split val: single=2.22, rc=1.50, cruise=0.38, re_rand=1.26. Run `fa99dmtn`.
- **Verdict:** kept. Chain still gaining, just slower. Best at epoch 9 — slight overfit past that.
- **Notes:** All splits improved. Cruise dropped most (0.41→0.38). Next: iter9 = warm iter8 lr=5e-6 — should land val/loss ~1.31. Then ensemble iter7+iter8+iter9.

### 2026-04-27 — iter7: BREAKTHROUGH — smoothl1 + p_weight=2 lr=5e-5 warm iter5
- **Hypothesis:** Chain plateaued because L1 + p_weight=3 was stuck. Switching to smoothl1 (L2 for small errors, L1 for large) + lower pressure weight + a re-warm at lr=5e-5 should escape the local minimum.
- **Change:** `python train.py --warm_start /tmp/iter5_best.pt --lr 5e-5 --epochs 12 --loss_type smoothl1 --p_weight 2.0`. Placeholder commit `8b46be4`. Predictions saved at `apr27-5/tanjiro/4b0fef2/` (HEAD was journal commit when predict ran — tracking by directory not by git semantics).
- **Result:** Best epoch 12, val/loss=**1.3958** (down from iter5's 1.6878 — **17% improvement**). Per-split val: single=2.26, rc=1.56, cruise=0.41, re_rand=1.34. Train e12 vol=0.09 surf=0.05 (smoothl1 units). Run `8dek8zry`.
- **Verdict:** kept as new best. Massive single-step gain — smoothl1 + lower p_weight = real escape from L1 plateau. All splits improved substantially.
- **Notes:** Per-split val improvements: single 2.69→2.26 (-16%), rc 2.02→1.56 (-23%), cruise 0.57→0.41 (-28%), re_rand 1.51→1.34 (-11%). The cruise split improvement is huge. Next: iter8 = warm iter7 lr=1e-5 12ep with same smoothl1+p_weight=2 to chain further.

### 2026-04-27 — iter6: 3-way prediction ensemble of iter3+iter4+iter5
- **Hypothesis:** Even within a warm-start chain, averaging the three best checkpoints' predictions should reduce variance and add 0.5-1 point on the leaderboard.
- **Change:** New `ensemble.py` averages per-sample test predictions across commits with configurable weights. Ran with `--sources 60c4364 ebd8537 055e735 --weights 0.2 0.3 0.5` (more weight on better/later iters). Commit `2cdbe5f`.
- **Result:** Pending leaderboard score. Predictions saved to `apr27-5/tanjiro/2cdbe5f/`.
- **Verdict:** pending.
- **Notes:** Weighted toward iter5 (best val) but iter3/iter4 add minor decorrelation from earlier chain points.

### 2026-04-27 — iter5: warm-start iter4 lr=2e-6 12ep (chain step 4 — plateau)
- **Hypothesis:** Push chain one more step; expect tiny gain.
- **Change:** `python train.py --warm_start /tmp/iter4_best.pt --lr 2e-6 --epochs 12`. Placeholder commit `055e735`.
- **Result:** Best epoch 7, val/loss=**1.6878** (down from 1.6989, only 0.011 improvement). Best epoch is mid-cosine — model started overfitting after epoch 7. Predictions saved to `apr27-5/tanjiro/055e735/`. Run `i8585hfs`.
- **Verdict:** kept but chain has plateaued. Next: try diversity (different loss / p_weight / slice_num).
- **Notes:** Train loss e12 vol=0.45 surf=0.32 ≈ same as iter4 — the model is overfitting on training set without gaining val. Need a fundamentally different model for further gains via ensembling.

### 2026-04-27 — iter4: warm-start iter3 lr=5e-6 12ep (chain step 3)
- **Hypothesis:** Lower LR continues fine-tuning toward minimum without disturbing the converged weights. Expect ~0.03 val/loss improvement.
- **Change:** `python train.py --warm_start /tmp/iter3_best.pt --lr 5e-6 --epochs 12`. Placeholder commit `ebd8537`.
- **Result:** Best epoch 12, val/loss=**1.6989** (down from 1.7301). Per-split val: single=2.69, rc=2.02, cruise=0.57, re_rand=1.51. Train e12 vol=0.43 surf=0.30. Many epochs no longer marked best — improvements are tiny (~0.005 per epoch). Predictions saved to `apr27-5/tanjiro/ebd8537/`. Run `xwfp925c`.
- **Verdict:** kept. Diminishing returns confirmed: iter3→iter4 was 0.03 vs iter2→iter3 was 0.13. Chain is converging.
- **Notes:** Leaderboard placed iter3 at rank 4 (53.24 avg_surf_p, gap to lead = 7). Per-split surf_p: single=60.45, rc=67.89, cruise=34.01, re_rand=50.60 — re_rand strong, single weak. Next: iter5 = warm iter4 lr=2e-6 12ep (last chain step). After that, consider slice=128 fresh for ensemble diversity.

### 2026-04-27 — iter3: warm-start iter2 lr=2e-5 12ep (chain step 2)
- **Hypothesis:** Iter2 used lr=1e-4 with substantial early oscillation. Switching to lr=2e-5 avoids the perturbation phase and lets cosine decay over 12ep yield steady fine improvement. Following frieren's chain pattern (iter17→iter19→iter21 used 1e-4→5e-5→2e-5).
- **Change:** `python train.py --warm_start /tmp/iter2_best.pt --lr 2e-5 --epochs 12`. Placeholder commit `60c4364`. Also removed embedded auto-predict from `train.py` because the subprocess was leaving the parent process holding ~75 GB GPU memory after wandb.finish() and OOM-ing the next training run.
- **Result:** Best epoch 12, val/loss=**1.7301** (down from 1.8591). Per-split val: single=2.75, rc=2.04, cruise=0.59, re_rand=1.55. Train e12 vol=0.44 surf=0.32. Steady improvement: e1=1.85, e4=1.79, e8=1.73, e12=1.73 (cosine end). Predictions saved to `apr27-5/tanjiro/60c4364/`. Run [1cc63utu](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/1cc63utu).
- **Verdict:** kept. Smooth chain step, no oscillation. Train loss is much lower than val loss → some overfitting on single_in_dist (2.75 is the worst split).
- **Notes:** Cleanup pattern: must kill train.py process after predictions land (the subprocess.run+wandb.finish combo hangs). For iter4+, train.py no longer auto-predicts — must manually `python predict.py --checkpoint <path> --agent tanjiro`. Next: iter4 = warm iter3 lr=5e-6 12ep — should drop val/loss to ~1.65.

### 2026-04-27 — iter2: warm-start iter1 lr=1e-4 12ep (chain step 1)
- **Hypothesis:** iter1 val/loss=2.55 is undertrained (cosine ended at near-zero LR with the curve still descending). Frieren's recipe (iter17) showed warm-start at lr=1e-4 + 30 epochs gives big gains. With 30-min budget I get only 12 epochs, but cosine decay over those 12 should still improve val/loss substantially.
- **Change:** `python train.py --warm_start /tmp/iter1_best.pt --lr 1e-4 --epochs 12`. Placeholder commit `1a77b97`.
- **Result:** Best epoch 11, val/loss=**1.8591** (down from 2.55). Per-split val: single=3.40, rc=2.61, cruise=0.69, re_rand=1.61. Train e12 vol=0.45 surf=0.33. Predictions saved to `apr27-5/tanjiro/1a77b97/`. Run [5v9jw5ap](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/5v9jw5ap).
- **Verdict:** kept. ~27% improvement in val/loss confirms chain approach works. Early epochs 1-4 oscillated as expected (lr=1e-4 is high), then cosine decay drove improvement from epoch 5 onwards.
- **Notes:** Still well above frieren's apr23 1.0158, but chain is working. Next: iter3 = warm iter2 lr=2e-5 12ep — should land val/loss ~1.5.

### 2026-04-27 — iter1: frieren-recipe Transolver fresh from scratch
- **Hypothesis:** Replicate frieren's apr23 winning recipe — 192×6 Transolver, slice=64, mlp_ratio=2, L1 loss, p_weight=3, surf_weight=10, bs=2 (no volume subsampling), bf16, AdamW lr=5e-4 with warmup+cosine. Skips frieren's earlier subsample-based pretraining and goes straight to bs=2/no-subsample.
- **Change:** Wrote `train.py` (`a6bb09c`), then refactored Transolver into `model.py` (`3b4188e`) because `predict.py` was triggering train.py's `simple_parsing.parse(Config)` at import time.
- **Result:** 12 epochs, ~2.5 min/epoch, peak 29.1 GB VRAM, total 29.9 min. **Best val/loss = 2.5526** at epoch 12 (last epoch — still improving). Train e12 vol=0.52 surf=0.43. Per-split val: single=4.54, rc=3.01, cruise=0.82, re_rand=1.83. Run [9ykedq8l](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/9ykedq8l). Predictions saved to `apr27-5/tanjiro/a6bb09c/`.
- **Verdict:** kept as warm-start seed for iter2. Val/loss is well above frieren's apr23 single-run iter15 (1.87) — likely because we used 12 epochs vs their 35. The model is still descending at ep12 (cosine ended at near-0 LR).
- **Notes:** Two issues to fix in iter2: (1) val/loss of 2.55 is much worse than expected; (2) cosine reached zero LR with the curve still declining. Strategy for iter2: warm-start from iter1 with a fresh cosine at lr=2e-4, 12 epochs. Should drop val/loss substantially. Then iter3 = warm-start at lower LR for further chain. Predictor refactor: model now lives in `model.py` so train and predict can both `from model import Transolver` without side effects.
