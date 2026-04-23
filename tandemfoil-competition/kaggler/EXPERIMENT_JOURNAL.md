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

### 2026-04-23 — iter14-l1vol
- **Hypothesis:** L1 loss on volume (not just surf_p) may help robustness to pressure outliers in Part2 (high Re extreme values).
- **Change:** added --l1_vol flag; used with surf_p_weight=60.
- **Result:** val/avg_mae_surf_p=52.20 at epoch 4 (from 52.57).
- **Verdict:** kept — commit 474e6fe.
- **Notes:** Improvement slowing further (-0.37 vs -0.27 iter13). Plateau @ ~52. Next: ensemble OR architectural changes (Fourier features).

### 2026-04-23 — iter13-l1w60
- **Hypothesis:** push L1 surf_p weight higher (60 vs 30).
- **Change:** --surf_p_weight 60.
- **Result:** val/avg_mae_surf_p=52.57 at epoch 4 (from 52.84).
- **Verdict:** kept — diminishing returns (-0.27 vs prev -0.74).
- **Notes:** Plateau approaching. Next: try dropout for regularization or L1 on volume too.

### 2026-04-23 — iter12-l1w30
- **Hypothesis:** higher L1 surf_p weight (30 vs 10) pushes further.
- **Change:** --surf_p_weight 30.
- **Result:** val/avg_mae_surf_p=52.84 at epoch 4 (from 53.58).
- **Verdict:** kept — commit 87d74cb (same train.py, new ckpt).
- **Notes:** Consistent ~0.7 drop per iter. Target askeladd #1 at 50.97.

### 2026-04-23 — iter11-l1-ema995
- **Hypothesis:** L1 loss on surface pressure (direct MAE optimization) + higher EMA decay (0.995) beats L2+surf_p=40.
- **Change:** swapped surf_p loss from MSE to L1; --surf_p_weight 10 (L1 is smaller scale than MSE); --ema_decay 0.995.
- **Result:** val/avg_mae_surf_p=53.58 at epoch 4 (from 54.21).
- **Verdict:** kept — commit 87d74cb
- **Notes:** L1 + EMA continues slow descent. Leaderboard: askeladd 50.97, frieren 53.38. I'm competitive now.

### 2026-04-23 — iter10-ema099
- **Hypothesis:** add EMA (decay=0.99) to smooth weights against overfitting noise. Resume from iter9.
- **Change:** added EMA class; eval runs on EMA weights; save EMA weights as best.pt.
- **Result:** val/avg_mae_surf_p=54.21 at epoch 4 (from 54.91).
- **Verdict:** kept — commit 84d7df9 (will update with iter10 ckpt).
- **Notes:** Small but consistent improvement from EMA. Leaderboard: askeladd #1 at 51.96, frieren #2 at 55.32 — I'm between them now on val (but val != test).

### 2026-04-23 — iter9-surfp40
- **Hypothesis:** push surf_p_weight higher (40, was 20) for even more pressure focus.
- **Change:** --surf_p_weight 40 --lr 1e-5; resumed from iter8.
- **Result:** val/avg_mae_surf_p=54.91 at epoch 4 (down from 55.44).
- **Verdict:** kept — commit 3f7f4ce
- **Notes:** Marginal improvement (0.5). Train loss vol~0.13-0.22 vs val ~1+ — overfitting. Next: EMA for smoother weights.

### 2026-04-23 — iter8-surfp20
- **Hypothesis:** leaderboard ranks by avg surface pressure MAE; rebalance loss to weight surface pressure more (extra 20x) and select best ckpt by avg mae_surf_p instead of val/loss.
- **Change:** added `surf_p_weight=20` term to loss; added `select_by_surf_p=True` flag for ckpt selection. Resumed from iter7 best.
- **Result:** val/avg_mae_surf_p=55.44 at epoch 4 (val/loss=0.99). vs iter4 leaderboard 70.89 and askeladd #1 at 57.48.
- **Verdict:** kept — commit c263408
- **Notes:** Investigated no-slip BC first — found surface velocity NOT zero in this dataset (median 5.98 m/s on surface), so reverted that experiment. Surface here = all boundary nodes incl walls, not just airfoil with no-slip. Per-split mae_surf_p drops big on cruise (from baseline ~55 to 39).

### 2026-04-23 — iter7-lowlr-fullmesh
- **Hypothesis:** full-mesh fine-tune with even lower LR (2e-5, no warmup) avoids overfit of iter6 and pushes further.
- **Change:** --lr 2e-5 --warmup_epochs 0 --epochs 5 (resumed from iter6 1.02).
- **Result:** val/loss=0.9586 at epoch 2 (460s/epoch).
- **Verdict:** kept — commit 61474fd.
- **Notes:** Per-split @ epoch 2: single_in_dist=1.41, rc=1.12, cruise=0.35, re_rand=0.95. Slow improvements remaining; resume chain has diminishing returns. Time to try something new architecturally.

### 2026-04-23 — iter6-fullmesh
- **Hypothesis:** fine-tune subsample-trained model on full mesh (no subsampling) — match eval conditions, learn high-density-region dynamics.
- **Change:** --n_vol_train 0 --batch_size 2 --lr 5e-5 --epochs 10.
- **Result:** val/loss=1.0208 at epoch 1 (459s/epoch, 17.6GB VRAM). Overfit after epoch 1 (val climbed, train kept falling).
- **Verdict:** kept — commit 6a27e83
- **Notes:** Per-split @ epoch 1: single_in_dist=1.48, rc=1.23, cruise=0.39, re_rand=0.98. Big cruise drop (0.73→0.39). Next: even lower LR + 2-3 epochs to push further without overfit.

### 2026-04-23 — iter5-resume-lr1e4
- **Hypothesis:** iter4 still improving at end. Another resume cycle with lower LR=1e-4 should continue the descent.
- **Change:** same as iter4 but --lr 1e-4.
- **Result:** val/loss=1.2522 at epoch 20 (86s/epoch, 22.2GB).
- **Verdict:** kept — commit 9a8189e (same commit as iter4, new ckpt replaces).
- **Notes:** Per-split @ epoch 20: single_in_dist=1.30, rc=1.68, cruise=0.73, re_rand=1.30. Diminishing returns (1.55→1.25, gain 0.30 vs iter4's 0.64).

### 2026-04-23 — iter4-resume
- **Hypothesis:** iter3 was still improving at end. Resume from best.pt with lower LR (2e-4) for another 30 min → effectively 60 min of training, continues the descent.
- **Change:** added --resume_from flag; lr 5e-4→2e-4; warmup 2→1.
- **Result:** val/loss=1.5531 at epoch 21. Down from iter3's 2.19.
- **Verdict:** kept — commit 6cb7d3a
- **Notes:** Per-split @ epoch 21: single_in_dist=1.75, rc=2.09, cruise=0.82, re_rand=1.55. Train loss (vol=0.25, surf=0.10) now much lower than val, some overfitting, but val still improving. Next: another resume cycle at lr=1e-4.

### 2026-04-23 — iter3-bs8-vol20k
- **Hypothesis:** bs=8 + 20K vol pts + 25 epochs + warmup LR beats iter2. More epochs should help since loss still decreasing.
- **Change:** bs=4→8, n_vol=40K→20K, epochs=50→25, added LinearLR warmup 2ep + cosine.
- **Result:** val/loss=2.1905 at epoch 21 (86s/epoch, 22.2GB VRAM). Down from iter2's 3.01.
- **Verdict:** kept — commit b600021
- **Notes:** Per-split @ epoch 21: single_in_dist=2.50, rc=3.07, cruise=1.08, re_rand=2.11. Still improving; LR near zero by end. rc is the biggest remaining loss.

### 2026-04-23 — iter2-subsample
- **Hypothesis:** subsampling volume points (keep all surface, 40K vol/sample) speeds up training 3x with minimal loss in quality. More epochs → lower val/loss.
- **Change:** train.py adds `subsample_volume` called per-batch; bs=4 (was 2); n_vol_train=40000
- **Result:** val/loss=3.0106 at epoch 14 (137s/epoch, 11.1GB VRAM). Down from iter1's 4.98 at epoch 4.
- **Verdict:** kept — commit 016070d
- **Notes:** Per-split @ epoch 14: single_in_dist=3.83, rc=4.05, cruise=1.43, re_rand=2.73. Still monotonically improving at epoch 14 — should try even more aggressive subsampling or fewer layers to allow more epochs.

### 2026-04-23 — iter1-big256 (baseline)
- **Hypothesis:** larger Transolver (n_hidden=256, n_layers=8, slice_num=128) with bf16 autocast + grad checkpointing beats the 128/5 template.
- **Change:** model.py (added grad checkpoint), train.py (bs=2, bf16 autocast, grad clip=1.0, mirror ckpt to pvc)
- **Result:** val/loss=4.98 at epoch 4 (17.6GB VRAM, 7.5 min/epoch → only 4 epochs fit in 30 min)
- **Verdict:** kept — commit 9b456ae (preds already submitted)
- **Notes:** still monotonically improving each epoch; bottleneck is throughput, not capacity. Per-split val: single_in_dist=9.05 (hardest), rc=5.49, cruise=1.80, re_rand=3.59. Next: subsample volume points to fit more epochs.
