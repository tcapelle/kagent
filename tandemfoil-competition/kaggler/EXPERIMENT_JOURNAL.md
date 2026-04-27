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

### 2026-04-27 — iter4: chain finetune iter3 with p_weight bumped 3→6
- **Hypothesis:** rc split is the bottleneck (surf_p≈74) and pressure is the leaderboard metric, so doubling the pressure-channel weight in the L1 loss should bias gradients more toward pressure error and squeeze a few more points from the surface metric.
- **Change:** Resume from iter3 ckpt; same shape as iter3 (bs=2, full mesh, ep=10, lr=5e-6, no warmup) plus `--p_weight 6`.
- **Result:** **surf_p 52.74 → 52.05** (~1.3% gain). val/loss is not comparable (loss formula scaled by p_weight). Per-split surf_p: single≈54, rc≈74, cruise≈30, re_rand≈50. W&B `12v6mgmg`. Predictions submitted.
- **Verdict:** Kept — small but monotone. The bottleneck is still the rc (unseen camber) split.
- **Notes:** Trade-off seems neutral on Ux/Uy (no big regression observed). Next: lower LR + p_weight=6 chain (iter5), then if plateau, try larger model or augmentation.

### 2026-04-27 — iter3: chain finetune iter2 (bs=2, full mesh, lr=5e-6)
- **Hypothesis:** Drop LR another 4x (2e-5 → 5e-6) for fine polishing — frieren's chain pattern showed each LR cut yielded a few percent.
- **Change:** Resume from iter2 ckpt; --batch_size 2 --n_sub 0 --epochs 10 --lr 5e-6 --warmup_epochs 0.
- **Result:** 10 ep in 24.9 min. val/loss 2.27 → **2.23**, **surf_p 53.80 → 52.74** (~2% gain). Per-split surf_p: single_in_dist≈55, rc≈74, cruise≈30, re_rand≈53. W&B `0ttf86dz`. Predictions submitted.
- **Verdict:** Kept — small but monotone improvement. Diminishing returns; rc is the hardest split (75) — doesn't generalize as well to unseen front-foil camber.
- **Notes:** Loss still trending down at epoch 10 — could chain again at lr=1e-6, but expecting <1% gain. Better next move: stronger pressure weighting (pw=5-7) or bigger model.

### 2026-04-27 — iter2: chain finetune iter1 (bs=2, full mesh, lr=2e-5)
- **Hypothesis:** frieren's apr23 unlock was switching from bs=8/sub30k to bs=2/full-mesh at lr=2e-5 — surface pressure benefits from seeing full meshes and very low LR. Resume from iter1 ckpt and run 10 epochs.
- **Change:** new training run, same model/loss, --batch_size 2 --n_sub 0 --epochs 10 --lr 2e-5 --warmup_epochs 0, resumed from iter1 PVC checkpoint.
- **Result:** 10 epochs in 24.9 min, **val/loss 3.82 → 2.27**, **avg_surf_p 91.88 → 53.80** (best on last epoch — still descending). Per-split surf_p (best epoch 10): single_in_dist≈54, rc≈75, cruise≈30, re_rand≈55. W&B run `v75f4wqq`. Predictions submitted to apr27-5/thorfinn/8d436c1.
- **Verdict:** Kept — 42% relative improvement on the primary metric. Now within striking distance of apr27 leader (frieren 42.11). Best.pt updated.
- **Notes:** Memory peaked at 29 GB at bs=2 full mesh — plenty of headroom. Loss still trending down at last epoch, so iter3 should chain again at even lower LR (5e-6 → 1e-6) for fine polishing.

### 2026-04-27 — iter1: Transolver 192x6 + bf16 + L1 + 30k sub + pw3
- **Hypothesis:** A proven recipe from frieren's apr23 W&B runs — Transolver 192h x 6L slice_num=64, bf16 autocast, L1 loss, 30k node subsample (preserving all surface), pressure-channel weight 3x, sw=10, AdamW lr=5e-4 with 3-epoch warmup + cosine over 60 ep — should land near val/loss ~2 in one shot from scratch.
- **Change:** train.py rewritten with the full recipe (model unchanged from baseline shape, but with subsample dataset wrapper, bf16 autocast, weighted-channel L1, lambda LR scheduler, gradient clip 1.0). predict.py rewritten to load Transolver from the saved checkpoint with config.yaml. Refactored Transolver into model.py to avoid train.py CLI parsing during predict import.
- **Result:** 48 epochs in 28.2 min (timeout). Best epoch 31, val/loss=3.82, **avg_surf_p=91.88**. Per-split surf_p (best epoch): cruise=58.4, single_in_dist≈90, rc≈110, re_rand≈110. W&B run `thorfinn/iter1-192x6-bf16-sub30k-l1-pw3` (id 0ndqgt66).
- **Verdict:** Kept — first usable baseline. Predictions submitted to apr27-5/thorfinn/f745892. Far from frieren's apr27 score of 42.11 but a credible starting point for chained finetuning.
- **Notes:** First auto-predict failed because predict.py imported Transolver from train.py and triggered simple_parsing on the wrong argv; fixed by extracting the model into model.py. val/loss bounced epoch-to-epoch (likely surf-loss noise dominating). Next iter: chain finetune at bs=2, full mesh, low LR (1e-5 → 5e-6) following frieren's chain pattern.
