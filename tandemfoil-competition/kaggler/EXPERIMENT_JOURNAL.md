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

### 2026-04-28 — iter25: 10th warm-restart (lr=3e-5, pw=22, sw=22)
- **Hypothesis:** Continue cycling. Bump pw and sw 20→22.
- **Change:** Resume iter24; --lr 3e-5 --p_weight 22 --surf_weight 22.
- **Result:** **surf_p 39.35 → 38.77** (~1.5% gain). Per-split: rc=5.67, single=3.86, cruise=2.12, re_rand=4.26. W&B `4gmvmto1`.
- **Verdict:** Kept (best). Now 8.0% below frieren's 42.11.
- **Notes:** iter26 polish, iter27 11th warm-restart.

### 2026-04-28 — iter24: polish iter23 (lr=5e-6, pw=20, sw=20)
- **Hypothesis:** Standard polish.
- **Change:** Resume iter23; --lr 5e-6 (other unchanged).
- **Result:** **surf_p 39.75 → 39.35** (~1.0%, best ep8). W&B `bk30a2pi`.
- **Verdict:** Kept (best). Now 6.6% below frieren's 42.11.
- **Notes:** iter25 10th warm-restart with pw=22, sw=22 (+10% from iter23).

### 2026-04-28 — iter23: 9th warm-restart (lr=3e-5, pw=20, sw=20)
- **Hypothesis:** Continue cycling. Bump pw and sw 18→20.
- **Change:** Resume iter22; --lr 3e-5 --p_weight 20 --surf_weight 20.
- **Result:** **surf_p 40.16 → 39.75** (~1.0%, smaller jump). Per-split: rc=5.28, single=3.59, cruise=1.97, re_rand=4.02. W&B `a394tbto`.
- **Verdict:** Kept (best). Now 5.6% below frieren's 42.11.
- **Notes:** Cycle gain dropping (1.8% → 1.0%). Approaching the limits of this recipe. iter24 polish, iter25 may try something new (e.g., switching back to pw=18 sw=18 with bigger lr kick, or training a 2nd model for ensemble).

### 2026-04-28 — iter22: polish iter21 (lr=5e-6, pw=18, sw=18)
- **Hypothesis:** Standard polish.
- **Change:** Resume iter21; --lr 5e-6 (other unchanged).
- **Result:** **surf_p 40.52 → 40.16** (~0.9%, best ep8). W&B `ae3ypatq`.
- **Verdict:** Kept (best). Now 4.6% below frieren's 42.11.
- **Notes:** iter23 9th warm-restart with pw=20 sw=20 (slight bump from 18).

### 2026-04-28 — iter21: 8th warm-restart (lr=3e-5, pw=18, sw=18)
- **Hypothesis:** Same recipe as iter19 worked — repeat for 8th cycle.
- **Change:** Resume iter20; --lr 3e-5 --p_weight 18 --surf_weight 18.
- **Result:** **surf_p 41.25 → 40.52** (~1.8% gain). Per-split: rc=4.81, single=3.31, cruise=1.85, re_rand=3.68. W&B `kvwjhxgi`.
- **Verdict:** Kept (best). Now 3.8% below frieren's 42.11.
- **Notes:** Cycle gain stayed at 1.8% — pattern holds. iter22 polish, iter23 9th warm-restart.

### 2026-04-28 — iter20: polish iter19 (lr=5e-6, pw=18, sw=18)
- **Hypothesis:** Standard polish; lr=5e-6 (slightly higher than typical 4e-6 to allow more movement on the bigger update).
- **Change:** Resume iter19; --lr 5e-6 (other unchanged).
- **Result:** **surf_p 41.61 → 41.25** (~0.9%, best ep9). W&B `s90x42cb`.
- **Verdict:** Kept (best). Now 2.0% below frieren's 42.11.
- **Notes:** iter21 8th warm-restart with same proven recipe (lr=3e-5, pw=18, sw=18).

### 2026-04-28 — iter19: 7th warm-restart (lr=3e-5, pw=18, sw=18) — BEAT FRIEREN 42.11
- **Hypothesis:** 7th warm-restart cycle. Bigger kick (lr=3e-5 vs 2.5e-5) and bump sw 15→18. Goal: drop below 42.11.
- **Change:** Resume iter18; --lr 3e-5 --p_weight 18 --surf_weight 18.
- **Result:** **surf_p 42.39 → 41.61** (~1.8% gain). Per-split: rc=4.91, single=3.42, cruise=1.91, re_rand=3.75. Still descending. W&B `6jsip57v`.
- **Verdict:** Kept (best). 🎯 **Beat frieren's apr27 leader score (42.11)** for the first time in this session — by 1.2%.
- **Notes:** Total improvement since iter5 (51.82): 19.7% absolute. Going to polish + 8th warm-restart to push deeper.

### 2026-04-28 — iter18: polish iter17 (lr=4e-6, pw=18, sw=15)
- **Hypothesis:** Standard polish.
- **Change:** Resume iter17; --lr 4e-6 (other unchanged).
- **Result:** **surf_p 42.73 → 42.39** (~0.8%, best ep6). W&B `5rrrktxh`.
- **Verdict:** Kept (best). Within 0.7% of frieren's 42.11.
- **Notes:** iter19 will push 7th warm-restart with pw=18, sw=18, lr=3e-5 (bigger kick) to break through.

### 2026-04-27 — iter17: 6th warm-restart (lr=2.5e-5, pw=18, sw=15)
- **Hypothesis:** 6th warm-restart; bump pw 15→18 to push pressure focus harder. Target: drop below frieren's 42.11.
- **Change:** Resume iter16; --lr 2.5e-5 --p_weight 18 --surf_weight 15.
- **Result:** **surf_p 43.51 → 42.73** (~1.8% gain). Per-split: rc=4.19, single=2.97, cruise=1.69, re_rand=3.25. Still descending. W&B `1sdlp4d5`.
- **Verdict:** Kept (best). Now within 1.5% of frieren's apr27 score (42.11). One more cycle should put me ahead.
- **Notes:** Cycle gain dropped from 2.8% (cycle 5) to 1.8% (cycle 6) — diminishing but still meaningful. iter18 polish, iter19 7th warm-restart at pw=18 lr=2.5e-5 (or higher).

### 2026-04-27 — iter16: polish iter15 (lr=4e-6, pw=15, sw=15)
- **Hypothesis:** Standard polish with slightly higher lr (4e-6 vs 3e-6) to allow more movement in 10 epochs.
- **Change:** Resume iter15; --lr 4e-6 (other unchanged).
- **Result:** **surf_p 43.91 → 43.51** (~0.9%, best ep10). W&B `fxyy0wmw`.
- **Verdict:** Kept (best). Polish gave the typical ~1%.
- **Notes:** iter17 6th warm-restart with pw=18 (was 15) and lr=2.5e-5 — should push under 42 to top frieren's apr27 score.

### 2026-04-27 — iter15: 5th warm-restart (lr=2.5e-5, pw=15, sw=15)
- **Hypothesis:** 5th warm-restart cycle. Push pw further (12→15) and slightly higher LR (2e-5→2.5e-5).
- **Change:** Resume iter14; --lr 2.5e-5 --p_weight 15 --surf_weight 15.
- **Result:** **surf_p 45.18 → 43.91** (~2.8% gain — best gain yet). Per-split: rc=4.22 (slow improvement on hardest split), single=3.04, cruise=1.73, re_rand=3.33. W&B `5mgv0u7n`.
- **Verdict:** Kept (best). Now within ~4% of frieren's apr27 score (42.11). One more warm-restart cycle should put me ahead.
- **Notes:** Each warm-restart cycle has yielded 2-3%; total since iter5: 51.82 → 43.91 (15.3% reduction). The gradient on pressure is still effective at this LR/pw level. iter16 polish, iter17 6th warm-restart.

### 2026-04-27 — iter14: polish iter13 (lr=3e-6, pw=12, sw=15)
- **Hypothesis:** Standard polish at low LR.
- **Change:** Resume iter13; --lr 3e-6 (other unchanged).
- **Result:** **surf_p 45.53 → 45.18** (~0.8%, best ep9). W&B `k69eageq`.
- **Verdict:** Kept (best). Diminishing-returns polish.
- **Notes:** Continuing the cycle. iter15 5th warm-restart with pw=15 (stronger) lr=2.5e-5.

### 2026-04-27 — iter13: 4th warm-restart (lr=2e-5, pw=12, sw=15)
- **Hypothesis:** Push surface weight further (sw=12→15) and warm-restart to escape iter12 plateau.
- **Change:** Resume iter12; --lr 2e-5 --p_weight 12 --surf_weight 15.
- **Result:** **surf_p 46.58 → 45.53** (~2.3% gain). Per-split: rc=4.31, single=3.11 (val_loss inflated by sw=15). Still descending. W&B `78ltz16v`.
- **Verdict:** Kept (best). Pattern continues — 4 warm-restart cycles each yielded 2-3%.
- **Notes:** Within ~8% of frieren's apr27 score (42.11). Two more cycles should be enough to top them. iter14 polish, iter15 5th warm-restart.

### 2026-04-27 — iter12: polish iter11 (lr=3e-6, pw=12, sw=12)
- **Hypothesis:** Standard polish-after-warm-restart with the new sw=12 weighting.
- **Change:** Resume iter11; --lr 3e-6 --p_weight 12 --surf_weight 12.
- **Result:** **surf_p 47.00 → 46.58** (~0.9%, best ep7). Per-split: single≈55, rc≈73, cruise≈30, re_rand≈49. W&B `b952g8ux`.
- **Verdict:** Kept (best). Plateau hits earlier this cycle — best at ep7 instead of ep10.
- **Notes:** Going to push: iter13 = 4th warm-restart with sw=15 (was 12) to see if more surface weight helps. Pattern of (warm-restart → polish) cycles is yielding ~3-4% per pair.

### 2026-04-27 — iter11: 3rd warm-restart from iter10 (lr=2e-5, pw=12, sw=12)
- **Hypothesis:** 3rd warm-restart cycle with stronger pressure (pw=10→12) and surface (sw=10→12) weighting. Should escape current plateau again.
- **Change:** Resume iter10; --lr 2e-5 --p_weight 12 --surf_weight 12.
- **Result:** **surf_p 48.15 → 47.00** (~2.4% gain). Per-split: single≈55, rc≈73, cruise≈30, re_rand≈49 (rc still hardest, others improving). Still descending. W&B `xoko69un`.
- **Verdict:** Kept (best). Pattern holds: warm-restart cycles consistently yield 2-3% gains. Now closer to frieren's apr27 score (42.11) — still ~12% above.
- **Notes:** val/loss went UP (2.30 → 2.65) because of higher sw/pw scaling factors, but surf_p (the metric that matters) improved. Continue: iter12 polish, then iter13 4th warm-restart.

### 2026-04-27 — iter10: polish iter9 (lr=3e-6, pw=10)
- **Hypothesis:** Standard polish-after-warm-restart at lower LR.
- **Change:** Resume iter9; --lr 3e-6 --p_weight 10.
- **Result:** **surf_p 48.66 → 48.15** (~1%). Per-split: single≈55, rc≈70, cruise≈30, re_rand≈51. W&B `8ii5o16f`.
- **Verdict:** Kept (best). Plateau as expected post-restart.
- **Notes:** Going to attempt a 3rd warm-restart cycle (iter11) at lr=2e-5 with pw=12, sw=12 to push pressure even harder.

### 2026-04-27 — iter9: 2nd warm-restart from iter8 (lr=2e-5, pw=10)
- **Hypothesis:** First warm-restart (iter7) got 2.5% gain. Try a stronger second cycle: 2x the LR (1e-5→2e-5) and bump pw 8→10 to push pressure further. Same shape (bs=2 nosub ep=10).
- **Change:** Resume iter8; --lr 2e-5 --p_weight 10 --warmup_epochs 0.
- **Result:** **surf_p 50.30 → 48.66** (~3.3% gain — biggest single jump since iter2). Per-split: single≈55, rc≈70, cruise≈30, re_rand≈51. Still descending at ep10 (was 50.94 ep5, 49.18 ep8, 48.66 ep10). W&B `9qndq1e6`.
- **Verdict:** Kept (best). Two warm-restart cycles confirms the recipe. Big gains came from epoch 6+ once LR dropped on the cosine.
- **Notes:** rc split improved from 3.10 → 3.08 — hardest split is still resistant. May need yet another warm-restart cycle. iter10 will polish at lr=3e-6 pw=10, then iter11 try lr=3e-5 pw=10.

### 2026-04-27 — iter8: chain ft iter7 (lr=2e-6, pw=8)
- **Hypothesis:** Polish iter7 with low LR — same recipe as iter3 after iter2.
- **Change:** Resume from iter7; --lr 2e-6 --p_weight 8 (other unchanged).
- **Result:** **surf_p 50.51 → 50.30** (~0.4%). Per-split: single≈55, rc≈73, cruise≈30, re_rand≈53. W&B `1tx534yz`.
- **Verdict:** Kept (best). Plateau again — same shape as post-iter3 plateau (warm-restart → polish → small gain).
- **Notes:** Pattern is clear: warm-restart yields ~2.5% jumps, polishing yields ~0.5%. Going to attempt another warm-restart at even higher LR + pw=10 (iter9).

### 2026-04-27 — iter7: warm-restart from iter5 (lr=1e-5, p_weight=8)
- **Hypothesis:** iter5 plateaued at 51.82 with very low LR. A warm-restart at higher LR (1e-5, ~5x iter5) plus stronger pressure weight (pw=6→8) should kick the optimizer out of the local min and bias more toward pressure error.
- **Change:** Resume from iter5 ckpt; --lr 1e-5 --p_weight 8 (other hyperparams unchanged).
- **Result:** **surf_p 51.82 → 50.51** (~2.5% gain — first jump >1% since iter3). Per-split: single≈55, rc≈73, cruise≈30, re_rand≈53. Still descending at epoch 10. W&B `kn4rwxqy`.
- **Verdict:** Kept (best). Big win — confirms warm-restart escapes plateau.
- **Notes:** Both levers (LR kick + stronger pw) likely contribute. Next: chain ft at low LR to polish (iter8), then maybe another warm-restart cycle.

### 2026-04-27 — iter6: fresh from scratch with bs=2 nosub (frieren iter93 recipe)
- **Hypothesis:** frieren's run history showed a big drop (val~1.4 → ~1.0) when restarting from scratch with bs=2 full mesh — maybe my chain-from-sub30k path is stuck in a local min.
- **Change:** New training run, no resume; bs=2, n_sub=0, ep=10, lr=2e-5, warmup_epochs=1, p_weight=3 (matching iter2 first finetune step).
- **Result:** 10 ep in 24.8 min. **surf_p=109.48** (from scratch). Worse than iter1 baseline (91.88) and far from iter5 (51.82).
- **Verdict:** Discarded — fresh restart at bs=2 nosub doesn't beat the iter1+chain pipeline in 10 epochs. Restored iter5 ckpt as best.local.
- **Notes:** 10 epochs of full-mesh from scratch is just not enough; the model needs the sub30k pre-training to get fast initial progress. iter6 ckpt kept on PVC for possible ensemble use.

### 2026-04-27 — iter5: chain finetune iter4 (lr=2e-6, p_weight=6)
- **Hypothesis:** iter4 was still descending — drop LR another 2.5x for further polishing.
- **Change:** Resume from iter4 ckpt; --lr 2e-6 --p_weight 6 (same shape).
- **Result:** **surf_p 52.05 → 51.82** (~0.4% gain). val/loss 2.36 → 2.35. Per-split: single≈54, rc≈74, cruise≈30, re_rand≈53. W&B `2qnnkhmb`.
- **Verdict:** Kept (best). Plateau confirmed: per-epoch deltas <0.5%.
- **Notes:** Tried SWA of iter3+iter4+iter5 ckpts — gave 52.30, worse than just iter5 (51.82). Chain is monotonic so averaging back-tracks. Next: try a more aggressive lever (larger model 256x8 from scratch, or stronger p_weight=10 + more chain).

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
