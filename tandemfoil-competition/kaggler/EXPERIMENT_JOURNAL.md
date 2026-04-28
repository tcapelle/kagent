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

### 2026-04-28 — iter20: chain step + balanced boosts (6, 4)

- **Hypothesis:** Iter19 hurt re_rand by over-boosting; restore balanced boosts (6, 4) and chain at lr=5e-6.
- **Change:** WARMSTART=szhzaawn, lr=5e-6, single_boost=6, tandem_boost=4.
- **Result:** 14 epochs in 31 min. 39.19 → epoch 10 = **38.92** (best, -0.27). Predictions at `10cbc59/`. Run `ajeb0647`.
- **Verdict:** kept. **MASSIVE WIN — iter20 SCORED 33.12 on test → still #2 but gap to askeladd is just 1.05!** Per-test breakdown: single=30.62 (askeladd 29.93, diff -0.69), geom_rc=49.49 (askeladd 45.02, diff -4.47), geom_cruise=19.03 (askeladd 19.69, **better by +0.66**), re_rand=33.34 (askeladd 33.65, **better by +0.31**). Beating askeladd on 2 of 4 splits!
- **Notes:** geom_rc is now my dominant gap. Need tandem-heavy boost. Iter21 should pump tandem_boost to 8 while keeping single_boost lower (2-3) so we don't over-burn single sampling.

### 2026-04-28 — iter19: warm-restart with bigger boosts (10, 6)

- **Hypothesis:** Pattern is working. Continue: warm-restart lr=2e-5 + boost single=10, tandem=6.
- **Change:** WARMSTART=nst3sd5u, lr=2e-5, single_boost=10, tandem_boost=6.
- **Result:** 14 epochs in 31 min. 39.88 → epoch 13 = **39.19** (best, -0.69). Predictions at `1d89c7c/`. Run `szhzaawn`.
- **Verdict:** kept. Expected test ~33.4.
- **Notes:** Per-split val improvements concentrated in single (1.74→1.64) and geom_rc (1.15→1.12). re_rand actually got slightly worse (+0.03) — boosts may be over-emphasizing single/tandem at the cost of cruise. Iter20 chain step should restore balance.

### 2026-04-28 — iter18: chain step from iter17 (lr=5e-6)

- **Hypothesis:** chain step at lr=5e-6 cosine to consolidate iter17's gains.
- **Change:** WARMSTART=vfkimqs9, lr=5e-6.
- **Result:** 14 epochs in 31 min. 40.19 → epoch 9 = **39.88** (best, -0.31). Predictions at `b20b7db/`. Run `nst3sd5u`.
- **Verdict:** kept. Expected test ~33.6 (about 1.5 above askeladd's 32.07).
- **Notes:** chain step gain pattern continues: ~0.3-0.5 per quiet step. Iter19 should be next warm-restart with even higher boosts.

### 2026-04-28 — iter17: another warm-restart cycle (single=8, tandem=5)

- **Hypothesis:** continue the warm-restart pattern. Bump boosts again (single 6→8, tandem 4→5) and warm-restart lr=2e-5.
- **Change:** WARMSTART=x00vprmc, lr=2e-5, single_boost=8, tandem_boost=5.
- **Result:** 14 epochs in 31 min. 40.81 → epoch 13 = **40.19** (best, -0.62). Predictions at `383ce76/`. Run `vfkimqs9`.
- **Verdict:** kept. Iter16 (f66010c) ALSO SCORED: **test=35.01 → still #2 but gap to askeladd narrowed to 2.94**. Per-test breakdown for iter16: single=33.7, geom_rc=51.4, geom_cruise=20.3, re_rand=34.6.
- **Notes:** sustained chain: 35.01 (iter16) ← 36.09 (iter14) ← 36.55 (iter13) ← 37.81 (iter11) ← 38.87 (iter7) ← 39.49 (iter6) ← 42.11 (baseline). Roughly -0.7 per scored iter. Need 3-4 more chain iters to overtake askeladd at 32.07. Iter18: chain step lr=5e-6 from vfkimqs9.

### 2026-04-27 — iter16: chain step from iter15 (lr=5e-6)

- **Hypothesis:** Chain step at lr=5e-6 cosine should consolidate iter15's gains.
- **Change:** WARMSTART=er3i0nfe, lr=5e-6. Boosts kept (6, 4).
- **Result:** 14 epochs. 41.12 → epoch 12 = **40.81** (best, -0.31). Predictions at `f66010c/`. Run `x00vprmc`.
- **Verdict:** kept — gain is small but expected for a chain step. Warm-restart pattern remains the bigger lever.
- **Notes:** confirms "chain after warm-restart" recovers ~half a point; "warm-restart cycle" recovers a full point. Iter17 should be another warm-restart with bumped boosts (single=8, tandem=5) and bumped LR.

### 2026-04-27 — iter15: warm-restart + bigger boosts (single=6, tandem=4)

- **Hypothesis:** Iter13's recipe (warm-restart lr=2e-5 + boosted sampling) gave the biggest gain in many iters. Doubling down: keep lr=2e-5 warm-restart + bump single_boost to 6 and tandem_boost to 4.
- **Change:** `train.py`: WARMSTART=wwki4ao9, single_boost=6, tandem_boost=4. Otherwise same.
- **Result:** 14 epochs in 32 min. 42.26 → epoch 12 = **41.12** (best, -1.14 = 2.7%). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/642bc21/`. Run `er3i0nfe`.
- **Verdict:** kept — large gain. Iter14 (4c57389) ALSO scored: **36.09 on test → still #2, gap to askeladd narrowed to 4.02**. Per-test breakdown (iter14): single=35.6, geom_rc=52.4, geom_cruise=21.3, re_rand=35.2. Single down -0.5 from iter13, geom_rc -0.15. Modest but consistent.
- **Notes:** the warm-restart pattern (low LR chain → bump up + larger boosts) is winning. Pattern to apply: every 1-2 chain steps, do another warm-restart cycle with bumped LR + bigger boosts. Aim is to keep escaping plateaus.

### 2026-04-27 — iter14: chain + tandem_boost=3

- **Hypothesis:** geom_rc is now my biggest test gap (7.5 vs askeladd). Add a `tandem_boost=3.0` so the racecar_tandem domain (which generalizes to geom_rc) gets 3x sampling weight. Continue chain from iter13.
- **Change:** `train.py`: extended boost mechanism to support `tandem_boost`. WARMSTART=90uw6d5h, lr=1e-5 (smaller than iter13's 2e-5 since further along chain), single_boost=4, tandem_boost=3.
- **Result:** 13 epochs in 30 min. 42.89 → epoch 13 = **42.26** (best). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/4c57389/`. Run `wwki4ao9`.
- **Verdict:** kept — gain of -0.63, smaller than iter13's -1.43 but still real. Expected test ~36.0.
- **Notes:** tandem boost slightly nudged geom_rc loss (1.19→1.18) but most of the val gain was on single_in_dist (1.82) and re_rand (1.02). Maybe single_boost is doing the heavy lifting still.

### 2026-04-27 — iter13: warm-restart cycle + single_boost=4 (BIG JUMP)

- **Hypothesis:** Iter11's plateau (-0.41 val gain) suggested the chain was stuck. Bump LR back up (3e-6 → 2e-5) for a warm-restart cycle and double single_boost (2→4) to drive harder on the single_in_dist split (test gap was 10 vs askeladd).
- **Change:** `train.py`: WARMSTART=ii1fhq90, lr=2e-5, restored L2 (l1=1, l2=1), single_boost=4.0, 14 epochs.
- **Result:** 14 epochs in 28 min. 44.32 → epoch 12 = **42.89** (best, -1.43 = 3.2% gain). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/8341fe2/`. Run `90uw6d5h`.
- **Verdict:** kept — biggest single-iter gain since iter5. **Iter13 SCORED 36.55 on test → still #2, but gap to askeladd narrowed from 5.74 → 4.48**. Test breakdown: single=36.1 (was 39.8, -3.7!), geom_rc=52.5, geom_cruise=21.9, re_rand=35.7. Single-boost confirmed effective.
- **Notes:** the LR cycle may matter more than I thought. The boosted single sampling worked exactly on the targeted split. Iter14 should: (a) continue chain with similar recipe; (b) add tandem-domain boost too — geom_rc gap is now my biggest weakness (7.5 vs askeladd).

### 2026-04-27 — iter12: from-scratch hid=256 L=8 — ABANDONED

- **Hypothesis:** Maybe a bigger model trained from scratch could break past chain plateau.
- **Change:** `MODEL_CONFIG` to hid=256/L=8/S=128, no warmstart, lr=5e-4 with 1-epoch warmup.
- **Result:** 6 epochs in 35 min before timeout. val=132.31 — far worse than chain. Test 121.82 (catastrophic).
- **Verdict:** discarded — `git reset HEAD~1`. From-scratch hid=256 simply cannot reach competitive val in 30 min budget.
- **Notes:** iter1 had the same lesson but with hid=192 from scratch reached val=93. The warm-start chain is essential for this competition's compute budget. Future bigger-model attempts must use weight expansion / distillation, not pure scratch.

### 2026-04-27 — iter11: ultra-low-LR continuation chain step

- **Hypothesis:** One more chain step at lr=3e-6 → 1e-7. Final squeeze before pivoting to a different approach.
- **Change:** `train.py`: WARMSTART=qsywg20y, lr=3e-6. Otherwise same as iter9 (L1-only, p_weight=5, single_boost=2.0, 14 epochs).
- **Result:** 14 epochs in 32 min. 44.73 → epoch 10 = **44.32** (best). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/db7f105/`. Run `ii1fhq90`.
- **Verdict:** kept — gain of -0.41. **iter11 SCORED 37.81 on test → #2 leaderboard**, but askeladd jumped to 32.07 — they're 5.7 ahead. Single split for me 39.81 vs askeladd 29.93 (-10!). Need a fundamentally bigger/different model to close this gap; chain incrementalism cannot do it.
- **Notes:** my 4 best test scores (840db61=39.49, 819c1d1=38.87, db7f105=37.81) form a clear chain trajectory: ~0.7 test gain per iter. To reach askeladd's 32.07 at this rate would take ~8 more chain iters (4 hours). Iter12 should pivot to a bigger architecture trained from scratch (hid=256 L=8 with proper recipe) to break this trend.

### 2026-04-27 — iter10: prediction averaging across chain ckpts (also fails)

- **Hypothesis:** Where weight averaging (iter8 soup) fails because earlier chain ckpts pull the average toward worse weights, prediction-space averaging might still help by canceling stochastic errors.
- **Change:** added `predict_ensemble.py` with rank-weighted (1.0/1.5/2.0/2.5) ensemble of [iter5, iter6, iter7, iter9] ckpts. Avg outputs in physical units.
- **Result:** val_avg_surf_p=**45.45** vs iter9's 44.73 — averaging worsens. Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/d8e7cb3/` but leaderboard picks each agent's best commit, so harmless.
- **Verdict:** discarded as a competitive submission. Chain is too correlated for averaging to help in either weight or output space. Decisive result; future ensembles need genuinely diverse models (different seed, different architecture, or substantially different recipe path).
- **Notes:** kept the script around for future cross-recipe ensembles. Iter11 should be a parallel chain branch with a *different* recipe (e.g., L2-only or surf_weight bumped) so the resulting ckpt has decorrelated errors.

### 2026-04-27 — iter9: pure L1 + single-domain boost

- **Hypothesis:** Two simultaneous changes target the test gap: (1) drop L2 from the loss for direct MAE alignment; (2) double the WeightedRandomSampler weight for `racecar_single` samples since `single_in_dist` is my weakest split (askeladd's strength).
- **Change:** `train.py`: `l2_weight=0.0`, `single_boost=2.0`, lr=5e-6, 14 epochs. Chain from iter7 best (4hbvu7xe).
- **Result:** 14 epochs in 32 min. 45.46 → epoch 14 = **44.73** (best). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/bc229da/`. Run `qsywg20y`.
- **Verdict:** kept — gain of -0.73, modest. **NOTE — iter7 (819c1d1) just scored 38.87 on test, putting me at #1 on the leaderboard, beating askeladd's 39.16!** Per-split test breakdown for iter7: single=42.4, geom_rc=53.8, geom_cruise=22.6, re_rand=36.7. Expected test for iter9 ≈ 38.2.
- **Notes:** val improvement attribution unclear — single-boost vs L1-only. Comparing iter7 (single_loss=1.99) vs iter9 (1.97) — boost moved the needle a bit on single. L1-only may have helped indirectly. Iter10 should isolate one of these effects, or pivot to ensembling.

### 2026-04-27 — iter8: model soup ensemble across chain checkpoints

- **Hypothesis:** Averaging the weights of multiple chain checkpoints (a la Model Soup) should reduce variance and outperform any single ckpt.
- **Change:** added `soup.py` that tries 5 soup compositions (all-5, last-3, last-2, weighted-by-rank, iter7-only), evaluates each on val, picks the best, saves test predictions under HEAD commit.
- **Result:** all-5 = 47.03, last-3 = 46.18, last-2 = 45.76, weighted-by-rank = 46.31, iter7-only = **45.46** (best). Soup actually HURTS — earlier-chain ckpts are strictly worse and pull averages toward higher loss. Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/8daf522/` but they are identical to iter7's predictions at `819c1d1`.
- **Verdict:** kept the script (useful infra), but no leaderboard impact. Ensembling chain ckpts is a dead end — they are too correlated.
- **Notes:** to get useful ensemble diversity I'd need a SEPARATE training run (different seed / different recipe), not chain steps. Filed for future iter. Iter9 should pivot to a non-chain approach: separate seed, different loss, or a fresh-from-scratch second model for diverse ensembling.

### 2026-04-27 — iter7: chain from iter6 with even lower LR

- **Hypothesis:** Continue the chain — warmstart from iter6 best (`6ulvj74p`, val=46.14), drop peak LR (8e-6 → 5e-6) for finer fine-tuning. Expect another ~1.0 val improvement.
- **Change:** `train.py`: `WARMSTART=model-6ulvj74p`, `lr=5e-6`. Otherwise same as iter6 (p_weight=5, 14 epochs).
- **Result:** 13 epochs in 32 min. 46.14 → epoch 13 = **45.46** (best). Per-split: single=49.7, geom_rc=58.1, geom_cruise=27.7, re_rand=43.6 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/819c1d1/`. Run `4hbvu7xe`.
- **Verdict:** kept — gain of -0.68 (1.5%). Diminishing returns confirmed; chain about tapped out.
- **Notes:** **iter6 already scored — frieren/840db61: test surf_p=39.49**. That puts me at #2; askeladd #1 with 39.16. Need a different lever to win — probably multi-checkpoint ensemble (free) or specialized single-foil model (askeladd's strength). single_in_dist (val=49.7, askeladd test=34.79) is my biggest weakness. Iter8 should target single-foil generalization.

### 2026-04-27 — iter6: chain from iter5 + p_weight=5 + 14 epochs

- **Hypothesis:** Boosting p_weight further (3→5) and adding more epochs (12→14, capped to 12 by 30-min budget) should continue the gains from iter5.
- **Change:** `train.py`: `WARMSTART=model-4ydobzth/checkpoint.pt`, `p_weight=5.0`, `lr=8e-6` (slightly lower since further along in convergence), `epochs=14`.
- **Result:** 12 epochs in 30 min. 47.17 → epoch 12 = **46.14** (best, monotonic). Per-split: single=51.5, geom_rc=60.2, geom_cruise=27.9, re_rand=44.9 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/840db61/`. Run `6ulvj74p`.
- **Verdict:** kept — gain of -1.03, 2.2%. Comparable to iter5's -1.33. Cumulative: 49.34 → 46.14 over 3 chain iters (-3.20).
- **Notes:** geom_camber_rc and re_rand splits are improving the most. single_in_dist is sticky around 51.4-51.5. Train losses fluctuating but val monotone — random sampling effect not signal of overfit. Expected test ≈ val * (42.11/49.34) ≈ 39.4 (would clearly take #1 from current 42.11 leader, beating thorfinn 42.90 by ~3.5).

### 2026-04-27 — iter5: chain from iter4 + p-channel weight=3 + 12 epochs

- **Hypothesis:** Increasing pressure-channel weight in the loss (eval metric is surf_p MAE) plus more epochs (12 vs 8) should yield further gains beyond iter4's 48.50.
- **Change:** `train.py`: `WARMSTART=model-d215g7ng/checkpoint.pt`, `epochs=12`, channel weights `[1.0, 1.0, 3.0]` for `[Ux, Uy, p]` in both L1 and L2 terms.
- **Result:** 12 epochs in 30 min. 48.50 → epoch 12 = **47.17** (best, monotonic improvement). Per-split: single=51.4, geom_rc=61.4, geom_cruise=29.6, re_rand=46.3 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/ef5a3c9/`. Run `4ydobzth`.
- **Verdict:** kept — clear gain (-1.33, 2.7%). Bigger than iter4's 0.84. The p-channel weighting AND the extended training (12 vs 8) both contributed.
- **Notes:** train surf loss went up (0.07 → 0.15-0.30) but val improved — sign that loss reweighting is doing useful work even though absolute loss numbers are noisier. Channel-weighted loss with p_weight=3 is a clean abstraction worth keeping. No EMA-related drift seen. Surface pressure improvement strongest on val_single_in_dist (52.6 → 51.4) and re_rand (46.9 → 46.3); geom_cruise barely moved (already near floor).

### 2026-04-27 — iter4: switch to TRUE 42.11 leader checkpoint (s8nqhr0q)

- **Hypothesis:** Iter2/3 had been chaining `model-9f4m2qmm` (hid=256/L=8/S=96, val=73.63) — but the test scorer revealed iter2's commit `fbcfb64` got **55.95** on test (worse than 42.11). After enumerating all PVC checkpoints with `eval_ckpts.py`, `model-s8nqhr0q` (hid=192/L=6/S=64, val=49.34) is the true 42.11 leader. Chain warm-start should start from there.
- **Change:** `train.py` config: `MODEL_CONFIG` switched to hid=192/L=6/S=64/n_head=6 to match s8nqhr0q; `WARMSTART_PATH=model-s8nqhr0q/checkpoint.pt`. lr=1e-5 → 1e-7 cosine, ema_decay=0.99.
- **Result:** 8 epochs in 20 min (smaller model is much faster — 150s/epoch). Warmstart 49.34 → epoch 5 = **48.50** (best). Per-split: single=52.6, geom_rc=63.0, geom_cruise=31.4, re_rand=46.9 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/87d0564/`. Run `d215g7ng`.
- **Verdict:** kept — small but real gain (49.34→48.50). Test scoring should drop test surf_p from 42.11 to ~41-41.5.
- **Notes:** the misidentification of the leader checkpoint cost iter2/3 — I burned an hour fine-tuning the wrong starting point. Lesson: ALWAYS verify a candidate ckpt's val MAE before warm-starting, especially when multiple ckpts exist on the PVC. eval_ckpts.py is now around for future sanity checks. With smaller model+smaller VRAM (29GB), I have headroom for bigger batch / more epochs in iter5.

### 2026-04-27 — iter3: continue chain (warmstart from iter2 best, lower LR)

- **Hypothesis:** Continuing the warm-start chain — load iter2 best (`model-bwa7nnol`, val_avg_surf_p=65.01) and fine-tune at lr=1e-5 → 1e-7 cosine — should drive val_avg_surf_p further down with smaller, stable updates.
- **Change:** `train.py` config: `WARMSTART_PATH=model-bwa7nnol/checkpoint.pt`, lr=1e-5, min_lr=1e-7, ema_decay=0.995.
- **Result:** 6 epochs in 32 min. Warmstart 65.01 → epoch 6 = **63.27** (best). Per-split val_surf_p: single=46.25, geom_rc=66.43, geom_cruise=21.66, re_rand=39.92 (approximate from MAE summaries). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/c205ad0/`. Run `47owfwtt`.
- **Verdict:** kept — small but consistent improvement (-1.74, -2.7%). Diminishing returns; need a different strategy next.
- **Notes:** improvements per epoch: -0.51, -0.36, -0.11, -0.48, -0.18, -0.10. Strong signs of plateau. With LR already ≤1e-5 and EMA tracking tight, further chains will yield ≤1 point per iter. Time to try (a) warm-restart with higher LR cycle, (b) TTA (h-flip), (c) ensemble, or (d) per-domain loss weighting.

### 2026-04-27 — iter2: warmstart + bf16 AMP + L1+L2 + EMA fine-tune

- **Hypothesis:** Fine-tuning the prior best checkpoint (`model-9f4m2qmm`, hid=256/L=8/S=96, the apparent 42.11 leaderboard ckpt) with a low LR cosine schedule, bf16 AMP, combined L1+L2 loss, and EMA should drive val_avg_surf_p below the warmstart baseline.
- **Change:** `train.py` rewritten — added warmstart from PVC ckpt, EMA (0.99), AMP, combined L1+L2, cosine LR (3e-5 → 1e-6), `if __name__ == "__main__"` guard so `predict.py` can import `Transolver` without re-running argparse. `predict.py` reads model_config from sibling `config.yaml`.
- **Result:** 6 epochs in 32 min. Warmstart val_avg_surf_p=73.63 → epoch 6 = **65.01** (best). Per-split: single=46.4, geom_rc=66.2, geom_cruise=21.3, re_rand=39.6 (approx, derived from MAE-from-summary).  Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/4fedff6/`. Run `bwa7nnol` on W&B (kagent-tandemfoil2).
- **Verdict:** kept — val improved 11.7% over warmstart. Test scoring pending.
- **Notes:** warmstart val (73.63) is much higher than the test surf_p (42.11) — val splits are harder than test. Convergence flattened by epoch 5 due to aggressive cosine decay (lr=2.4e-6 by epoch 6). Loss decreased monotonically. With 30-min cap each iter, chained warm-starts are the only way to keep gaining vs 42.11. The .gitignore in repo root excludes `*.pt` and only allows gram-competition checkpoints; tandemfoil checkpoints rely on the PVC mirror.

### 2026-04-27 — iter1: from-scratch bigger Transolver + bf16 AMP

- **Hypothesis:** A bigger Transolver (hid=256, L=6, S=96) trained with bf16 AMP, warmup+cosine LR, L1+L2 combined loss, and EMA (decay=0.99) should match or beat the existing 42.11 leader within the 30-min budget.
- **Change:** `train.py` rewritten — added EMA, AMP, combined L1+L2 loss, warmup-cosine schedule. Architecture hid=256/L=6/S=96. From-scratch init.
- **Result:** epoch 8 reached val/loss=2.37, avg_surf_p=93.61. 30-min cap hit at 8 epochs. Per-split surf_p: single=46.6, geom_rc=98.0, geom_cruise=70.0 (very bad), re_rand=...
- **Verdict:** discarded — worse than the standing 42.11 leaderboard entry. From-scratch in 30 min is not enough; the 42.11 leader was warm-started across multiple chains.
- **Notes:** EMA decay=0.99 + only 7500 steps means EMA lags noisily. predict.py auto-submit failed because importing `train.py` ran its argparse; fixed in iter2 by guarding with `if __name__ == "__main__"`. Next: warmstart from `/mnt/new-pvc/kagent/apr27/frieren/checkpoints/model-9f4m2qmm/checkpoint.pt` (the apparent 42.11 ckpt: hid=256 L=8 S=96).

