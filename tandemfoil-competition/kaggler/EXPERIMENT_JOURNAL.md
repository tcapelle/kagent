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

### 2026-04-27 — v7: deep chain v3 with lr=1e-5 (15ep) + final-3 ensemble v7+v5 [0.72, 0.28]
- **Hypothesis:** Even gentler chain (1e-5, 6x lower than v3's 3e-5) refines v3's plateau and the result ensembles slightly better with v5 than v3 itself does.
- **Change:** `--warm_start v3 --lr 1e-5 --epochs 15`. After: ran ensemble eval v7+v5 sweep, found [0.72, 0.28] best.
- **Result:** v7 alone val=46.03 (epoch 14, beats v3=46.26 by 0.23). v7+v5 [0.72, 0.28] val=44.85 (beats v3+v5 [0.7, 0.3]=45.17 by 0.32). Per-split improvements: single 40.28 (was 40.78), geom_rc 61.37 (was 61.87), re_rand 47.92 (was 48.19).
- **Verdict:** Kept and submitted as final ensemble.
- **Notes:** v7 also overwrote final-2 ensemble at edward/2856b96 — fixed by re-submitting v7+v5 ensemble at new commit. Run `izweqran`.

### 2026-04-27 — final-2: re-submit ensemble v3+v5 [0.7, 0.3] at b765682 (val 45.17)
- **Hypothesis:** v6's auto-submit accidentally overwrote my best-ensemble predictions at edward/d4b1548. Re-submit at the v6 journal commit to restore the best ensemble for grading.
- **Change:** Re-ran predict_ensemble.py with v3+v5 [0.7, 0.3] weights at HEAD=b765682.
- **Result:** Predictions saved to edward/b765682. Val (from earlier eval): 45.166 — best of any combination tested.
- **Verdict:** Submitted as final answer.
- **Notes:** Also tested 3-way ensemble v3+v5+v6 — adding v6 strictly worse (best 3-way: 45.563 with [0.6,0.3,0.1]). v6 too correlated with v3 to add diversity. Sticking with [v3, v5] = [0.7, 0.3].

### 2026-04-27 — v6: fresh slice=64 different seed (35ep)
- **Hypothesis:** Another fresh slice=64 model (different random init) provides ensemble diversity beyond what slice=128 (v5) gave us.
- **Change:** `--epochs 35` (no warm_start, no special args; just different random seed via fresh init).
- **Result:** Best val avg_surf_p=58.99 at epoch 35 (last epoch — still descending). 27 min total. Per-split: single_in_dist=54.16, geom_rc=77.62, geom_cruise=42.95, re_rand=61.22.
- **Verdict:** Kept as ensemble component — undertrained alone (58.99 vs v3 46.26) but contributes seed diversity.
- **Notes:** 35 epochs vs v1's 40 — got cosine LR slightly less decayed. Still useful for ensemble. Run `pnn6ips2`.

### 2026-04-27 — final: ensemble v3 (slice=64) + v5 (slice=128)
- **Hypothesis:** Averaging predictions from architecturally-diverse models (slice_num=64 vs 128) reduces variance and improves test MAE. Different slice counts capture different physics-mode partitions, so their errors should be partially decorrelated.
- **Change:** Added `predict_ensemble.py` that loads N checkpoints, runs each, and averages predictions (in physical units) with optional weights. Used equal weights for v3+v5.
- **Result:** Predictions saved to `edward/1f1c6b1/`. Val: not measured locally (no val ensemble harness). Best individual val: v3=46.26. Expected ensemble: 44-46 if errors are partially decorrelated, else ~46-48.
- **Verdict:** Submitted as final answer. Ensemble of two strongest checkpoints (v3 and v5) — diverse via slice count, both well-trained.
- **Notes:** Per-model val: v3 [single=41.33, geom_rc=63.04, cruise=31.33, re_rand=49.32]; v5 [single=49.62, geom_rc=70.21, cruise=34.27, re_rand=53.89]. v3 dominates on every split, so a 0.5/0.5 weighting may dilute its strength — could try [0.7, 0.3] weights as v6 if time permits.

### 2026-04-27 — v5: chain warm-start from v4 slice128 (lr=1e-4, 25ep)
- **Hypothesis:** Refining v4's undertrained slice=128 model brings it closer to v3's quality, while preserving the architectural diversity needed for the v3+v5 ensemble.
- **Change:** `--warm_start v4 --slice_num 128 --lr 1e-4 --epochs 25` (no warmup, plain cosine).
- **Result:** Best val avg_surf_p=52.00 at epoch 22 (vs v4=71.95 → 28% improvement). 27.8 min total. Per-split: single_in_dist=49.62, geom_rc=70.21, geom_cruise=34.27, re_rand=53.89.
- **Verdict:** Kept as ensemble component — diverse from v3 (slice=64) and now competitive on its own.
- **Notes:** Plateaued around 52-55 from epoch 12-25, slow tail. Run `rf1ojxjz`. Final: ensemble v3 + v5 predictions to push below v3's 46.26.

### 2026-04-27 — v4: fresh slice_num=128 for ensemble diversity (40ep target)
- **Hypothesis:** A fresh model with double the slice tokens (different inductive bias) gives ensemble diversity vs v3's slice=64 chain — even if individually worse, complementary errors should help when averaged.
- **Change:** `--slice_num 128 --epochs 40` (no warm_start). 67s/epoch (vs 46s for slice=64) due to 2x slice tokens.
- **Result:** Best val avg_surf_p=71.95 at epoch 26 (timeout-limited at 27/40 epochs). Per-split: single_in_dist=79.08, geom_rc=90.55, geom_cruise=50.00, re_rand=68.20. 30 min total, 15.2 GB peak VRAM.
- **Verdict:** Kept as ensemble component — solo too weak (71.95 vs v3 46.26), but this is the trade-off for diversity. Will chain v5 from v4 to refine before ensembling with v3.
- **Notes:** Cosine targeted 40 epochs but only got 27, so LR didn't fully decay. Consistent with thorfinn's slice=128 fresh result (70.52). Run `f72q8jpo`.

### 2026-04-27 — v3: second chain warm-start from v2 (lr=3e-5, 25ep)
- **Hypothesis:** Even gentler chain (3x lower LR than v2) lets cosine refine the v2 minimum further without disrupting it.
- **Change:** `--warm_start v2.checkpoint --lr 3e-5 --epochs 25` (no warmup, plain cosine).
- **Result:** Best val avg_surf_p=46.26 at epoch 23 (vs v2=47.67 → 3% improvement). 19.3 min total. Per-split: single_in_dist=41.33, geom_rc=63.04, geom_cruise=31.33, re_rand=49.32.
- **Verdict:** Kept — pushes within 0.3 of thorfinn's test 45.94 on val (and beats their val 52.08 by 6 points).
- **Notes:** Plateau around 46.2 across e16-25. Diminishing returns from more chaining. Only single_in_dist and re_rand really improve; geom_rc still stuck at 63 (vs thorfinn 61.7) — that's the differentiator. Run `l0nw6exf`. Next: try fresh-seed model for ensemble OR architectural change (slice_num=128 fresh).

### 2026-04-27 — v2: chain warm-start from v1 (lr=1e-4, 30ep)
- **Hypothesis:** Chain warm-start at lower LR refines v1's solution by exploring nearby minima with slow cosine decay (thorfinn went 56→52 doing this).
- **Change:** Added `--warm_start` arg pointing to v1 checkpoint; lr=1e-4 (5x v1's final), epochs=30, no warmup, plain cosine decay.
- **Result:** Best val avg_surf_p=47.67 at epoch 29 (vs v1=55.07 → 13% improvement). 23.2 min total. Per-split: single_in_dist=43.16, geom_rc=64.42, geom_cruise=32.40, re_rand=50.68.
- **Verdict:** Kept — beats thorfinn's val (52.08) by 4.4 points.
- **Notes:** Disruptive at first (e3=56.11) then cosine decay rescued and pushed below v1 by epoch 10. geom_rc still the weak split (64). Run `0zkmlvv9`. Next: try ensemble v1+v2 OR train fresh diverse model for ensemble.

### 2026-04-27 — v1: thorfinn-clone baseline
- **Hypothesis:** Thorfinn's recipe (L1 + p_weight=3 + surf_weight=10 + sub40k + bs=4 + bf16 + 40ep) is the gold standard. Cloning it should land me at val avg_surf_p ≈ 50-60.
- **Change:** Wrote train.py from scratch with thorfinn-style config (192/L6/h6/mlp2/slice64); split model into model.py for predict.py importability; added auto-submit + best-checkpoint mirror to PVC.
- **Result:** Best val avg_surf_p=55.07 at epoch 39; train completed 40 epochs in 30.1 min (10.5 GB peak). Per-split: single_in_dist=52.73, geom_rc=72.73, geom_cruise=38.06, re_rand=56.77.
- **Verdict:** Kept — solid baseline, ~12 points above thorfinn's val (52.08) most likely just seed/sampling variance.
- **Notes:** geom_camber_rc is the worst split (~73, while others are 38-57). Cosine LR fully decayed by e39, last 5 epochs all near-tie. Test scoring pending. Run `it01dzgl`. Next: chain warm-start at lower LR (thorfinn went 56→52 with this).
