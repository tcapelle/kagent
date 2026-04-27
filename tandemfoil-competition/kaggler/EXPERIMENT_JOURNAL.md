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

### 2026-04-27 — Summary
- **Best result:** test 42.77 (rank 4) at commit c773fa7 (v7+v5 [0.72, 0.28] ensemble, val 44.86).
- **Latest submission:** v7+v10 per-split ensemble at f888fbe (val 43.87, awaiting scoring — likely test ~42).
- **Models trained:** v1 (thorfinn-clone), v2-v3 (chain warm-starts at lr=1e-4 then 3e-5), v4-v5 (slice=128 fresh + chain), v6 (fresh slice=64), v7 (chain v3 lr=1e-5), v8 (chain v5 lr=3e-5), v9 (256x8 fresh — discarded), v10 (chain v8 lr=1e-5).
- **What worked:** chain warm-starts at progressively lower LR (1e-4 → 3e-5 → 1e-5); ensembling slice=64 chain (v7) with slice=128 chain (v10) for architectural diversity; per-split weight tuning.
- **What didn't:** bigger model (256x8) needed too many epochs to compete in 30-min budget; SWA weight averaging marginal; adding more models to ensemble (v3, v5, v6) strictly hurt because they're correlated.
- **Why thorfinn won (35.26):** they trained 12 architecturally diverse models and used per-split routing/blending across them. To match, I'd need 6-8+ more models with different (n_hidden, n_layers, fun_dim, slice_num) combinations.

### 2026-04-27 — v10: chain v8 with lr=1e-5 (12ep) + final-6 ensemble v7+v10 per-split (val 43.87)
- **Hypothesis:** Even gentler chain on v8 refines it slightly. Pairs better with v7 in ensemble.
- **Change:** `--warm_start v8 --slice_num 128 --lr 1e-5 --epochs 12`. Then per-split sweep v7+v10.
- **Result:** v10 alone val=48.06 (epoch 10, beats v8=48.56 by 0.5). Ensemble v7+v10 [0.6, 0.4] uniform=43.90; per-split optima=43.87 (single 0.65/0.35, geom_rc 0.6/0.4, cruise 0.55/0.45, re_rand 0.55/0.45). Beats v7+v8 [0.6, 0.4]=44.02 by 0.12.
- **Verdict:** Submitted per-split as new best.
- **Notes:** Per-split optima all favor v10 slightly more than v8 favored. Run `8fazsiug`.

### 2026-04-27 — v9: bigger model n_hidden=256 n_layers=8 (22ep)
- **Hypothesis:** A genuinely larger architecture (1.5x params, 33% more layers) might give the diversity my v3/v5/v7/v8 family lacks. Even if individually slower to converge, complementary errors should help the ensemble.
- **Change:** `--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --mlp_ratio 2 --epochs 22`. 79s/epoch (1.7x slower), 18.2GB peak VRAM.
- **Result:** Best val avg_surf_p=75.47 at epoch 21 — much worse than v7=46.03. Cosine LR didn't fully decay (timeout-limited at 22 epochs vs needed 40+).
- **Verdict:** Discarded for ensemble. Adding v9 to v7+v8 ensemble hurts at every weighting tested ([0.5,0.4,0.1]=45.12, [0.55,0.4,0.05]=44.46, [0.45,0.40,0.15]=45.97 — all worse than v7+v8 alone=44.02).
- **Notes:** Bigger model needs longer training to compete. Not feasible within 30-min budget. The v7+v8 ensemble [0.6, 0.4] at val=44.02 (test 42.77 from c773fa7 with v7+v5 variant) remains my best. Run `ohxn4jiy`. Final standing: rank 3 at test 42.77 (thorfinn 36.82, tanjiro 41.60).

### 2026-04-27 — final-5: per-split ensemble (val 44.00)
- **Hypothesis:** Different test splits favor different ensemble weights — pick optimal weights per-split independently. Inspired by thorfinn's per-split routing (they jumped 45→37 with this trick).
- **Change:** Added `predict_per_split.py` (uses YAML config for per-split weights) and `eval_per_split.py` (sweeps weights and reports per-split optima). Submitted with: single [0.7, 0.3], geom_rc [0.6, 0.4], cruise [0.5, 0.5], re_rand [0.6, 0.4] over [v7, v8].
- **Result:** Per-split val=44.004 (vs uniform v7+v8 [0.6, 0.4]=44.022 → marginal -0.018).
- **Verdict:** Submitted. Marginal improvement because my v3/v5/v7/v8 are all in the same model family — limited diversity. Thorfinn likely has 12 truly diverse models, hence their bigger gain.
- **Notes:** Tested 4-way ensemble with v3+v7+v5+v8 — all per-split optima fell on v7+v8 family, didn't pick v3 or v5. To gain more, would need more diverse architectures (n_hidden=256, different activation, etc.). Time-limited so submitted current best. Run f233e58.

### 2026-04-27 — v8: chain v5 slice128 with lr=3e-5 (18ep) + final-4 ensemble v7+v8 [0.6, 0.4]
- **Hypothesis:** Chain v5 (slice=128, val 52.00) at lower LR refines it to be a better ensemble partner. Once v8 is closer to v7's quality, the diversity between slice=64 and slice=128 should give a stronger ensemble than v7+v5.
- **Change:** `--warm_start v5 --slice_num 128 --lr 3e-5 --epochs 18`. Then ensemble eval v7+v8 sweep, found [0.6, 0.4] best.
- **Result:** v8 alone val=48.56 at epoch 17 (beats v5=52.00 by 3.44). Ensemble v7+v8 [0.6, 0.4] val=44.02 (beats v7+v5 [0.72, 0.28]=44.86 by 0.84).
- **Verdict:** Kept and submitted as new final ensemble.
- **Notes:** Per-split improvements vs v7+v5: single 39.77 (was 40.28), geom_rc 60.13 (was 61.37), cruise 29.35 (was 29.85), re_rand 46.83 (was 47.92). All splits improved. Slightly closer weight balance (0.6/0.4) than the v7+v5 case (0.72/0.28) because v8 is much stronger than v5. Run `f3s5vkhf`.

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
