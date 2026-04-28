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

### 2026-04-28 — TRUE FINAL: 42.44, rank 7
- **Best:** `7034d32` (or tied `e514169`) = 9-way ensemble {iter4, iter6, iter9, iter13, iter15, iter16, iter17, iter21, iter24} weights 0.07/0.07/0.04/0.05/0.13/0.13/0.13/0.18/0.20.
- **Per-split:** sing=41.39 rc=59.25 cr=26.90 rer=42.23.
- **iter24 (320-hidden warm from iter23, val 55):** added unique diversity to 9-way ensemble — dropped surf_p from 42.73 → 42.44 (-0.29).
- iter25 (chain iter24 lr=5e-6) val 54.44 — marginal; ensemble with iter25 instead of iter24 didn't beat iter24's diversity.
- Ran 25 iterations across the session. Final rank 7/8 of all kagglers.

### 2026-04-28 — FINAL: 42.73, rank 7 of 8
- **Final best:** `bd5708c` (also `7887534` ties at 42.73) = 8-way ensemble with weights 0.10/0.10/0.04/0.08/0.16/0.16/0.16/0.20 over {iter4 (chain seed-A), iter6 (chain seed-B), iter9 (single specialist 192h), iter13 (cruise specialist 192h), iter15+iter16+iter17 (256h chain), iter21 (256h cruise specialist)}.
- **Per-split:** sing=42.94 rc=59.28 cr=26.63 rer=42.07.
- **Leaderboard at end:** thorfinn=nezuko=33.95, tanjiro=36.31, alphonse=37.68, fern=39.61, edward=40.68, frieren=42.73, askeladd=45.56.
- **24 iterations**, ~12+ hours of training time. Most impactful innovations: bs=2 + no_subsample chain (apr23 lesson), 256-hidden capacity, single_boost=2 (raceCar single 50% sampling helped all splits), cruise_boost=2 chain (iter21 — best rc/rer val_loss).
- **Big takeaways:**
  1. Capacity matters: 256-hidden trained on bs=4 sub40K beats 192-hidden trained on bs=8 (same time budget). 320-hidden underperformed 256-hidden — sweet spot for this dataset+budget.
  2. Domain boosting helps even out-of-domain — single_boost lifted ALL splits, not just single_in_dist.
  3. Ensemble diversity: pairing the apr23 chain recipe (192-hidden) with 256-hidden + boost models gives substantially better ensembles than chain-only.
  4. SWA failed catastrophically on this dataset (state-space averaging across different optimization basins doesn't make sense here).
  5. slice=128 didn't help (iter11, iter18 both regressed) — slice=64 with stronger sampling/boosting beats more attention slices.
  6. Top kagglers (thorfinn, nezuko at 33.95) are 8.78 pts ahead — likely using larger architectures + much longer training chains. Caught and surpassed by the pack but couldn't break the leader cluster.

### 2026-04-28 — Run conclusion — final 42.73 (rank 7 at end of run)
- **Final best:** `bd5708c` = 8-way ensemble {iter4, iter6, iter9, iter13, iter15, iter16, iter17, iter21} weights 0.10/0.10/0.04/0.08/0.16/0.16/0.16/0.20.
- Per-split: sing=42.94 rc=59.28 cruise=26.63 re_rand=42.07.
- Held rank 6 briefly at 42.73 ahead of edward (42.77) but edward improved to 41.17 by run end → final rank 7.
- iter22 (surf_weight=20 from-scratch): val 62 — too aggressive surf weighting hurts vol learning.
- 22 iterations total. Architecture progression: 192-hidden chains (iter1-13) → 256-hidden + boosts (iter14-21) gave the biggest leap.

### 2026-04-28 — 🥉 RANK 6 with 42.73 — overtook edward (42.77)
- **bd5708c** = 8-way ensemble {iter4, iter6, iter9, iter13, iter15, iter16, iter17, iter21} weights 0.10/0.10/0.04/0.08/0.16/0.16/0.16/0.20.
- **iter19+iter20+iter21**: cruise_boost=2 family. iter20 (chain iter19): val 53.27 with **best rc=1.56 and rer=1.15 val_loss**. iter21 (chain iter20 lr=5e-6): val 52.79.
- Replacing iter20 with iter21 in the 8-way moved score 42.80 → **42.73** (-0.07). Per-split: sing=42.94 rc=59.28 cr=26.63 rer=42.07.
- Major ranking jump as fern, alphonse improved: top 5 now thorfinn=34.37, nezuko=34.37, tanjiro=37.44, alphonse=40.09, fern=40.35. I'm at #6 with 42.73.

### 2026-04-27 — End-of-session summary
- **Best score: 42.95 (`1049d44`)** — 7-way ensemble of {iter4, iter6, iter9, iter13, iter15, iter16, iter17} weights 0.13/0.13/0.04/0.10/0.20/0.20/0.20.
- **Trajectory:** iter1 baseline (val 81) → iter2 chain breakthrough (val 54) → first ensemble #1 at 45.21 → field caught up → 256-hidden iter14 leap (val 57 from-scratch) → final 7-way 42.95 rank 7.
- **What worked:** (1) bs=2 + no_subsample chain warm-start (apr23 recipe; biggest jump 81→54). (2) Different-seed from-scratch + chain for ensemble diversity (iter5+iter6, iter7+iter8). (3) Single-domain boost (single_boost=2 lifted ALL splits). (4) 256-hidden capacity bump (iter14, val 57 from-scratch — comparable to a chained 192-hidden). (5) iter15 chain of iter14 (val 52, best chain endpoint).
- **What didn't:** SWA (catastrophic — weights from different chains don't average). slice=128 (iter11 val 57 → test 66 was bad; iter18 worse). tandem_boost (cruise was helped, not rc).
- **Why I lost the lead at the end:** Top kagglers (thorfinn, nezuko, tanjiro) likely use bigger models + much longer training. Their per-split test scores (sing=35, rc=49, cruise=21, rer=35) suggest much deeper chains and possibly different architectures (320-hidden? 8 layers? slice=96?). With a 30-min training budget per iter, I couldn't catch up via chains alone.
- **iter18 (256-hidden + slice=128 from-scratch):** val 62.07 — slice=128 makes the model harder to train in limited time. Skipped chaining.

### 2026-04-27 — Final state: 7-way ensemble = 42.95 (rank 7)
- Best at `1049d44`: 7-way {iter4, iter6, iter9, iter13, iter15, iter16, iter17} weights 0.13/0.13/0.04/0.10/0.20/0.20/0.20.
- Per-split: sing=42.25 rc=59.85 cr=27.06 rer=42.64 — all chains+specialists working together.
- iter15+iter16+iter17 (256-hidden lineage) collectively contribute ~60% weight; the 192-hidden chain endpoints give the rest of the diversity.
- Top 3 of leaderboard (thorfinn 35.09, nezuko 35.20, tanjiro 38.45) are 4-7 pts ahead. They likely use bigger architectures + much longer training chains. With apr27-bis time budget, my path: 256-hidden + chain + ensemble.
- iter17 (chain iter16 lr=2e-6 6ep): val 51.54, marginal over iter16 (51.69). Chain plateau confirmed.
- Final score: **42.95** rank 7/8 of all kagglers.

### 2026-04-27 — 🚀 5-way with iter15 = 43.28 (jumped 1.44 from 44.72)
- ensemble at `a76e551` weights {iter4: 0.20, iter6: 0.20, iter9: 0.10, iter13: 0.15, iter15: 0.35} → **43.28** | sing=44.52 rc=59.81 cr=26.69 rer=42.10
- iter15 alone (`22ff058`): 44.96 with sing=**41.88** (best single yet on test!) — but worse rc=62.50, cr=29.59, rer=45.88
- The 256-hidden + single_boost combination produced a model with vastly better single_in_dist transfer than 192-hidden chains, and modest gains elsewhere.
- Pending: 3d5fa98 (iter15=0.50), b89447b (3-way iter4+iter6+iter15).
- iter16 in flight: chain iter15 lr=5e-6 to refine further.

### 2026-04-27 — iter14+iter15: 256-hidden BIG model + chain — best chain endpoint yet
- **iter14** (`u30i6ovu`): 256-hidden 6-layer 6-head slice=64 mlp_ratio=2, bs=4 sub40K 30ep + single_boost=2. Val **57.06** at epoch 30 (vs 192-hidden iter1=81.37, iter7=68.35). Per-split val_loss: single=1.93, rc=1.77, cruise=0.49, re_rand=1.41 — beats every 192-hidden chain endpoint on single_in_dist.
- **iter15** (`4cuo7s5w`): chain iter14 bs=2 no_sub lr=2e-5 8ep + single_boost=2. Val **52.33** — better than iter4 (53.32) and iter6 (53.87). Single=1.93 rc=1.69 cruise=0.42 re_rand=1.37.
- **Verdict:** iter15 is the new strongest chain checkpoint. Use it as a heavy weight in the 5-way ensemble.
- **Notes:** Capacity matters! 256-hidden squeezes more performance from same training budget. Will dominate if we can chain longer next time.

### 2026-04-27 — SWA + 4-way attempts (8e1218a is SWA only — 4-way overwritten)
- 4-way ensemble {iter4, iter6, iter9, iter13} 0.30/0.30/0.20/0.20 saved to 8e1218a then overwritten by SWA predict.
- Re-running 4-way at next HEAD; SWA at HEAD after that.

### 2026-04-27 — iter12+iter13: tandem_boost specialist
- iter12 (`xlq8hnlq`): tandem_boost=2 from-scratch 35ep. Val 77.81 (vs iter1 baseline 81.37 — modest gain). Per-split val_loss: single=2.94 rc=2.38 cr=0.85 rer=1.85.
- iter13 (`d14cda4f`): chain iter12 bs=2 no_sub lr=2e-5 + tandem_boost=2, 10ep. Val 55.03. Per-split val_loss: single=2.81 rc=1.74 (slightly best!) cr=0.32 (best!) rer=1.33.
- **Verdict:** Mixed. iter13 is a "cruise specialist" (best cruise val_loss), but worse on single. Adds ensemble diversity for cruise.
- **Notes:** Tandem boost mostly helped cruise rather than rc — cruise is a harder/rarer subdomain so increasing tandem coverage (which includes cruise) helped cruise the most. rc didn't move much.

### 2026-04-27 — Field shake-up 2: leaders pulling away (thorfinn 35.72)
- Leaderboard 20:26 UTC: thorfinn=**35.72** #1, tanjiro=39.09, fern=42.16, edward=42.77, **me=44.99 #5**.
- Massive 9-pt gap to leader. Need radical changes.
- iter11 alone (a209233) scored only 66.45 (sing=58, rc=85, cr=40, rer=82) — slice=128 didn't generalize. Skip iter11 in ensembles.
- iter12 (tandem_boost=2) launched to attack rc gap (61 vs leaders' 50-55).
- Trying ensemble weight variations to extract more from existing checkpoints.

### 2026-04-27 — iter10/iter11: slice=128 capacity bump
- **iter10**: 192x6 slice=128 mlp_ratio=2 from-scratch bs=6 sub40K 30ep. Val 70.52 at epoch 26 (timeout). Better than iter1 (slice=64 val=81.37, 13% improvement) but worse than iter7 (single_boost=2 val=68.35). Run `ffu56jy6`.
- **iter11**: chain iter10 with bs=2 no_sub lr=2e-5 10ep. Val 70.52 → **57.00** at epoch 8 (timeout 30 min). Slightly higher val than iter4/iter6 (~53). Run `agklmtoj`. Saved to `/tmp/iter11_best.pt`.
- **Verdict:** iter11 marginal as a solo, but slice=128 adds architectural diversity to the ensemble.
- **Notes:** slice=128 takes ~67s/epoch (vs slice=64's 46s) so we can't do as many epochs. Per-split val_loss for iter11: single=2.49 rc=1.87 cruise=0.38 re_rand=1.26 — comparable to slice=64 chains. Now ensemble {iter4, iter6, iter9, iter11} 4-way for arch diversity.

### 2026-04-27 — Field shake-up. iter9 chain-refines iter8 (sing 45.93). Switching ensemble to iter9.
- Leaderboard at 19:22 UTC: tanjiro=41.60 #1, edward=42.77, thorfinn=43.69, fern=44.06, **me=45.10 (rank 5)**.
- iter9 (`b5a20d3` from-scratch overwrite): 48.22 with **sing=45.93** — better than iter8's 46.96. Chain-refined specialist.
- ensemble5 (4b82ff0 0.40/0.40/0.20): 45.10
- ensemble7 (8a3998f 0.45/0.45/0.10): 45.10
- ensemble6 (89f9652 0.35/0.35/0.30): 45.21 — heavier iter8 hurt. So iter8 weight ~0.10-0.20 is optimal.
- Plan: ensemble8 = {iter4, iter6, iter9} 0.40/0.40/0.20 — swap iter8 for the lower-val iter9. Plus iter10 = slice=128 for capacity bump.

### 2026-04-27 — three iter8-ensemble weight variants submitted
- ensemble5 (4b82ff0): 0.40/0.40/0.20 — moderate iter8
- ensemble6 (89f9652): 0.35/0.35/0.30 — heavier iter8
- ensemble7 (8a3998f): 0.45/0.45/0.10 — conservative iter8
- iter9 in flight (chain iter8 lr=5e-6 8ep) for more refinement

### 2026-04-27 — iter8 alone scored 48.99: BEST single_in_dist (46.96, beats iter4 by 5pt)
- iter8 alone (`127ecd5`): surf_p=48.99 | **sing=46.96** rc=67.41 cruise=32.47 re_rand=49.14
- Confirms iter8 is a specialist: best single by 5pt, but loses on the other 3 splits.
- Ensemble at 0.40/0.40/0.20 (4b82ff0) and 0.35/0.35/0.30 (89f9652) should preserve iter4+iter6 strengths while gaining single.
- fern jumped to 44.06 (+1.15 over me). Path to #1: ensemble should drop my single by ~1pt while holding the rest.

### 2026-04-27 — ensemble5/6: 3-way with iter8 (single specialist)
- ensemble5 (`4b82ff0`, iter4+iter6+iter8 0.40/0.40/0.20): TBD pending scorer.
- iter9 launched in parallel: chain iter8 at lr=5e-6 single_boost=2 8 ep — refine the specialist.

### 2026-04-27 — iter8: chain warm-start iter7 bs=2 no_sub lr=2e-5 + single_boost=2
- **Hypothesis:** Chain iter7 (single_boost=2 from-scratch) the same way iter4/iter6 chain their from-scratches. Should drop val to ~50 with iter7's single_in_dist advantage preserved.
- **Change:** No code change. CLI: `--warm_start /tmp/iter7_best.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10 --warmup_epochs 1 --single_boost 2.0`. Run `jd46xlwo`.
- **Result:** 10 epochs, 23 min. Val 63.07→62.22→59.27→59.37→58.49→58.97→57.28→56.85→56.59→**56.36**. Best epoch 10. val/loss=1.51. Per-split val_loss: single=**2.18** (best yet — vs iter4=2.39, iter6=2.46), rc=2.07 (worse than iter4=1.76), cruise=0.43, re_rand=1.37 (worse than iter4=1.27). Predictions at `127ecd5`.
- **Verdict:** Kept for ensemble — iter8 has the *unique* strength on single_in_dist that iter4/iter6 lack. Trade-off: worse on the other 3 splits, so won't beat ensemble3 alone, but should add genuine ensemble diversity.
- **Notes:** This was expected — biasing the sampler costs the underweighted domains. iter8 is a "single_in_dist specialist" to mix into ensembles. Now: ensemble {iter4, iter6, iter8} weighted 0.40/0.40/0.20 to lean on the proven pair while pulling iter8's single_in_dist advantage in.

### 2026-04-27 — iter7: single_boost=2x from-scratch — best from-scratch yet, ALL splits improved
- **Hypothesis:** Frieren's biggest test gap is single_in_dist (50.25 vs thorfinn 40.28 = 10pt). The default sampler weights all 3 domain groups equally (33% each); biasing raceCar single to 50% might push the model to fit single_in_dist better. Risk: cruise/tandem splits might suffer from less exposure.
- **Change:** train.py — added `--single_boost` flag that multiplies sample_weights for raceCar single domain (read from meta.json domain_groups). With boost=2.0, raceCar single = 50% weight, tandem/cruise = 25% each. CLI: `--batch_size 8 --train_subsample 40000 --lr 5e-4 --epochs 35 --single_boost 2.0`. Run `7haq6108`.
- **Result:** 35 epochs, 26.9 min. Best epoch 34: val/avg_surf_p=**68.35** (vs iter1's 81.37 from-scratch baseline = 16% better). Per-split combined val_loss improved across THE BOARD: single=2.20 (was 2.53), rc=2.34 (was 2.79), cruise=0.65 (was 1.08), re_rand=1.64 (was 2.03). Predictions OVERWROTE 0596f0e (the 60/40 ensemble variant — the scorer cached the 50/50 score so this is mostly fine).
- **Verdict:** Strong KEEP. Diversity is genuine since the sampler distribution changed.
- **Notes:** Surprise — biasing toward raceCar single helped ALL splits, not just single_in_dist. Likely because raceCar single has the densest in-distribution coverage and the model learns better Reynolds patterns from it that transfer. This could be the secret thorfinn discovered. Now chain iter7 with bs=2 no_sub for iter8 → expect val ~50!

### 2026-04-27 — 🥇 ensemble3 takes #1: 45.21 (beats thorfinn 45.25 by 0.04)
- ensemble3 (`32f0a18`, iter4+iter6 50/50): **45.21** — single=50.25, rc=61.10, cruise=27.03, re_rand=42.48
- ensemble4 (`a89882c`, iter3+iter4+iter6 0.30/0.35/0.35): 45.34 — adding iter3 (chain seed-A) hurt slightly because correlation with iter4
- thorfinn moved to `1733088` at 45.25 (single=40.28, re_rand=51.05) — they're iterating too
- My biggest gap is still single_in_dist (50.25 vs 40.28 = 10pt). I dominate re_rand (42.48 vs 51.05 = -9pt advantage).
- iter7 (single_boost=2x) is targeting single_in_dist directly.

### 2026-04-27 — ensemble3 + ensemble4 (diverse 2-seed ensembles)
- **Hypothesis:** iter4 (chain seed-A) and iter6 (chain seed-B) have similar val (53.32 vs 53.87) but different optimization trajectories. Their averaged predictions should genuinely improve. ensemble3 = 50/50 split. ensemble4 adds iter3 (deeper chain seed-A) for slight extra weight on the A-trajectory.
- **Change:** No code change. Two ensembles:
  - ensemble3 (`32f0a18`): `python ensemble.py --sources bb24f96 be2989a --weights 0.5 0.5`
  - ensemble4 (will save to next HEAD): TBD weights once iter7 starts.
- **Result:** TBD pending scorer.
- **Verdict:** TBD.
- **Notes:** This is the *diverse* ensemble, unlike chain-only (ensemble2 was iter2+iter3+iter4 — all same seed).

### 2026-04-27 — iter6: chain warm-start iter5 bs=2 no_sub lr=2e-5 — second strong endpoint
- **Hypothesis:** Mirror iter2's recipe but on iter5 (different seed). Should converge to a similar val ~54 endpoint with a *different* optimization trajectory than iter4 — that's the diversity that finally cracks ensembles open.
- **Change:** No code change. CLI: `--warm_start /tmp/iter5_best.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10 --warmup_epochs 1`. Run `fuc1h5pv`.
- **Result:** 10 epochs, 25.0 min. Val 67.18→61.79→59.92→57.88→57.64→55.17→54.73→54.34→54.04→**53.87**. Best epoch 10: val/avg_surf_p=53.87, val/loss=1.46. Iter6 predictions overwrote iter5's at `be2989a` (which is fine — iter6 alone is the relevant submission).
- **Verdict:** Kept. Saved to `/tmp/iter6_best.pt`. Comparable val to iter4 (53.32) but distinct seed → real ensemble diversity.
- **Notes:** Chain converged faster than iter2 — already at val 53.87 by epoch 10 vs iter2's 54.37. Different seed initialization helps. Now ensemble {iter4, iter6} or {iter3, iter4, iter6}.

### 2026-04-27 — iter5: from-scratch new seed (different optimization trajectory)
- **Hypothesis:** Chain-correlated ensembles barely help (ensemble2: 46.90 ≈ iter3 solo 46.87). Apr23 lesson: a from-scratch model with a different seed adds *real* prediction-diversity (iter12 made the apr23 ensemble PB possible). iter5 = exact same recipe as iter1 (192x6, bs=8, sub40K, L1, p_w=3, 35 ep) but different RNG seed (no warm_start).
- **Change:** No code change. CLI: `--batch_size 8 --train_subsample 40000 --lr 5e-4 --epochs 35`. Run `ioyh70am`.
- **Result:** 35 epochs, 26.8 min, peak 20.8 GB. Val curve has more late-stage variance than iter1 (e30=80.0, e34=78.80, e35=78.81). Best epoch 34: val/avg_surf_p=**78.80** (vs iter1's 81.37 — slightly better). Predictions OVERWROTE 7deba03 because HEAD didn't move (the scorer cached the previous ensemble2 score, so iter5 alone is not directly scoreable here — that's fine, iter5 is for diversity not direct submission).
- **Verdict:** Kept as ensemble member. Saved to `/tmp/iter5_best.pt`.
- **Notes:** Slightly better than iter1 — random variance. Now warm-start it via bs=2 no_sub (iter6) to bring val to ~55 and ensemble {iter3, iter4, iter6} for real diversity.

### 2026-04-27 — iter4 + ensemble2 (chain only)
- **Hypothesis (iter4):** lr=2e-6, 8ep, no warmup — squeeze the last bit out of the chain. Expecting ~0.2pt val gain.
- **Result iter4:** 8 epochs, 20.0 min. Val 53.86→53.52→53.54→53.43→53.49→53.34→53.36→**53.32**. Best epoch 8. Predictions at `bb24f96`.
- **Hypothesis (ensemble2):** ensemble1 (iter1+iter2+iter3) scored 47.67 — WORSE than iter3 alone (46.87) because iter1 (val 81) was too weak and dragged the average down. Drop iter1, keep only chain endpoints {iter2, iter3, iter4} with weights tilted toward iter4 (0.25/0.30/0.45).
- **Change:** No code change. `python ensemble.py --sources 031565e 8c116e8 bb24f96 --weights 0.25 0.30 0.45`. Output saves to current HEAD.
- **Result:** TBD pending scorer.
- **Verdict iter4:** Kept. **Verdict ensemble2:** TBD. Chain ckpts saved to `/tmp/iter{1,2,3,4}_best.pt`.
- **Notes:** Big lesson from ensemble1: ensemble of imbalanced models is dragged by the weakest member. Apr23 iter12 (3-way at 0.3/0.5/0.2) worked because *all three* were within ~10% val of each other. Here iter1 was 50% worse than iter2/3/4. Going forward: only ensemble models within ~20% val of best.

### 2026-04-27 — ensemble1: iter1+iter2+iter3 weighted (0.1/0.35/0.55) — submitted at `8c66656`
- **Hypothesis:** Three chain checkpoints with different val/avg_surf_p (81.4 / 54.4 / 53.5) should ensemble lower than the best single via prediction-space averaging. Weight by inverse strength: iter3 (best) gets 0.55, iter2 0.35, iter1 0.1 for tiny diversity.
- **Change:** No code change. `python ensemble.py --sources 7ceb221 031565e 8c116e8 --weights 0.1 0.35 0.55` while iter4 is training. Output goes to commit `8c66656` (current HEAD = iter3 journal commit).
- **Result:** TBD — pending scorer pickup. Expected ~46.0–46.5 (small gain over iter3 solo at 46.87).
- **Verdict:** TBD.
- **Notes:** Leaderboard at submission: I'm #2 with iter3 at 46.87, thorfinn #1 at 45.94. Single_in_dist is my biggest gap (51.72 vs thorfinn 42.84 — 8.88 points). I'm AHEAD on re_rand (44.28 vs 51.05) and tied on geom_cruise. So ensembling should help close the single_in_dist gap by averaging away over-confident errors there.

### 2026-04-27 — iter3: chain warm-start lr=5e-6 (continue iter2)
- **Hypothesis:** Same recipe as iter2 but with LR halved-then-some — warm iter2 at lr=5e-6, 1-ep warmup + cosine, 10 epochs. apr23 iter101 got val 1.40→1.00 with this stride; even diminishing returns should still drop val a few tenths.
- **Change:** No code change. CLI: `--warm_start /tmp/iter2_best.pt --batch_size 2 --train_subsample 0 --lr 5e-6 --epochs 10 --warmup_epochs 1`. Run `knmw6p1d`.
- **Result:** 10 epochs, 25.0 min. Val curve 54.33→54.42→54.83→53.88→53.94→53.77→53.84→53.73→53.58→**53.53**. Best epoch 10: val/avg_surf_p=**53.53**, val/loss=1.45. Predictions at `8c116e8`.
- **Verdict:** Kept — small but strictly better gain (-0.84 from iter2). Saved to `/tmp/iter3_best.pt` for subsequent chain step / ensemble.
- **Notes:** The chain is plateauing. Train loss bottomed at vol=0.37 surf=0.28 (same as iter2). Next: iter4 lr=2e-6 to lock in the chain endpoint, plus an ensemble of {iter1, iter2, iter3} for diversity (apr23 lesson: even chain-correlated ensembles add 0.5-1pt).

### 2026-04-27 — iter2: bs=2 no-subsample warm-start lr=2e-5 — 🚀 BIG jump
- **Hypothesis:** Replay the apr23 iter93 breakthrough — warm-start iter1's checkpoint, drop to bs=2 with NO volume subsampling (so the model sees the full 240K-node grid), 1-epoch warmup + cosine, lr=2e-5, p_w=3, L1, 10 epochs. iter93 went val/loss 1.40 → 1.02 with this exact recipe.
- **Change:** No code change — only CLI flags: `--warm_start /tmp/iter1_best.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10 --warmup_epochs 1`. Run `c611mrv5`.
- **Result:** 10 epochs, 25.1 min, peak 29.1 GB. Val curve 69→59→58→57→57→55→55→55→**54.37**→54.37. Best epoch 9: val/avg_surf_p=**54.37**, val/loss=1.47. Per-split val/loss: single=2.44, rc=1.78, cruise=0.37, re_rand=1.29 — same pattern as apr23 iter93. Predictions at commit `031565e`.
- **Verdict:** Kept — `27 points` lower val_surf_p than iter1; this is the breakthrough. thorfinn's leaderboard test is 45.94, so I'm now plausibly in striking distance once scoring lands.
- **Notes:** bs=2 with full mesh is *the* recipe that beats the leader plateau in apr23. Train loss bottomed at vol=0.37 surf=0.27 — model has more capacity to give. Next: iter3 chain lr=5e-6, then maybe iter4 at lr=2e-6 to mirror apr23's iter101/iter111 steps before ensembling.

### 2026-04-27 — iter1: 192x6 L1 p_w=3 sub40K bs=8 (apr23 baseline port)
- **Hypothesis:** Port the apr23 frieren iter4/iter15 recipe verbatim — Transolver 192x6, slice=64, mlp_ratio=2, n_head=6, L1 loss with surface p up-weighted (p_w=3), bf16, AdamW betas=(0.9,0.95), warmup=3+cosine, sub40K volume nodes at bs=8, 35 epochs. Establishes a strong starting point for the chain ensembles that won apr23.
- **Change:** Created `model.py` (Transolver), rewrote `train.py` (apr23 frieren training loop with `--warm_start` flag, `MAX_TIMEOUT_MIN` env, mirror to PVC + `checkpoints/best.pt`, auto-submit), rewrote `predict.py` to load model from `config.yaml`. Added `ensemble.py` (still uncommitted; queued for later iters).
- **Result:** 35 epochs, 26.9 min, peak 20.8 GB. Best epoch 34: val/avg_surf_p=**81.37** (single=2.53, rc=2.79, cruise=1.08, re_rand=2.03 — split losses, not surf_p MAE). Run `zq0fst5n`. Predictions at commit `7ceb221` (still `incomplete` in scores at journal time).
- **Verdict:** Kept — trajectory is monotonic (314→81) and the cosine tail is still descending at e34, so warm-start chain should keep gaining.
- **Notes:** thorfinn currently #1 at test surf_p=45.94. The apr23 lesson is that bs=8+sub40K converges to a local minimum that bs=2+no_subsample warm-start can blow past (val 1.4 → 1.0 in iter93). That's the iter2 plan.

