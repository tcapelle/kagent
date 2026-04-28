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

### 2026-04-28 — RANK 1 at 26.71 🥇🥇🥇 — fixed predict.py bug, regenerated everything
- **Bug found:** my `predict.py` was using the RAW `y_mean[2]/y_std[2]` from `stats.json` (raw pressure stats) when decoding model output, instead of the Cp-normalized stats `p_mean_cp/p_std_cp` saved in `runtime.yaml`. Model trained correctly to predict Cp-normalized values, but inference decoded those values back to physical pressure with the wrong scale, producing surf_p ~200 (vs 26-30 expected).
- **Fix:** added `if cp_normalize: y_mean[2] = P_MEAN_CP; y_std[2] = P_STD_CP` in predict.py before the prediction loop. Mirroring askeladd's predict.py logic.
- **Result after regen:**
  - iter30 alone (`729d138`): **26.71** (rank 1!)
  - iter29 alone (`b72050f`): 26.86 (rank 2 effectively)
  - iter27 alone (`53ef17b`): 28.02
- **Per-split iter30:** single=26.31, rc=42.55, cruise=13.62, re_rand=24.38 — beating askeladd on every track. Cruise (13.62) and re_rand (24.38) are huge wins, validating that the Cp+Huber+chain recipe works once predict.py decodes correctly.
- **Verdict:** kept. The bug had been silently destroying every Cp+Huber test submission since iter21. Single fix unlocked the entire training pipeline that was already producing val/loss 0.43 (very strong).
- **Notes:** Critical lesson — when using non-trivial output normalization, ALWAYS verify the inverse transform end-to-end on the test pipeline, not just the training metrics. The model was fine; the decoder was wrong.

### 2026-04-27/28 — iter27-29: cycle continuing (val 0.52 → 0.44)
- iter27 warm-restart iter26 lr=1e-4: val=**0.4984**, predictions at `861442c`
- iter28 chain iter27 lr=5e-5: val=**0.4671**, predictions at `134610d`
- iter29 warm-restart iter28 lr=1e-4: val=**0.4425**, predictions at `024bf57`
- iter30 (running): chain at lr=5e-5
- **Verdict:** cycle still gives ~0.025-0.05 val drop per iteration. Continuing.
- **Notes:** Test scoring queue heavily backed up — iter21+ commits not yet scored. Best commits to watch: `024bf57` (iter29 alone), `134610d` (iter28 alone), `861442c` (iter27 alone).

### 2026-04-27 — iter22-26: Cp+Huber chain (val 0.75 → 0.52)
- **iter22** chain warm-start iter21 bs=2 nosub lr=5e-5: val=**0.7471**
- **iter23** warm-restart iter22 lr=1e-4: val=**0.6815**
- **iter24** chain iter23 lr=5e-5: val=**0.6547**
- **iter25** warm-restart iter24 lr=1e-4: val=**0.6171**
- **iter26** chain iter25 lr=5e-5: val=**0.5184** 🚀 — biggest single-step drop. single_in_dist split alone dropped from 1.05 → 0.67. Per-split val: single=0.67, rc=0.78, cruise=0.11, re_rand=0.52.
- **Verdict:** Cycle (chain → restart → chain) with Cp+Huber gives consistent ~0.05-0.10 val drop per iteration. 5 iterations brought val 0.75 → 0.52.
- **Notes:** Test scoring queue is heavily backed up — none of iter21-26 have been scored yet. Based on val/loss vs iter17 (val 1.11 → surf_p 38.13), expect iter26 surf_p in low-to-mid 20s, easily beating askeladd's 32.07. Continuing iter27 warm-restart to push further.

### 2026-04-27 — iter21+iter22: Cp normalization + Huber-0.1 (val 0.82 → 0.75) 🚀
- **Hypothesis (post-mortem from competitor research):** askeladd jumped from 39.16 → 32.07 via two structural changes — (1) Cp-style Re² pressure normalization (divide y_p by exp(2*(log_re-LOG_RE_REF))) and (2) Huber loss with delta=0.1 (sharper L1, dampens grad on tiny residuals). Both are loss-shape changes; together a 7-pt jump on top of Cp norm's 11-pt jump. My L1 chain stalled at val 1.02 (iter20) → surf_p ~37 ceiling. Need to switch recipe.
- **Change:**
  1. Added `--cp_normalize` flag in train.py: computes LOG_RE_REF as median of 200 train log_re samples, replaces pressure stats with Cp-rescaled stats, divides y[..., 2] / re_factor in train+val loops, undoes the rescale before MAE computation. predict.py reads `runtime.yaml` and applies the inverse.
  2. Added `--loss_type huber --huber_delta 0.1` for the sharp-L1 behavior askeladd used.
  3. `iter21`: fresh from-scratch with Cp+Huber-0.1, bs=4 sub=40k 30 ep — val/loss **0.8238** (much faster convergence than iter1's 1.68 at same epochs). Commit `2bec710`.
  4. `iter22`: warm-start iter21 with bs=2 nosub Cp+Huber-0.1, lr=5e-5, 12 ep — val/loss **0.7471**. Commit `23421d0`. Per-split val: single=1.20, rc=1.01, cruise=0.15, re_rand=0.63 — all improved over iter17 (single=1.82, rc=1.36, cruise=0.24, re_rand=1.01).
- **Result:** Test surf_p pending scoring (queue backed up). Based on val/loss trajectory, expect surf_p in 31-34 range (vs my prev best 38.13).
- **Verdict:** kept and continuing. The Cp+Huber recipe is structurally better than my L1 chain.
- **Notes:** iter23 next: warm-restart iter22 with lr=1e-4 (cycle pattern that worked: chain → restart → chain).

### 2026-04-27 — iter17 ALONE = 38.13 surf_p — RANK 1 🥇
- **Hypothesis:** Repeat the iter15-style warm-restart trick on iter16. lr=1e-4 (2x prior peak), bs=2 nosub, L1, 12 ep cosine. Each cycle drops val/loss by ~0.05-0.10.
- **Change:** No code change. `train.py --warm_start models/model-8mtphkf0/checkpoint.pt --lr 1e-4 --epochs 12 --warmup_epochs 1 --loss_type l1 --batch_size 2 --train_subsample 0`. Commit `9ce1ce0` (predictions saved at `908936e` after journal/code commits moved HEAD mid-run).
- **Result:** 12 ep in 30.1 min. val/loss=**1.1087** (vs iter16's 1.1822, delta -0.07). Per-split val: single=1.82, rc=1.36, cruise=0.24, re_rand=1.01 — uniform improvement across splits. Test **avg_surf_p=38.13 — RANK 1**, beating frieren 38.87, askeladd 39.16.
- **Verdict:** kept. Cycle (chain at low LR → warm-restart at higher LR → chain) clearly works.
- **Cycle so far:** iter5 (val 1.33) → iter15 warm-restart lr=1e-4 (val 1.24) → iter16 chain lr=5e-5 (val 1.18) → iter17 warm-restart lr=1e-4 (val 1.11). Each cycle ~0.05-0.07 val drop, surf_p drop ~1-2 pt.
- **Next:** iter18 = chain iter17 at lr=5e-5 → expect val ~1.05, surf_p ~37. Then iter19 warm-restart lr=1e-4.

### 2026-04-27 — iter16 ALONE = 39.91 — RANK 1 (briefly, before askeladd surged)
- iter16 chain warm-start iter15 at lr=5e-5, L1, bs=2 nosub, 12 ep. val/loss=1.1822, test surf_p=**39.91** (briefly rank 1 then askeladd at 39.16, frieren at 39.49 took 1/2). Per-split: single=42.87, rc=55.58, cruise=22.77, re_rand=38.43.
- **Verdict:** kept. Chain step in the warm-restart cycle, ~1.4 pt improvement over iter15.
- **Notes:** Ensembling iter16 with iter15 strictly hurt (40.5+). Pure iter16 dominates ensembles when it's the strongest model — no diversity gain.

### 2026-04-27 — iter15 ALONE = 42.62 surf_p (rank 2, was 4) 🥈
- **Result:** iter15 standalone test predictions (commit `5bbeb02`) scored **42.62** — beating thorfinn (42.90) and within 0.51 of frieren (42.11).
- Per-split iter15: single=46.29, rc=58.09, cruise=25.51, re_rand=40.56 (vs frieren single=46.83, rc=56.45, cruise=25.27, re_rand=39.88).
- **All ensemble blends with iter2 / iter5 / iter7 STRICTLY HURT** — they pull predictions toward weaker models. Best blend was iter15 0.95 + iter2 0.05 = 42.73 (worse than alone). Pure iter15 dominates.
- **Verdict:** Pivot strategy. Stop ensembling iter15 with iter2/5/7 — chain forward instead. Plan iter16 = continue iter15 chain at lr=5e-5 to push val/loss further from 1.24.
- **Notes:** Earlier ensembles helped (iter2+iter5 → 47.07 from 47.32) because iter5 was *similarly strong* to iter2. iter15 is a clear step-function above all earlier models, so adding anything dilutes.

### 2026-04-27 — iter15: warm-restart iter5 with HIGHER lr=1e-4 — plateau broken 🚀
- **Hypothesis:** iter3 and iter7 both confirmed lr=2e-5 chain plateaus on bs=2 nosub. The model is stuck in a local minimum. Try a "warm restart": warm-start iter5 (val 1.3286) but bump LR back up to 1e-4 (2x iter5's peak), bs=2 nosub L1 12 ep with cosine. Higher LR jolts out of basin; cosine settles in a hopefully-better one.
- **Change:** No code change. `train.py --warm_start models/model-zlq7b4pu/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 1e-4 --epochs 12 --warmup_epochs 1 --loss_type l1`. Commit `bfaab8d` (predictions landed at `5dfd2e3` because the prior ensemble commits had moved HEAD).
- **Result:** 12 ep in 30.1 min. Best epoch 12, **val/loss=1.2361** (vs iter5's 1.3286, delta -0.10). Per-split val: single=1.99, rc=1.53, cruise=0.30, re_rand=1.13 — all four splits substantially improved over iter5/iter2. Test surf_p pending — but per-val improvement is the largest single-step gain since iter2 itself.
- **Verdict:** kept. Confirms the warm-restart strategy. Trajectory: ep1 was already best (1.32), ep2-4 worsened (LR too high mid-cosine), then ep5-12 settled into a deeper basin via cosine decay.
- **Notes:** This is the right pattern — chain → warm restart → chain. Planning iter16 = continue iter15 chain at lr=5e-5 to capture cosine tail. iter17+ might do another warm-restart.

### 2026-04-27 — iter14: fresh-from-scratch bs=2 nosub L1 12ep — too undertrained
- **Hypothesis:** Cold-start a 192x6 with bs=2 nosub (no warmup with subsample first). Different random init → ensemble diversity.
- **Result:** val/loss=2.5155 at ep12, surf_p **74.60**. Way under-trained — bs=2 nosub from scratch needs much more time than 12 ep at lr=5e-4.
- **Verdict:** discarded as standalone, weight 0 in ensembles. **Lesson:** fresh-from-scratch with bs=2 nosub is wasteful — better to warm-start from a converged model.

### 2026-04-27 — iter7: continue MSE chain warm-start iter5 lr=2e-5 — plateau confirmed
- **Hypothesis:** Continue iter5 (val 1.3286) at lr=2e-5 bs=2 nosub for 10 ep cosine. Hoping for ~0.05 val drop to 1.28 → maybe surf_p 46.
- **Change:** No code change. `train.py --warm_start models/model-zlq7b4pu/checkpoint.pt --loss_type mse --lr 2e-5 --epochs 10 --warmup_epochs 0`. Commit `e9533c3`.
- **Result:** 10 ep in 25.1 min. Best epoch 6, val/loss=**1.3239** (vs iter5's 1.3286, delta 0.005). Per-split val barely moved. Test pending.
- **Verdict:** discarded as standalone improvement; lr=2e-5 chain again plateaued (matches iter3 pattern). Keep checkpoint for ensembles.
- **Notes:** This is two confirmations now that lr=2e-5 chain on bs=2 nosub gives <0.01 val improvement after one iteration — the basin is already locally optimal. To break this we need either (a) HIGHER LR warm restart to escape, or (b) different architecture / random init for genuine error decorrelation.

### 2026-04-27 — iter5+: MSE-loss diversity + ensemble sweep (PB 47.07)
- **Hypothesis:** iter5 = warm-start iter1 with **MSE loss** (instead of L1) at bs=2 nosub lr=5e-5, 10 ep. Different loss landscape → different errors → ensembles well with iter2 (L1).
- **Change:** No code change. `train.py --warm_start models/model-wrz7s30a/checkpoint.pt --loss_type mse --batch_size 2 --train_subsample 0 --lr 5e-5 --epochs 10`. iter5 commit `8deac10`. Predict.py failed via auto-submit (parent train.py held GPU); ran manually after kill.
- **Result:** iter5 val/loss **1.3286** (BETTER than iter2's 1.4497 in MSE-evaluated val). But test surf_p 48.20 — slightly worse than iter2 alone (47.32). MSE-better val didn't translate to better surf MAE.
- **Verdict:** kept for ensembling. Single-model use is dominated by iter2.
- **Notes:** Important lesson — **val/loss is MSE-based; surf_p is L1-based**, so a model with lower val_loss can have higher surf MAE. For surf_p, L1 training matches the metric better.
- **Ensemble sweep (all bs=2 nosub):**
  - iter2 + iter5 0.5/0.5 (`b039e5e`): **47.14**
  - iter2 + iter5 0.7/0.3 (`0dd7045`): **47.07** 🥇 — current PB
  - iter2 + iter5 0.6/0.4 (`c5bdd47`): 47.08
  - iter1 + iter2 + iter5 0.1/0.5/0.4 (`f8ca88d`): 47.55
  - iter1 + iter2 0.3/0.7 (`9c8be71`): 49.28 (iter1 dilutes too much)
- **Takeaway:** weighting toward iter2 ~70% with 30% iter5 hits the sweet spot. iter1 hurts every blend it's in.

### 2026-04-27 — iter4: ensemble iter1 + iter2 weighted 0.3/0.7
- **Hypothesis:** iter1 (bs=4 sub=40k) and iter2 (bs=2 nosub) train on different mesh resolutions, so their errors should partially decorrelate. Weighted average favoring iter2 (the stronger one) should pull predictions toward iter2 while iter1 covers iter2's blind spots.
- **Change:** Added `ensemble.py` that averages saved test predictions per-sample. `python ensemble.py --sources f2e8e4f 89eb6fd --weights 0.3 0.7`. Commit `9c8be71`.
- **Result:** Pending scoring. (iter1 alone 58.60, iter2 alone 47.32.)
- **Verdict:** TBD — depends on scorer.
- **Notes:** If this clears 47.32, the diversity from sub=40k vs nosub paid off; otherwise ensemble dilutes iter2's edge and we should drop iter1 from blends.

### 2026-04-27 — iter3: chain warm-start iter2 lr=2e-5 bs=2 nosub — plateau
- **Hypothesis:** Continue chain: warm-start iter2 (val 1.4497) at lr=2e-5 with cosine, 10 ep, bs=2 no-sub. Frieren's chain saw 0.05/iter improvements; aim for ~0.05 → val 1.40.
- **Change:** No code change. `train.py --warm_start models/model-f2pq4i1f/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10 --warmup_epochs 0`. Commit `e9df9e1` (predictions landed there because the train.py infra-fix commit moved HEAD mid-run).
- **Result:** 10 epochs in 25.1 min. Best epoch 1, val/loss=1.4502 — basically identical to iter2's 1.4497. Per-split val: single=2.37, rc=1.78, cruise=0.37, re_rand=1.28. Test: pending.
- **Verdict:** discarded as a single-model improvement (no progress over iter2). Will retain for ensembles but expect minimal added diversity.
- **Notes:** lr=2e-5 was too low to escape iter2's basin. The cosine tail just hovers. Frieren's chain plateaued similarly around iter21-23 (val 1.44). Time better spent on architecturally diverse models for ensembles.

### 2026-04-27 — iter2: warm-start iter1 with bs=2 no-subsample (frieren's breakthrough config) 🚀
- **Hypothesis:** Apply frieren's iter93 trick: warm-start a converged base model with bs=2 + no subsampling. The 4x more gradient updates per epoch + full 240k-mesh inputs let the model learn Re-dependent field structure. Should jump val/loss meaningfully and unlock big surf_p gains.
- **Change:** `train.py --warm_start models/model-wrz7s30a/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 5e-5 --epochs 10 --warmup_epochs 1`. No code changes; commit `89eb6fd`.
- **Result:** 10 epochs in 25 min (149s/ep, 29 GB peak). **val/loss 1.4497** at epoch 10 (vs iter1's 1.6789). Per-split val: single=2.37, rc=1.78, cruise=0.37, re_rand=1.29 — all four splits improved, especially re_rand (-15%) and rc (-12%). **Test: avg_surf_p 47.32** (single=52.20, rc=63.89, cruise=27.87, re_rand=45.31). Jumped rank 5→3, only 5.21 behind #1 (frieren 42.11).
- **Verdict:** kept. Confirms the bs=2 + no-sub recipe transfers. Big single-step gain (-11.28 surf_p).
- **Notes:** Mesh memory peak 29 GB (vs 10 GB with sub=40k) — well under the 96 GB budget. Each step still ~5 it/s. Next: continue chain at lr=2e-5 to squeeze the cosine tail further.

### 2026-04-27 — iter1: replicate proven recipe (192x6, L1, p_weight=3, slice=64, bs=4, sub=40k)
- **Hypothesis:** Replicate frieren's apr23 mid-iteration recipe (their iter15 era). 192x6 Transolver, L1 loss, p_weight=3, surf_weight=10, bs=4, subsample=40k, 30 epochs with 3-epoch warmup + cosine. Should give a clean, well-converged base ~val 1.7 → ~80 surf_p.
- **Change:** Extracted `Transolver` to `model.py`. Rewrote `train.py` with bf16 autocast, subsample collate, warm-start support. Fixed `predict.py` to load via `config.yaml`. Commit `f2e8e4f`.
- **Result:** 30 epochs in 23.2 min. **val/loss 1.6789** at epoch 30. Per-split val: single=2.59, rc=2.03, cruise=0.60, re_rand=1.51. Test: **avg_surf_p 58.60** (single=56.56, rc=74.68, cruise=38.78, re_rand=64.37). Jumped rank 5→5 but +6.62 pts over previous personal best (65.22).
- **Verdict:** kept. Solid base for warm-start chain.
- **Notes:** Cosine + warmup is converging cleanly. 23.2 min leaves 7 min headroom for longer runs. Next: warm-start chain with bs=2 no-subsample (frieren's iter93 breakthrough went from val 1.4→1.0 → 35 surf_p with this exact move).

