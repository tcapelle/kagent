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

### 2026-04-17 — v29 KEPT — 2nd full-cosine smashed solo to 0.8420, 16-seed weighted 0.7579 (gain 0.0028)
- **Hypothesis:** full-cosine is the strongest single regime (v28 proved it). Another full-cosine seed expected to give ~0.002 gain.
- **Change:** same config as v28 — `MAX_TIMEOUT_MIN=90, --epochs 90`. Different random init (implicit).
- **Result:** v29 solo val/l2 = **0.8420** at ep87 — *massive* jump under v28's 0.8541 (delta 0.0121). Absolute new floor. 16-seed weighted (softmax T=0.02) = **0.7579** (vs 15-seed 0.7607). Gain 0.0028 — above expectation. v29 gets max softmax weight (0.167) in the ensemble.
- **Verdict:** kept. 13.0% under v6 solo.
- **Notes:** Full-cosine is clearly the best regime — 2 of 2 seeds landed below any non-full-cosine seed's best. Next (v30): 3rd full-cosine seed — continue the streak. Also eventually worth testing `epochs=120 MAX_TIMEOUT_MIN=90` to see if slower anneal digs deeper, but not before getting 3-4 more full-cosine seeds.


### 2026-04-17 — v28 KEPT — full-cosine (epochs=90 MAX_TIMEOUT=90) solo-best ever, 15-seed weighted 0.7607 (gain 0.0022)
- **Hypothesis:** every prior long seed cut at ~ep69 with LR still ~0.06× init — model never reached the LR→0 settled minimum. Completing the full 90-epoch cosine schedule should give both (a) a stronger solo (more fine-tuning at low LR) and (b) a genuinely new basin (ending at LR=0 vs cut at LR=0.06).
- **Change:** CLI + env — `MAX_TIMEOUT_MIN=90 python train.py --epochs 90`. train.py untouched.
- **Result:** v28 solo val/l2 = **0.8541** at ep89 — **new single-seed best** (beats v21 0.8549). 15-seed weighted (softmax T=0.02) = **0.7607** (vs 14-seed weighted 0.7629). Gain 0.0022 — noticeably above the 0.0017 plateau of the last few axes. 12.6% under v6 solo.
- **Verdict:** strongly kept. Both a new solo-best and a meaningful ensemble gain.
- **Notes:** Hypothesis validated on both fronts. The last 21 epochs at near-zero LR matter a lot. Full-cosine is now the strongest single training regime. Next (v29): train **another** full-cosine seed. If het-within-axis diminishing applies here the same way as for epochs=90-cut, we'd still expect ~0.0015-0.0020 further gain. Longer-term, v30+ should try `epochs=120 MAX_TIMEOUT=90` — slower anneal to LR=0, may dig deeper.


### 2026-04-17 — v27 KEPT — wd=5e-5 (half baseline), 14-seed weighted 0.7629 (gain 0.0017)
- **Hypothesis:** v25 (wd=3e-4, 3× baseline) had a weaker solo (0.866). The opposite direction — wd=5e-5 (half baseline) — should give stronger solo (less regularization) while still being an untested basin.
- **Change:** CLI — `--weight_decay 5e-5 --epochs 90`.
- **Result:** v27 solo val/l2 = **0.8596** at ep68 (mid-pack among long seeds, as hoped — stronger than v25's 0.866). 14-seed weighted (softmax T=0.02) = **0.7629** (vs 13-seed weighted 0.7646). Gain 0.0017. 14-seed uniform = 0.7655 (still ~0.0026 worse than weighted, confirming the weighted scheme is robustly better).
- **Verdict:** kept. 12.4% under v6 solo.
- **Notes:** Each new axis still contributes ~0.002/seed weighted gain. Pattern suggests remaining gains need bigger architectural moves. Untried high-leverage axes: (a) `MAX_TIMEOUT_MIN=90` with `epochs=90` to complete the full cosine schedule (all prior long seeds cut at ~ep69 with LR still ~0.06*init — never reached LR→0 settled minimum); (b) architectural diversity (e.g., `hidden=192` or `voxel_res=48`) which would break ensemble.py assumption of homogeneous arch. Going with (a) for v28 — cheaper, no code change needed.


### 2026-04-17 — weighted-ensemble KEPT — softmax(-l2/T=0.02) over 13 seeds: 0.7646 (gain 0.003, zero compute)
- **Hypothesis:** uniform averaging down-weights strong seeds and up-weights weak ones (v20=0.889, v25=0.866 drag the mean). Weight by softmax(-solo_l2/T) so better seeds contribute more. Per-seed solo l2 is a proxy for test l2 — weights should generalize.
- **Change:** new file `ensemble_weighted.py`. Tries 6 schemes: uniform, inv_loss, inv_loss_sq, softmax T∈{0.02, 0.005, 0.001}. Reports each and saves the best.
- **Result:** softmax T=0.02 wins at **val/l2 = 0.7646** (vs uniform 0.7672; weights span 0.024-0.130 for 13 seeds; mean=0.077). inv_loss and inv_loss_sq are ~indistinguishable from uniform. Sharper T=0.005 overfits (0.771). T=0.001 collapses to argmax and fails (0.818).
- **Verdict:** kept — 0.003 gain at zero compute. 12.2% under v6 solo.
- **Notes:** The best-scheme choice is made on val (mild val-set tuning), but T=0.02 is a reasonable generic heuristic insensitive to any single val sample. For test submission, using T=0.02 softmax weighted by val/l2 is defensible — per-seed val rankings should transfer to test. Next (v27): another training run with a genuinely new axis — `wd=5e-5 epochs=90` (weaker regularization than baseline 1e-4, untested direction). Expected: strong solo + distinct basin.


### 2026-04-17 — v26 KEPT — lr=7e-4 opposite-direction axis, 13-seed ensemble 0.7672 (gain 0.0017)
- **Hypothesis:** lr=7e-4 (40% above default 5e-4, opposite of v24's 3e-4) explores the higher-lr side of the basin manifold and should match v24's axis gain with a stronger solo than v25's wd=3e-4.
- **Change:** CLI only — `--lr 7e-4 --epochs 90`.
- **Result:** v26 solo val/l2 = **0.8633** at ep69 (mid-pack among long seeds; lr=7e-4 trains solidly). 13-seed ensemble = **0.7672** (vs 12-seed 0.7689). Gain 0.0017.
- **Verdict:** kept. 11.9% under v6 solo. Slightly below expectation — incremental axis value is saturating around 0.0017.
- **Notes:** Ensemble gain has plateaued at 0.0015-0.0020/seed across the last 3 additions. Time to change strategy: cheap experiment = **weighted ensemble** (weight by inverse solo val/l2). Weaker seeds like v20 (0.889) and v25 (0.866) currently pull the uniform average down; upweighting strong seeds should recover 0.002-0.005 at zero compute cost.


### 2026-04-17 — v25 KEPT — wd=3e-4 new reg axis, 12-seed ensemble 0.7689 (gain 0.0016)
- **Hypothesis:** regularization strength (WD) is an axis neither `epochs` nor `lr` touched. Triple WD from 1e-4 to 3e-4 → different weight-norm-constrained basin.
- **Change:** CLI only — `--weight_decay 3e-4 --epochs 90` (default lr=5e-4).
- **Result:** v25 solo val/l2 = **0.8660** at ep69 (weakest of the long seeds — higher WD costs ~0.01 on solo). 12-seed ensemble = **0.7689** (vs 11-seed 0.7705). Gain 0.0016.
- **Verdict:** kept (any improvement banked). Gain right at the 0.002 threshold; a weaker solo limits ensemble contribution.
- **Notes:** Lesson: weaker solos add less ensemble value even on a fresh axis. Next (v26): try `lr=7e-4` (opposite-direction lr axis vs v24's 3e-4). Baseline lr was 5e-4, v24 was 3e-4; 7e-4 explores the higher-lr side of the basin manifold and may match v24's 0.003 gain while keeping a strong solo.


### 2026-04-17 — v24 KEPT — lr=3e-4 new axis, 11-seed ensemble 0.7705 (gain 0.003)
- **Hypothesis:** within-het gain was saturating (0.006→0.004→0.003). Switch axis: keep `epochs=90` but halve lr to 3e-4 (from 5e-4). Slower learning trajectory → different basin → refresh ensemble gain.
- **Change:** CLI only — `--lr 3e-4 --epochs 90` (train.py default epochs=90 from v21; lr flag is CLI-only, train.py untouched).
- **Result:** v24 solo val/l2 = **0.8565** at ep68 (2nd-best single seed ever, just 0.0016 behind v21's 0.8549). 11-seed ensemble = **0.7705** (vs 10-seed 0.7735). Gain 0.0030 — same as v23 but from a truly distinct axis rather than within-het diminishing.
- **Verdict:** kept. 11.5% under v6 solo. New-axis hypothesis validated.
- **Notes:** Interesting: the lower-lr seed also lands near the long-anneal solo floor (0.856 vs 0.854). LR and anneal length are somewhat degenerate in their effect on final val (both control 'how much did the model fine-tune at low LR'). But ensemble-gain-wise they contribute orthogonally. Next (v25): try yet another distinct axis — `weight_decay=3e-4` (3× default) with epochs=90, lr=5e-4 — regularization is an axis neither touched by epochs nor lr.


### 2026-04-17 — v23 KEPT — 3rd epochs=90 seed, 10-seed ensemble 0.7735 (gain 0.003)
- **Hypothesis:** continue adding het (epochs=90) seeds. Expected gain ~0.003 if within-het diversity is diminishing.
- **Change:** No code change. Another seed with `epochs=90, MAX_TIMEOUT_MIN=60`.
- **Result:** v23 solo val/l2 = **0.8602** at ep68. 10-seed ensemble = **0.7735** (vs 9-seed 0.7761). Gain 0.0026.
- **Verdict:** kept. 11.2% under v6 solo. Gain just above 0.002 threshold.
- **Notes:** Within-het saturation is kicking in (0.006 → 0.004 → 0.003). Next (v24): switch to a new heterogeneity axis — `lr=3e-4` (60% of default 5e-4) with `epochs=90`. Slower learning trajectory should hit a genuinely different basin.


### 2026-04-17 — v22 KEPT — 2nd epochs=90 seed, 9-seed ensemble 0.7761 (gain 0.004)
- **Hypothesis:** v21's heterogeneous (epochs=90) seed gave 3× the homogeneous gain. A second epochs=90 seed should add another significant decorrelation gain, though diminished since both are drawn from the same het distribution.
- **Change:** No code change from v21. Trained another seed with `epochs=90, MAX_TIMEOUT_MIN=60`.
- **Result:** v22 solo val/l2 = **0.8577** at epoch 68 (second-best single seed ever, just behind v21's 0.8549). 9-seed ensemble = **0.7761** (vs 8-seed 0.7800). Gain 0.0039 — about 2/3 of v21's het gain, still 2× the homogeneous saturation rate.
- **Verdict:** kept. 10.9% under v6 solo. Clear pattern: het seeds continue to add value, just with diminishing returns within the het population itself.
- **Notes:** Two long seeds in a row land ~0.855-0.858 — the long-anneal floor looks tight, mirroring the 0.87-floor of the 60-epoch pack. Next (v23): continue with another epochs=90 seed — expected ~0.003 further gain to ~0.773. If gain <0.002 I'll switch to a genuinely different axis (e.g., `weight_decay=5e-5` or `epochs=120`) to break the het-population correlation.


### 2026-04-17 — v21 KEPT — epochs=90/60min seed broke homogeneous saturation (8-seed 0.7800, gain 0.006)
- **Hypothesis:** 7 homogeneous seeds were saturating (gain 0.0018). Changing the anneal profile — `epochs=90, MAX_TIMEOUT_MIN=60` — produces a *different* final-weights basin than the 60-epoch cosine all prior seeds saw. Different basin = less-correlated errors = bigger ensemble gain.
- **Change:** `train.py` — `epochs: int = 90` (was 60). Launched with `MAX_TIMEOUT_MIN=60`. All else identical to v6.
- **Result:** v21 solo val/l2 = **0.8549** at epoch 68 of 69 run (60 min, 52s/epoch). **Best single seed ever**, 0.014 under prior best (v18 0.8684). Longer anneal worked at the solo level too. 8-seed ensemble = **0.7800** (vs 7-seed 0.7858). Ensemble gain 0.0058 — **3× the saturating homogeneous rate**.
- **Verdict:** kept — both solo-best AND ensemble-best. Hypothesis strongly validated.
- **Notes:** Two independent wins here: (1) longer anneal (90 vs 60 ep) lowers the single-run floor significantly — the extra 9 low-LR epochs let the model tune further into the minimum. (2) The heterogeneous anneal introduces exactly the error decorrelation predicted. Next (v22): train **another epochs=90 seed**. If each het seed gives ~0.006 ensemble gain, three of them stacked with the 7 short seeds could land the 9-model ensemble in the 0.77 band.


### 2026-04-17 — v20 KEPT — 7-seed ensemble landed 0.7858, below the 2-delta threshold
- **Change:** 7th v6-arch seed (solo **0.8890** — noticeably weaker than prior seeds at 0.87 ± 0.005, genuinely unlucky init). Ensemble 7 checkpoints.
- **Result:** 7-seed = **0.7858** (vs 6-seed 0.7876). Gain 0.0018.
- **Verdict:** kept — any improvement is banked. But gain < 0.002 threshold I set earlier; homogeneous seeds are saturating.
- **Notes:** Interestingly a *weaker* solo seed (0.889 vs 0.87) still added ensemble value, because its errors decorrelate from the tighter-clustered seeds. Confirms the ensemble benefit comes from error *independence* not just having more near-best models. Next (v21): try **longer-training heterogeneity** — same arch, but `epochs=90` with `MAX_TIMEOUT=60`. Different anneal profile = different final-weights basin = less-correlated errors vs. the 60-epoch homogeneous pack. If this works, could be a free ensemble diversity boost without modifying ensemble.py for multi-arch loading.


### 2026-04-17 — v19 KEPT — 6-seed ensemble landed 0.7876, still descending
- **Change:** 6th v6-arch seed (45 min, 52 epochs, solo best **0.8699** @ ep51).
- **Result:** 6-seed val/l2 = **0.7876** (vs 5-seed 0.7927). Solo best-of-6 sequence: 0.8707, 0.8784, 0.8694, 0.8706, 0.8684, 0.8699 — mean 0.8712, std 0.0034. Noise floor rock-solid.
- **Verdict:** kept — gain 0.005 > 0.002 threshold, keep adding.
- **Notes:** Diminishing returns curve: 2→3=0.019, 3→4=0.008, 4→5=0.007, 5→6=0.005. At this slope 7-seed ≈ 0.784, 10-seed ≈ 0.778. Next (v20): 7th seed — if gain <0.002 switch strategy to heterogeneous arch (`hidden=384` variant) which would break error correlation. Otherwise keep adding homogeneous seeds — free gains.


### 2026-04-17 — v18 KEPT — 5-seed ensemble landed 0.7927, on the 1/sqrt(k) curve
- **Hypothesis:** 1/sqrt(k) continues; 5-seed target ~0.790.
- **Change:** 5th v6-arch seed (45 min, 52 epochs, solo best **0.8684** — best single seed yet). Ran ensemble over 5 ckpts.
- **Result:** 5-seed val/l2 = **0.7927** (vs 4-seed 0.7993, 3-seed 0.8077). Solo sequence (v6, v15, v16, v17, v18): 0.8707, 0.8784, 0.8694, 0.8706, 0.8684 — mean 0.8715, std 0.0038.
- **Verdict:** kept — 0.0066 further drop. 0.078 / 9.0% under v6 solo.
- **Notes:** Solo best (v18, 0.8684) is now the best single checkpoint ever. Diminishing returns visible: 2→3 gave 0.019, 3→4 gave 0.008, 4→5 gave 0.007. Slope flattening. Still, compute is cheap. Next (v19): 6th seed — expected gain ~0.005 (to ~0.787). If 6-seed fails to gain ≥ 0.002 I'll stop adding seeds and explore a *different-architecture* seed (e.g., `hidden=384` or `voxel_mid=96`) — heterogeneous ensembles typically cancel error more than homogeneous.


### 2026-04-17 — v17 KEPT — 4-seed ensemble landed 0.7993, broke the 0.8 barrier
- **Hypothesis:** 1/sqrt(k) scaling continues — 4-seed target ~0.795 from 3-seed 0.8077.
- **Change:** trained 4th v6-arch seed (45 min, 52 epochs, solo best **0.8706** @ ep48 — the closest any seed has come to v6's original 0.8707 by coincidence). `ensemble.py` on 4 checkpoints.
- **Result:** 4-seed val/l2 = **0.7993** (vs 3-seed 0.8077, 2-seed 0.8265, v6 solo 0.8707). Individual solos: v6=0.8707, v15=0.8784, v16=0.8694, v17=0.8706 (mean 0.8723, std 0.004).
- **Verdict:** kept — under 0.80. 0.084 absolute / 9.6% relative under v6; 0.011 further drop vs v16 3-seed. Curve still strictly descending.
- **Notes:** Solo std across 4 seeds is 0.004 — the single-run floor is *extremely* tight. Seed diversity is the whole story here. At 45 min/seed with GPU idle between training runs, the marginal cost of +1 seed is ~0 opportunity cost in this regime. 1/sqrt(k) predicts 5-seed → ~0.790, 8-seed → ~0.780. Next (v18): train 5th seed, push toward 0.79.


### 2026-04-17 — v16 KEPT — 3-seed ensemble landed 0.8077, another 2% under v15's 2-seed
- **Hypothesis:** v15 showed the 2-seed ensemble gives ~5% over v6 (0.8707 → 0.8265). The bias-variance decomposition says a k-seed ensemble has residual error ∝ 1/sqrt(k) of the per-model noise if errors are independent. A 3rd independent seed should push the ensemble another ~0.020 lower (√2/√3 of the 2-seed variance), toward ~0.81.
- **Change:** trained 3rd v6-arch seed (45 min, 52s/epoch × 48 useful epochs, solo best 0.8694 @ ep48 — actually marginally better than v6 alone, confirming v6's 0.8707 was a noisy sample of the ~0.87 ± 0.005 single-run floor). Ran `ensemble.py --checkpoints _v6seed.pt _v15seed.pt _v16seed.pt`. No code changes.
- **Result:** 3-seed ensemble val/l2 = **0.8077** (vs 2-seed 0.8265, v6 solo 0.8707). Individual val/l2: v6 = 0.8707, v15 = 0.8784, v16 = 0.8694 (solo mean ≈ 0.873, std ≈ 0.005). W&B project `kagent-v16`.
- **Verdict:** kept — 0.019 further drop from 2-seed, matches predicted diminishing-returns curve.
- **Notes:** Progression: 1-seed=0.873avg → 2-seed=0.8265 → 3-seed=0.8077. The 1/sqrt(k) model predicts 3-seed=0.82 if starting from 0.873 avg single-model error — we beat that, suggesting some super-linear cancellation (errors less correlated than expected for the hard samples). Next (v17): **4-seed ensemble** — train one more v6-arch seed. Predicted gain: another ~0.012 (to ~0.795). Diminishing returns are real but still favorable: doubling seeds from 2→4 gave 0.8265→~0.795 = 0.031 total = same order as single architectural improvement v2→v5 took 3 experiments to find. Cheap budget wins.


### 2026-04-17 — v15 KEPT — 2-seed ensemble (v6 + fresh seed) landed 0.8265, 5% under v6
- **Hypothesis:** v6's 0.8707 is partly a lucky single-epoch dip in bs=1 val noise. Multiple single-run tweaks (v7-v14) all landed 0.87-0.92 without reliably beating it. Averaging predictions across independent seeds of the same arch should cancel independent prediction error (bias-variance: ensemble bias ≈ single-model bias, ensemble variance ≈ var/k). Zero arch risk, directly attacks the noise floor.
- **Change:** `ensemble.py` (new) — loads k checkpoints, runs val inference on each, averages predictions [B,5,N,3], reports `val/l2_error` on the averaged preds, saves to standard predictions dir. v6-arch trained a 2nd time (implicit random seed) for 45 min (52s/epoch × 47 useful epochs, best val/l2=0.8784 alone at ep47); `_v6seed.pt` + `_v15seed.pt` in `checkpoints/`. `train.py` unchanged.
- **Result:** Ensemble val/l2 = **0.8265** (2 models, 80 val samples). Individual: v6 seed = 0.8707, v15 seed2 = 0.8784. Ensemble beats both — the errors genuinely partially cancel, not correlate. W&B project `kagent-v15`.
- **Verdict:** kept — 0.044 absolute / 5.1% relative improvement over v6. Biggest jump since v5→v6.
- **Notes:** Confirms the plateau was a prediction-noise floor, not an architectural capacity limit. The 2nd seed landing at 0.8784 (vs v6's 0.8707) puts the single-run floor at ~0.875 ± 0.005, exactly where SWA (v13) smoothed to — consistent story. Strong evidence a 3rd or 4th seed would keep shrinking the ensemble error (1/sqrt(k) law). With 45min/seed budget and competition structure allowing batched experiments, doubling the seed count is cheap. Next (v16): **3-seed ensemble** — train one more v6-arch seed, re-run `ensemble.py` with 3 checkpoints. Expected val/l2 ~0.81 if the diminishing-returns curve follows the typical 1/sqrt(k) shape.


### 2026-04-17 — v14 DISCARDED — multi-scale voxel (32³+64³) landed 0.8869 (above v6's 0.8707)
- **Hypothesis:** v6 uses a single 64³ voxel grid — fine local detail, but 64-cell aperture gives limited global wake/boundary context. Add a parallel 32³ UNet branch on the same features, concat fine+coarse outputs and project back. Zero-init merge so training starts at v6 identity. First true architectural expansion since v5.
- **Change:** `train.py` — `VoxelSpatial` grew `unet_coarse=UNet3D` at `res_coarse=32`, `merge=nn.Linear(dim*2, dim)` with zero-init weights+bias; `_voxel_pass()` helper runs scatter→UNet→gather at arbitrary res; forward concatenates fine+coarse and projects. Params 7.69M → ~8.7M.
- **Result:** val/l2 = **0.8869** at epoch 38 (timeout hit at epoch 38, 45.0 min, ~71s/epoch due to 2× UNet forward, 7.4 GB). W&B project `kagent-v14`. Val descended but never reached v6's 0.8707 band.
- **Verdict:** discarded — 0.016 worse than v6; 2× UNet cost per step means only 38 epochs fit vs v6's 52, so anneal tail was cut short.
- **Notes:** Multi-scale did not help within budget — the extra global context is real but doesn't overcome the lost anneal epochs. If I kept multi-scale, I'd need to either shrink `voxel_mid` or go to `MAX_TIMEOUT=60` (untested). Given v7-v14 all landed 0.87-0.92 without beating v6, the evidence points to v6's 0.8707 being a lucky single-epoch dip in bs=1 val noise. Next (v15): **multi-seed ensemble** — train a 2nd v6-arch seed, average predictions with v6. Independent-error cancellation typically gives 1–10% error reduction on ensembled predictions with zero arch change. This directly attacks the noise-floor rather than trying to move the mean down.


### 2026-04-17 — v13 DISCARDED — SWA over last 23 epochs landed 0.8753 (above v6's lucky 0.8707)
- **Hypothesis:** v6's 0.8707 is likely a lucky single-epoch dip in batch_size=1 val noise. Stochastic Weight Averaging (SWA) over the cosine-anneal phase averages model weights across the late-training basin — a memoryless alternative to v10's EMA that is not dragged down by early random init. Classic noise-floor reduction for small-val regimes.
- **Change:** `train.py` — `cfg.swa_start: int = 30` (1-indexed), `swa_state` dict updated by running-mean after each `scheduler.step()` starting at epoch ≥ swa_start. After training, SWA weights loaded into model, validated; replaces raw-best checkpoint if SWA val < raw best.
- **Result:** val/l2 = **0.8753** (SWA, 23 epochs averaged, beat raw best 0.8758 at ep51/52). 45.6 min, 52s/epoch, 6.2 GB. W&B project `kagent-v13`. SWA helped marginally (raw 0.8758 → SWA 0.8753) — variance reduction worked. But both are above v6's 0.8707.
- **Verdict:** discarded — 0.0046 worse than v6. The SWA smoothed floor confirms ~0.875 is this architecture's natural late-training level; v6 won by catching a lower single-epoch oscillation.
- **Notes:** v13's raw best (0.8758) is itself a different-seed sample of the ~0.87–0.88 band — right in the bs=1 val noise envelope around v6's 0.8707. SWA collapses the envelope to ~0.875 reliably, which is strong evidence that re-rolling any v6-like config gives 0.87–0.88 ± 0.005. To actually move the floor, I need an architectural change that lowers the entire band, not just a variance reducer. Next (v14): **multi-scale voxel** — add a 32³ parallel UNet branch alongside the 64³ one, concat features, project back. 32³ gives ~2× coarser spatial context (global wake patterns the fine 64³ misses), at ~12% of 64³'s compute. First true architectural expansion since v5 (SDF feature); tests whether the 64³-only bottleneck is the real plateau.


### 2026-04-17 — v12 DISCARDED — bs=2 lr=7e-4 regressed 0.033 (fewer updates > lower grad noise)
- **Hypothesis:** v6 trains at bs=1 which has very noisy per-step gradients. Doubling batch to 2 with `lr` scaled sqrt(2)× (5e-4 → 7e-4) should give smoother updates and a cleaner descent path. Simplest unexplored lever.
- **Change:** `train.py` — `cfg.lr=7e-4`, `cfg.batch_size=2`. No other code changes. VRAM 6 → 12 GB as expected.
- **Result:** val/l2 = **0.9040** at epoch 54 (45.3 min, 50s/epoch). W&B project `kagent-v12`. Val tracked 0.05–0.10 above v6 at matched epochs throughout (ep20 v12 1.13 vs v6 ~1.0; ep40 v12 0.94 vs v6 ~0.90), and kept that gap even through the cosine anneal.
- **Verdict:** discarded — 0.033 worse than v6. Fewer gradient steps beat smoother gradients.
- **Notes:** With bs=2, epoch is 365 steps instead of 730 → half the gradient updates in same wall-clock. Linear-LR scaling would be 1e-3 (not 7e-4); possibly undertuned, but the SGD-noise argument probably flips on this dataset regime — small-data noise is actually *useful* regularization, not pure loss. Lesson: stick with bs=1. Next (v13): target the other known weakness — "v6's 0.8707 looks partly lucky in batch_size=1 val noise" — via **Stochastic Weight Averaging (SWA)** over final 12 epochs (constant averages of model weights — averaged in parameter space, not gradient space). Unlike EMA (v10), SWA is memoryless of early random init and couples to the *cosine anneal* phase where the model oscillates around a broad minimum. Classic fix for noisy-val plateaus.


### 2026-04-17 — v11 DISCARDED — Fourier pos encoding smoother but 0.017 above v6
- **Hypothesis:** v6's 64³ voxel grid discretizes position coarsely (~1 cell per ~2% of bbox). Adding sinusoidal Fourier features of per-sample-bbox-normalized pos at 4 frequency bands (scales 1π, 2π, 4π, 8π) gives the pointwise MLP fine positional detail the voxel aggregation loses, in the spirit of NeRF coord encoding.
- **Change:** `train.py` — `fourier_encode(pos_norm, n_bands=4)` helper; `VoxelResidualModel.__init__` gains `fourier_bands=4` arg, `in_dim` += 24 (2·3·4); in `forward()` per-sample bbox-normalize pos to [-1,1], compute `pos_fourier`, concat before `proj_in`. 7.69M → 7.70M params. `predict.py` unchanged (uses default `fourier_bands=4`).
- **Result:** val/l2 = **0.8878** at epoch 52 (45.0 min, 53s/epoch, 6.1 GB). W&B project `kagent-v11`. Trajectory descended smoothly with no major dips — late epochs clustered 0.88–0.89 (0.8902@51, 0.8878@52, 0.8922@50, 0.8936@47) vs v6's noisier 0.87–0.93 with the 0.8707 lucky point.
- **Verdict:** discarded — 0.017 above v6, within bs=1 val noise but strictly worse on the scored metric.
- **Notes:** Fourier appears to genuinely stabilize late-phase val (smaller variance, no sub-0.88 lucky points) — a *better-behaved* curve that still doesn't beat v6's noisy minimum. Two takeaways: (1) smooth descent + no lucky minimum suggests Fourier pushes the model into a different, possibly-more-generalizable basin that happens to sit just above 0.87; (2) v6's 0.8707 is almost certainly partly luck in the batch_size=1 val — but the leaderboard is scored on the single checkpoint with best val, so we need that exact point. Next (v12): test **batch_size=2, lr=7e-4** — simplest unexplored lever. Reduces both training gradient noise and (if we also report ensemble val) val noise. If plateau persists, try multi-scale parallel voxels (32³ + 64³).


### 2026-04-17 — v10 DISCARDED — EMA(0.999) weights regressed ~0.036
- **Hypothesis:** v6's 0.8707 was likely a lucky raw-val point at `batch_size=1` (val noise ~0.02 between epochs). EMA(0.999) tracks a moving average of weights, which should give a smoother, more reliable val curve and let the best point reflect stable generalization rather than noisy peaks. Isolating EMA from v3's combined EMA+mirror-flip failure.
- **Change:** `train.py` — added `EMA` class (shadow = clone of params, `update()` after each `optimizer.step()`, `apply()/restore()` around validation + checkpointing). `cfg.ema_decay=0.999`. Otherwise identical to v6.
- **Result:** val/l2 = **0.9067** at epoch 52 (45.7 min, 53 s/epoch, 6.2 GB). W&B project `kagent-v10`. Val descended monotonically (every epoch a new best) — EMA smoothing worked as expected for noise reduction. But the EMA val magnitude plateaued at ~0.907, well above v6's lucky 0.8707.
- **Verdict:** discarded — 0.036 worse than v6.
- **Notes:** The monotone descent is evidence EMA removed val noise. But the average level EMA settles at is higher than the best raw-val point v6 hit — i.e. the variance we smoothed away contained the 0.8707 win. At `decay=0.999` (half-life ~693 steps ≈ 1 epoch) the EMA weights trail the current model by ~1 epoch of progress; with cosine LR decaying over 60 epochs, that lag costs a few hundredths of val/l2. Possible rescue: (a) `decay=0.9995` (half-life ~2 epochs) with a longer run — likely still trails. (b) EMA only over *last N epochs* so early random init doesn't pollute. (c) raw-model checkpoint + tester-side ensembling. For now, reverting to v6. Next (v11): try the one spatial thing not yet done — multi-scale voxel (concatenate outputs of 32³ + 64³ UNets). Gives the model both long-range (32³ bigger effective receptive field) and fine (64³) spatial context. Memory+compute <2× because of how the UNet scales with grid volume.


### 2026-04-16 — v9 DISCARDED — hidden=384 per-epoch gain lost to 2.3× slowdown
- **Hypothesis:** v6 train loss kept descending (0.007 at ep52), val plateauing → generalization limited by capacity, not optimization. Bump `hidden` 256→384 (MLP width), keep everything else v6. Initial try at `epochs=60` aborted because cosine T_max=60 stretched past actual run time (same trap as v8); restarted as v9b with `epochs=30` so cosine anneals to 0 exactly at timeout.
- **Change:** `train.py` — `hidden: int = 384`, `epochs: int = 30`. Launched with `MAX_TIMEOUT_MIN=60`.
- **Result:** val/l2 = **0.9222** at epoch 30 (last, 56.3 min, 75–300 s/epoch depending on GPU contention, 8.4 GB). W&B project `kagent-v9`. Per-epoch val was clearly ahead of v6/v8 at the same epoch (ep20 v9b 0.9707 vs v6 ~1.05), confirming the capacity helps — but each epoch took 2.3× longer, so only 30 epochs fit.
- **Verdict:** discarded — 0.05 worse than v6's 0.8707. Capacity trade-off lost against wall-clock.
- **Notes:** GPU contention from shared workload caused erratic epoch times (75s best, 295s worst). Even without contention, at ~75s/epoch steady-state we could only fit ~46 epochs in 60 min; still short of v6's 52 well-annealed epochs. The capacity-gain-per-epoch is real but the time tax is too steep on this GPU share. Next (v10): isolate EMA weights (decay=0.999) from v3's combined EMA+mirror failure — EMA is a cheap, known-good variance-reduction for noisy `batch_size=1` val.


### 2026-04-16 — v8 DISCARDED — more training time alone did not beat v6
- **Hypothesis:** v6 was still descending at timeout (0.8745 → 0.8707 in last 2 epochs). Extend `epochs=60→75` and `MAX_TIMEOUT_MIN=45→60` — same architecture, same features, just more wall-clock and a cosine schedule that stays warmer longer in the ep 50-60 zone where v6 was still finding wins.
- **Change:** `train.py` — `epochs: int = 75` (no other code change). Launched with `MAX_TIMEOUT_MIN=60`.
- **Result:** val/l2 = **0.8760** at epoch 68 (69 epochs run, 60.3 min, 52s/epoch, 6.1 GB). W&B project `kagent-v8`. Descent pattern: 0.9080@ep48 → 0.8928@ep53 → 0.8796@ep62 → 0.8760@ep68 → 0.8762@ep69 (last).
- **Verdict:** discarded — 0.005 worse than v6's 0.8707, inside val noise (batch_size=1 oscillates ~0.02 epoch-to-epoch). No clear win.
- **Notes:** Why not better? v6's cosine with T_max=60 cooled LR to near-zero by ep52 and that low-LR phase is probably what locked in the 0.8707. v8's cosine with T_max=75 left LR ~3-4× higher at the same wall-clock, so it kept moving around instead of annealing into a basin. Lesson: if I want more-training wins, I should *also* keep the final-LR-near-zero phase long enough to exploit. Next (v9): the true bottleneck is almost certainly model capacity + noisy val — try `hidden=384` with T_max matching actual-run epochs (~58). If capacity helps, val/l2 should drop cleanly at any epoch count; if not, move to multi-scale voxel or batch_size=2 to reduce val noise.


### 2026-04-16 — v7 DISCARDED — wall-offset vector feature regressed ~0.036
- **Hypothesis:** scalar SDF tells the model *how far* to the wall but not *which direction*. Adding the 3D offset vector (`nearest_airfoil_pos - pos`) should give an explicit wall-normal-ish direction prior for pressure-gradient physics, on top of v6's longer training.
- **Change:** `train.py` — `compute_sdf()` also returns per-point offset vector (N,3); `SDFDataset`/`collate_sdf` carry it; model `in_dim` 21→24 with `offset_feat = offset/5.0` concatenated. `predict.py` updated to compute and pass offset. Otherwise identical to v6.
- **Result:** val/l2 = **0.9070** at epoch 51 (vs v6's 0.8707 at 52). Strictly worse the entire run (epoch 1: 1.58 vs v6 1.52; epoch 24: 1.03 vs v6 ~0.95). 52s/epoch, 6.1 GB. W&B project `kagent-v7`.
- **Verdict:** discarded — reverted to v6.
- **Notes:** 3 extra input features increased in_dim by 14% (21→24) but the scaling `offset/5.0` may have interacted poorly with early training dynamics. Offset magnitude is correlated with SDF magnitude (they measure the same thing up to a direction), so the model gets redundant signal that slows feature disentanglement. A cleaner direction prior would be a unit-normalized offset (wall-normal unit vector) separate from the scalar SDF, so the two features have independent meaning. Next (v8): try something architecturally simpler — bigger model (hidden=384) + more time, capitalizing on v6 still-descending curve.


### 2026-04-16 — v6 longer training (60 epochs, 45 min)
- **Hypothesis:** v5 was still dropping val/l2 in the last 5 epochs (0.9253 → 0.9089). Cosine LR schedule tuned to 60 epochs (so LR is still decent at epoch ~50) plus 45-min timeout should let it converge further.
- **Change:** `train.py` — `epochs=60`, MAX_TIMEOUT=45. No architecture changes.
- **Result:** val/l2 = **0.8707** at epoch 52. 45.4 min, 52s/epoch, 6.1 GB. W&B project `kagent-v6`. Mae: check W&B.
- **Verdict:** kept — clean 0.038 improvement over v5. Still descending at timeout (0.8745 → 0.8707 last 2 epochs).
- **Notes:** Confirms more time helps. Per-epoch improvement slows but doesn't plateau — more training budget would keep extracting wins. Next (v7): combine long training with a better wall-feature — replace scalar SDF with the full 3D vector to the nearest airfoil point (encodes both distance AND wall-normal direction, which drives pressure gradient physics).


### 2026-04-16 — v5 SDF-to-airfoil feature
- **Hypothesis:** near-wall flow physics (boundary layer, pressure gradient) depend strongly on distance to the wall. The airfoil-mask bit tells the model *if* a point is on the wing, but not how far off-wall. Add per-point Euclidean distance to the nearest airfoil point as an input feature (both raw/5 and log1p-transformed so the model can key on both near-field and far-field scales).
- **Change:** `train.py` — `compute_sdf()` (GPU cdist, chunked), precompute per sample once at startup (~20s for 810 samples), `SDFDataset` wrapper + `collate_sdf`. Model `in_dim` 19→21 (added sdf_raw, sdf_log). `predict.py` does the same precompute. Arch identical to v2 otherwise.
- **Result:** val/l2 = **0.9089** at epoch 35 (vs v2's 0.9228 at epoch 31). 30.6 min, 52s/epoch, peak 6.1 GB. 7.69M params. W&B run in project `kagent-v5`.
- **Verdict:** kept — clear win of 0.014 and still descending at timeout (best epoch = last epoch).
- **Notes:** SDF gives epoch-1 val/l2 = 1.52 vs v2's 1.64 — model uses it from the start. Cost is ~20s startup + 0 per-epoch overhead (SDF is a fixed feature). Next (v6): could try multi-scale voxel, or pair SDF with Fourier-encoded pos, or train longer (still descending).


### 2026-04-16 — v4 DISCARDED — bigger UNet (voxel_mid=96) slightly worse
- **Hypothesis:** v2 was still descending at timeout (epoch 31). Doubling spatial capacity (voxel_mid 64→96, params 7.7M→15M) with more epochs (40→50) and more time (27→35 min) should push lower without aug/EMA confounds.
- **Change:** `train.py` — `voxel_mid=96`, `epochs=50`; `predict.py` updated to match.
- **Result:** val/l2 = **0.9349** at epoch 32 (vs v2's 0.9228). 35.5 min, 6.8 GB, 67s/epoch. W&B run in project `kagent-v4`.
- **Verdict:** discarded — slightly worse. Best checkpoint was the last epoch (still descending), so given more budget it might eventually beat v2, but not a convincing win.
- **Notes:** Big-model slower per step (67 vs 52 s/epoch) → fewer effective epochs in same wall-clock. Val noise pattern is the same shape as v2 — just offset. Capacity alone isn't the bottleneck. Next (v5): add a real physics feature — signed distance to airfoil — so the model has an explicit wall-distance prior.


### 2026-04-16 — v3 DISCARDED — EMA + y-mirror aug regressed ~0.06
- **Hypothesis:** stack two free wins on v2: (1) EMA(0.999) weights smooth noisy B=1 val; (2) random y-mirror augmentation doubles effective data (F1 wing is y-symmetric). Epochs=50, MAX_TIMEOUT=30 min.
- **Change:** `train.py` — added `EMA` class, update each step, swap in before validate/save. Random flip of `pos[...,1]`, `v_in[...,1]`, `v_out[...,1]` with p=0.5 during training.
- **Result:** val/l2 = **0.9861** at epoch 35 (vs v2's 0.9228 at epoch 31). Consistently ~0.05–0.08 behind v2 throughout training. 30.7 min, 6.2 GB. W&B run `7c7qljbi` (project `kagent-v3`).
- **Verdict:** discarded — reset to v2. Hurt not helped.
- **Notes:** Can't separate EMA vs mirror-flip effects in this run. Most likely culprit: y-mirror assumption may be wrong (dataset has yaw or asymmetric wing geometries → flipping invents OOD data). EMA by itself usually helps; but v3 might be stuck in a "not-converged early EMA lag" regime combined with harder targets. Next (v4): isolate by trying more capacity + more epochs without aug or EMA.


### 2026-04-16 — v2 voxel-UNet spatial context (64³)
- **Hypothesis:** v1 was a per-point MLP — zero spatial interaction. Near-wall flow depends on neighbors (wakes, pressure coupling). A 3D voxel-UNet (scatter-mean features into 64³ grid, run UNet, trilinear scatter-back) gives every point global+local context with the bottleneck giving receptive field ≫ wing chord. Residual around v1's per-point backbone so spatial block only needs to learn the correction.
- **Change:** `train.py` — added `VoxelSpatial` (scatter/gather + 3-level UNet3D, GroupNorm), inserted between 2 pre-blocks and 4 post-blocks of ResMLP. Zero-init UNet output conv → block starts as identity. Axis permutation `[2,1,0]` on grid_sample coords to match (x,y,z)↔(W,H,D). `hidden=256, voxel_res=64, voxel_mid=64`. Moved training code into `main()` so predict.py import doesn't trigger `sp.parse`. 7.69M params.
- **Result:** val/l2 = **0.9228** at epoch 31 (timeout cut), mae (Ux,Uy,Uz)=(0.624, 0.286, 0.419). 27.1 min, 52s/epoch, peak 6.1 GB. W&B run `eji6edpc`. Predictions at `predictions/apr16/alphonse/cde4a6b`.
- **Verdict:** kept — **30% improvement over v1** (1.3200 → 0.9228). Mae dropped across all components; largest in Ux (0.884 → 0.624), the hardest/largest-std axis.
- **Notes:** Smooth descent, still dropping at timeout (epoch 30: 0.9303, 31: 0.9228) — more epochs would keep winning. Val noise persists (batch_size=1). Next (v3): give it more time. Easy wins: larger unet_mid=96, per-point kNN for fine detail the 64³ voxel misses (airfoil is only ~5 voxels wide in some axes), EMA weights, 60-epoch budget with smaller MAX_TIMEOUT overhead.


### 2026-04-16 — v1 residual ResMLP + no-slip + normalized loss
- **Hypothesis:** baseline predicts absolute velocity from scratch — a residual around `velocity_in[-1]` is a much stronger starting point because frame-to-frame changes are small relative to the mean flow (~35 m/s mean Ux). Hard no-slip BC guarantees zero at airfoil. Normalized MSE loss stops the ~20 m/s Ux std from dominating the gradient.
- **Change:** `train.py` — `ResidualPointMLP` (hidden=384, n_blocks=8). Input features: normalized velocity_in (15) + pos (3) + airfoil mask (1) = 19. Output: delta in normalized space; denormalize and add to last input frame. Zero-init last linear → starts at exact persistence. Post-process no-slip mask. Loss is MSE on (pred - gt)/vel_std. Grad clip 1.0.
- **Result:** val/l2 = **1.3200** at epoch 21, mae (Ux,Uy,Uz)=(0.884, 0.375, 0.641). 26 epochs in 25 min, ~55s/epoch, peak 8.1 GB. 4.75M params. W&B run `ajszccxm`. Commit `adeebc6`.
- **Verdict:** kept — clean win vs baseline ~1.76 on mar29 val; zero-init residual made training stable from epoch 1 (epoch 1 already 1.59, below baseline's final).
- **Notes:** Val oscillates 0.05 between epochs — batch_size=1 is noisy. Loss kept dropping at end, so more epochs likely helps. predict.py broke because importing train.py triggered `sp.parse(sys.argv)` on predict's args; fixed by wrapping train.py body in `main()` + `if __name__ == "__main__":`. Per-point MLP — no spatial interaction. Next (v2): voxel-UNet spatial module.

