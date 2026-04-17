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

### 2026-04-17 — exp26: chain warm-start from exp25 (triple-scale refine) + lr=2e-5
- **Hypothesis:** Exp25 showed same undertrained pattern as exp18/22. Following recipe: chain at lr=2e-5 for 30 more min to let ucoarse branch refine while preserving main weights. If pattern holds, should beat exp24 (0.9091) by 0.005-0.015.
- **Change:** --warm_start=<exp25 ckpt model-lkb1o42z> --lr=2e-5. No code changes.
- **Result:** TBD
- **Verdict:** TBD
- **Notes:** Slower epochs (100s) so fewer iterations. May need to chain twice.

### 2026-04-17 — exp25: triple-scale voxel (add 8³ ultra-coarse branch)
- **Hypothesis:** Multi-scale voxel (exp18/19) showed coarse branch added real signal. Extend same pattern: add parallel G/4=8³ ultra-coarse branch per VoxelMixer. Captures scene-level context (wing upper vs lower, wake near vs far). Zero-init (proj_agg_ucoarse=0, last conv_ucoarse=0) → warm-start safe.
- **Change:** VoxelMixer adds proj_agg_ucoarse + conv_ucoarse (G/4 grid). Forward adds sampled_u to output.
- **Result:** val/l2=**0.9192** @ epoch 16 (30.0 min). train=0.0044. Val oscillated 0.9192-0.9534. 100s/epoch.
- **Verdict:** HOLD — worse than exp24 (0.9091) by 0.0101. Same undertrained pattern as exp18/22. Val still chaotic at timeout.
- **Notes:** Following the established recipe: exp26 = chain at lr=2e-5 to stabilize and extract. Expect exp26 to recover and surpass 0.9091 like exp19/23 did.

### 2026-04-17 — exp24: chain warm-start from exp23 + lr=5e-6
- **Hypothesis:** Exp23 val stable E14-22 (0.9096-0.9115). Very low LR chain should squeeze last fine-tune. Expected Δ~0.001-0.003.
- **Change:** --warm_start=<exp23 ckpt model-mnvmn263> --lr=5e-6. No code changes.
- **Result:** val/l2=**0.9091** @ epoch 2 (31.0 min). train=0.0036. Val stable 0.9094-0.9104.
- **Verdict:** KEPT — micro +0.0005 over exp23. Chain fully saturated at 5e-6. Pivoting to arch.
- **Notes:** Current chain tail: 0.9238→0.9202→0.9194→(FiLM)0.9096→0.9091. Going to exp25 = triple-scale voxel (add 8³ ultra-coarse branch, same zero-init recipe as exp18).

### 2026-04-17 — exp23: chain warm-start from exp22 (FiLM refine) + lr=2e-5
- **Hypothesis:** Exp22 FiLM destabilized by high LR but trending down. Chain at lr=2e-5 should let FiLM params settle without further destroying main weights. If FiLM has signal, this extracts it; if not, we'll plateau around exp22's 0.9281.
- **Change:** --warm_start=<exp22 ckpt model-79ynl9v3> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9096** @ epoch 19 (30.3 min). train=0.0037. Val stable 0.9096-0.9115 E14-22.
- **Verdict:** KEPT — +0.0098 over exp21 (0.9194). FiLM validated: absolute t signal helps. Second confirmation of the pattern "new arch regresses then chain recovers + beats".
- **Notes:** Pattern: exp18/19 (multi-scale: -0.017 → +0.016), exp22/23 (FiLM: -0.009 → +0.010). Both pairs net positive, both arch additions provide real signal. Next: chain further (exp24 @ lr=5e-6) then another arch.

### 2026-04-17 — exp22: FiLM time conditioning (warm-start from exp21)
- **Hypothesis:** t is still unused. Sample-specific absolute time (0.3-0.4 range) correlates with flow development stage. Add TimeEncoder (MLP 1→64→64→2*hidden*17) producing per-block (γ, β) FiLM params, applied as h*(1+γ)+β after each block's norm. Zero-init final MLP layer → FiLM=identity at init → warm-start preserves exp21 exactly.
- **Change:** train.py: added TimeEncoder class, _apply_film helper. ResBlock.forward and VoxelMixer.forward now take optional `film` kwarg. BaselineMLP.forward calls time_enc(t) once, iterates blocks with per-block FiLM slice.
- **Result:** val/l2=**0.9281** @ epoch 19 (31.2 min). train=0.0047. Val oscillated 0.9281-0.9576.
- **Verdict:** HOLD — worse than exp21 (0.9194) by 0.0087. lr=2e-4 too aggressive for fresh FiLM layers; disrupted fine-tuned core. Trajectory trending down at end (0.9355→0.9286→0.9355). Not committed to git.
- **Notes:** FiLM output scale grew too fast. Next: exp23 chain at lr=2e-5 to let FiLM refine while preserving main weights. If exp23 still loses, revert to exp21.

### 2026-04-17 — exp21: chain warm-start from exp20 + lr=1e-5
- **Hypothesis:** Cheap-harvest: one more chain step at lr=1e-5 should extract last refinement. Expected Δ~0.001-0.002 based on geometric decay (0.0036→half).
- **Change:** --warm_start=<exp20 ckpt> --lr=1e-5. No code changes.
- **Result:** val/l2=**0.9194** @ epoch 22 (30.0 min). train=0.0039.
- **Verdict:** KEPT — +0.0008 over exp20. Chain fully saturated now (was expected). Val oscillated 0.9194-0.9219 E14-22.
- **Notes:** Multi-scale chain progression: 0.9238→0.9202→0.9194 (Δ=0.0158→0.0036→0.0008). Time to pivot to arch change.

### 2026-04-17 — exp20: chain warm-start from exp19 + lr=2e-5
- **Hypothesis:** Exp19 val stable at 0.9238-0.9283 (E12-23). Best at E12 suggests quick overfit. Chain at lr=2e-5 should refine without moving too far. Expect Δ~0.003-0.007.
- **Change:** --warm_start=<exp19 ckpt model-fdr738d3> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9202** @ epoch 10 (30.9 min). train=0.0040. Val stable 0.9202-0.9228 E10-24. run run id in PVC model-<exp20>.
- **Verdict:** KEPT — +0.0036 over exp19 (0.9238). Chain refinement of multi-scale arch. Best at E10 again (similar quick-minimum as exp19). Val floor 0.9202-0.9211 through E24.
- **Notes:** Multi-scale chain progression: exp18(0.9411 undertrained) → exp19(0.9238, Δ=0.0158 with lr=5e-5) → exp20(0.9202, Δ=0.0036 with lr=2e-5). Geometric decay. Distance to leader: 0.9202 - 0.8132 = 0.107.

### 2026-04-17 — exp19: chain warm-start from exp18 + lr=5e-5
- **Hypothesis:** Exp18 val was still dropping rapidly at E22-24 (0.9469→0.9411). Coarse branch needs more epochs at a lower LR to refine without destroying fine-branch weights. Chain at lr=5e-5 for 30 more min. If it beats exp17's 0.9396, commit the full multi-scale arch.
- **Change:** --warm_start=<exp18 ckpt model-5qrjp5if> --lr=5e-5. No code changes.
- **Result:** val/l2=**0.9238** @ epoch 12 (32.0 min). train=0.0040. Val settled 0.9238-0.9283 E12-23. 76-170s/epoch (some batches slower).
- **Verdict:** KEPT — +0.0158 over exp17 (0.9396). Biggest gain since exp15 (Δ=0.0126). Multi-scale architecture validated: the coarse branch added real signal the 32³ branch couldn't capture. Now #3 or better on leaderboard (was #4 at 0.9487, nezuko=0.9299).
- **Notes:** Best at E12 is unusual — model quickly adapted + overfit slightly after. E12-23 all within 0.9238-0.9283 (stable basin). Next: exp20 chain at lr=2e-5 to squeeze more.

### 2026-04-17 — exp18: multi-scale voxel (16³ coarse parallel, warm-start from exp17)
- **Hypothesis:** Chain saturating again. Arch change gave biggest recent win (exp15 Δ=0.0126). Add parallel 16³ coarse voxel branch to each VoxelMixer — captures larger-scale flow structures (wake, separation zones) that 32³ 3×3 conv can't reach. Zero-init (proj_agg_coarse=0, last conv_coarse=0) → sampled_c=0 at init → warm-start equivalent to exp17. Fresh training allows coarse branch to learn contributions.
- **Change:** VoxelMixer now has a parallel G_coarse=G/2 branch (proj_agg_coarse + conv_coarse) summed with fine output. Zero-init end-to-end. Refactored fine-branch logic into _voxel_mm/_gather helpers.
- **Result:** val/l2=**0.9411** @ epoch 23 (30.5 min). train=0.0052. run 5qrjp5if. 76s/epoch (38% slower).
- **Verdict:** HOLD — slightly worse than exp17 (0.9396 vs 0.9411). Val was still rapidly dropping at E22-24 (0.9469→0.9411→0.9494). Coarse branch undertrained — lr=2e-4 was too aggressive + too few epochs. Not committed to git (don't regress).
- **Notes:** Param count 59.7M→61.1M. 76s vs 55s/epoch. Next: exp19 = chain from exp18 PVC ckpt at lr=5e-5 to let coarse branch mature.

### 2026-04-17 — exp17: chain warm-start from exp16 + lr=2e-5
- **Hypothesis:** Exp16 val was still dropping at E33 (0.9422, best E24 0.9420). One more chain at lr=2e-5 should extract the last fine-tuning gains — similar pattern to the exp12→exp13 chain (+0.0025). With a richer arch (mean+max) there may be more to extract.
- **Change:** --warm_start=<exp16 ckpt model-4fk50oi3> --lr=2e-5. No code changes.
- **Result:** val/l2=**0.9396** @ epoch 25 (30.1 min). train=0.0044. run uxpz2abi. 55s/epoch.
- **Verdict:** KEPT — +0.0024 over exp16. Val oscillated 0.9396-0.9433 E25-33. Chain clearly saturating on this arch at this LR.
- **Notes:** Chain on mean+max arch: exp15(0.9487)→exp16(0.9420,Δ=0.0067)→exp17(0.9396,Δ=0.0024). Geometric decay returns. Time to pivot: next is arch change (exp18 — multi-scale voxel).

### 2026-04-17 — exp16: chain warm-start from exp15 + lr=5e-5
- **Hypothesis:** Exp15 val was still slightly dropping at E33 (0.9487). The new proj_agg layers were still learning. Another 30min of fine-tuning at lr=5e-5 should let the mean+max aggregation fully mature.
- **Change:** --warm_start=<exp15 ckpt> --lr=5e-5. No code changes.
- **Result:** val/l2=**0.9420** @ epoch 24 (30.0 min). train=0.0047. run 4fk50oi3. 54s/epoch.
- **Verdict:** KEPT — +0.0067 over exp15 (0.9487). Early epochs looked bad (E1 0.9513, E2 0.9603) but converged cleanly after E10. Mean+max aggregation wasn't saturated yet.
- **Notes:** Lesson: don't trust early chain epochs; give at least 15-20 epochs before bailing. Val still dropping E30→E33 (0.9450→0.9422). Another chain at lr=2e-5 might squeeze more. Chain progression: Δ=0.040→0.006→0.004→0.0025→(arch)→0.0067. Distance to leader: 0.9420 - 0.8245 = 0.1175.

### 2026-04-17 — exp15: mean+max voxel aggregation (warm-start partial)
- **Hypothesis:** Current voxel aggregation is scatter_add/count (mean) — loses extreme features (high-velocity gradients, shear layers). Add parallel scatter_reduce(amax) branch, concat with mean (2×dim), project back via Conv3d(2*dim, dim). This captures peaks per voxel (turbulent events). Pre-voxel layers weights can warm-start from exp13, only new proj_agg needs fresh training.
- **Change:** VoxelMixer adds scatter amax + 1x1 Conv3d(2D→D) projection. Identity init on mean half, zero on max half → first forward pass identical to exp13 baseline. Warm-start with strict=False.
- **Result:** val/l2=**0.9487** @ epoch 33 (30.1 min). train=0.0050. 4.8GB peak. run 4cun281z. 54s/epoch.
- **Verdict:** KEPT — +0.0126 over exp13 (0.9613). Mean+max captures voxel-local peaks the mean-only lost. Val still dropping at E33 (last epoch).
- **Notes:** Larger single-experiment gain than recent chain steps. This is the first successful arch change in a while. Next: chain-finetune exp15 with lower LR.

### 2026-04-17 — exp14: pos jitter=0.5 (DISCARDED)
- **Hypothesis:** Warm-start + pos jitter (0.5 × voxel_size Gaussian noise) as regularization to unstick saturated chain.
- **Change:** train.py: added cfg.pos_jitter, applied Gaussian noise to pos.
- **Result:** val/l2=**1.1116** @ epoch 1 (broke model). Train loss exploded to 0.028+ immediately.
- **Verdict:** DISCARDED — 0.5 × voxel_size is far too large; warm-started model's SDF features + voxel assignments are sensitive to position perturbations. val climbed rather than fell.
- **Notes:** Lesson: fine-tuned model is brittle to input perturbation. Jitter scale <0.05 might work but arch change is higher-leverage.

### 2026-04-16 — exp13: chained warm-start from exp12 + lr=2e-5
- **Hypothesis:** Exp12 plateau in 0.964-0.966 suggests we're at LR=5e-5 convergence. One more chain at lr=2e-5 to squeeze last ~0.002. After this plateau, pivot to arch changes.
- **Change:** Run with --warm_start=<exp12 ckpt> --lr=2e-5.
- **Result:** val/l2=**0.9613** @ epoch 36 (30.3 min). train=0.0053. run rfc1ntlv.
- **Verdict:** KEPT — +0.0025 over exp12. Chain progression: Δ=0.040→0.006→0.004→0.0025, clear geometric decay.
- **Notes:** Chain saturated; best checkpoint still at exp13.

### 2026-04-16 — exp12: chained warm-start from exp11 + lr=5e-5
- **Hypothesis:** Exp11 val oscillated 0.968-0.977 mid-run, best at E23 (0.9681). LR=1e-4 still too high for fine-tuning. Half again to 5e-5, warm-start from exp11. Expected gain ~0.004 (diminishing chain).
- **Change:** Run with --warm_start=<exp11 ckpt> --lr=5e-5.
- **Result:** val/l2=**0.9638** @ epoch 25 (30.0 min). train=0.0057. run 9jtypb06.
- **Verdict:** KEPT — +0.004 over exp11 (0.9681). Chain plateau approaching: val oscillated 0.964-0.966 at E20-25. Train at 0.0057 (still dropping slowly).
- **Notes:** Chain progression: exp8(1.014) → exp10(0.974, Δ=0.04) → exp11(0.968, Δ=0.006) → exp12(0.964, Δ=0.004). Geometric decay in gains. One more chain expected ~0.002.

### 2026-04-16 — exp11: chained warm-start from exp10 + lr=1e-4
- **Hypothesis:** Exp10 (0.9742) hit minimum mid-run (E36) then slowly climbed — the 2e-4 cosine was slightly too high late, allowing oscillation. Chain another warm-start from exp10 with lr=1e-4 (half), adds 30 more min of fine annealing. Train=0.0070 suggests capacity room remains. Each warm-start cycle has diminishing returns but should add 10-20% improvement per run until plateau.
- **Change:** Run train.py with --warm_start=<exp10 ckpt> --lr=1e-4.
- **Result:** val/l2=**0.9681** @ epoch 23 (30.3 min). train=0.0062. 3.6GB peak. run dyjblcu8.
- **Verdict:** KEPT — +0.006 over exp10. Val plateaued in 0.968-0.977 range after E20 (minimum at E23 then oscillation). Chain is working but diminishing.
- **Notes:** LR=1e-4 was too high for late-cycle refinement — val kept oscillating at end. Next chain at lr=5e-5 should extract remaining fine-tuning gains.

### 2026-04-16 — exp10: warm-start from exp8 + 30 more min
- **Hypothesis:** Exp8 val was still dropping linearly at 30min timeout (best 1.0137). Rather than architectural change, reload exp8 ckpt and train another 30min with a fresh cosine LR schedule at lower peak (2e-4 vs 5e-4). Effectively doubles training budget without needing arch changes. SGDR-style warm restart: new annealing cycle may find better minima from a pre-trained init.
- **Change:** train.py: Added `warm_start: str | None` to Config; loads state_dict post model init. Set cfg.lr=2e-4 for fine-tune run.
- **Result:** val/l2=**0.9742** @ epoch 36 (30.2 min). train=0.0070 (vs exp8's 0.0100, ~30% lower). 3.6GB peak. run dgolkvqw. 46-67s/epoch.
- **Verdict:** KEPT — beats exp8 (1.0137) by 0.0395 (massive jump). Warm-start effectively doubled compute budget. Val still dropping slowly at end (0.9742 → 0.9768 after peak, so we found the minimum). Now rank #2 above nezuko (0.9867).
- **Notes:** This is the biggest single-experiment win. Compound warm-start is a clear winning strategy while budgets are tight. Next: exp11 warm-start from exp10 with even lower LR.

### 2026-04-16 — exp9: subsample 50k→30k (DISCARDED)
- **Hypothesis:** Exp8 (n_blocks=8) was still dropping val at timeout. Subsample 50k→30k saves step time → more epochs to finish the cosine schedule.
- **Change:** subsample_train 50000→30000.
- **Result:** val/l2=1.0141 @ epoch 50 (29.9 min). train=0.0109, 2.6GB peak. run oxfbajwq. 36s/epoch (all 50 epochs fit).
- **Verdict:** DISCARDED — tied exp8 (1.0137) within noise. The extra subsampling regularized too much: train loss only reached 0.0109 vs exp8's 0.0100. More epochs + more regularization ≈ same effective learning. Need different mechanism to beat exp8.
- **Notes:** Lesson: aggressive subsampling acts as a regularizer ceiling. Future "buy more epochs" attempts should look at reducing other costs (grid_size, n_fourier) rather than subsample further.

### 2026-04-16 — exp8: n_blocks 6→8 on exp7 base
- **Hypothesis:** Exp7 val (1.0189) plateaued around E45-50 with train still dropping; exp5 showed depth 4→6 gained 0.016. Try 6→8 with subsample=50k to keep per-step fast. Expect ~47s/epoch → 38 epochs. More rounds of voxel mixing at same grid should extract more spatial structure, which has been our best lever so far.
- **Change:** Config.n_blocks 6→8 (train.py + predict.py). All else identical to exp7.
- **Result:** val/l2=**1.0137** @ epoch 39 (30.5 min timeout). train=0.0100, 3.5GB peak. run g6bus2pg. 47s/epoch.
- **Verdict:** KEPT — beats exp7 (1.0189) by 0.0052. Smooth descending val curve (E34=1.023 → E39=1.014), clearly under-trained at timeout. Depth continues to help.
- **Notes:** Val was still dropping linearly at timeout — more epochs would likely push to ~1.00 or below. Bottleneck is epochs, not capacity. Next: buy more epochs via subsample=30k (same arch, faster steps).

### 2026-04-16 — exp7: train-time point subsampling to 50k
- **Hypothesis:** Exp5 (best: 1.0430) had train loss still dropping at timeout (0.0091, down from 0.0103 at epoch 27). More epochs should help. Subsampling points 100k→50k at train time only (val stays 100k) gives 1.44x step speedup → ~52 epochs vs 38 in same 30min budget. Voxel grid remains same density per voxel (just slightly sparser) so spatial structure preserved.
- **Change:** Config.subsample_train=50000. Train loop: randomly sample K=50k point indices per step, slice v_in/v_out/pos, remap idcs_airfoil via inverse index. Val unchanged (all 100k).
- **Result:** val/l2=**1.0189** @ epoch 50 (29.8 min). train=0.0103, 2.7GB peak. run jdwouppy. 35s/epoch (fit all 50 epochs in budget).
- **Verdict:** KEPT — beats exp5 (1.0430) by 0.024. Subsampling acts as regularization AND enables more epochs. Final train loss 0.0103 is higher than exp5's 0.0091, but val is much better → subsampling = better generalization (new point sampling per step ≈ implicit augmentation).
- **Notes:** Val still dropping at end — could benefit from even more epochs or a deeper net. Subsampling was a 2x win (0.024 drop) without any architectural change. Next: try subsample=30k for even more epochs, or push the net deeper (n_blocks=8) now that per-step is cheap.

### 2026-04-16 — exp6: multi-scale voxel (alternate grids 32/16) (DISCARDED)
- **Hypothesis:** Alternate VoxelMixer grids 32/16 to get multi-scale receptive field at no extra param cost.
- **Change:** Block i even → grid=32, odd → grid=16. 3 blocks each scale.
- **Result:** val/l2=1.0470 @ epoch 42 (30.1 min). 43s/epoch.
- **Verdict:** DISCARDED — 0.004 worse than exp5 (1.0430). Coarse grid (16³ ≈ 6k voxels, >10 pts/voxel) oversmooths; losing 3 fine-scale blocks cost more than multi-scale gained.
- **Notes:** Lesson: reducing fine-scale depth to make room for coarse scale is a net loss. If multi-scale helps, do it via parallel branches, not alternation.

### 2026-04-16 — exp5: n_blocks 4→6 on exp4 base
- **Hypothesis:** Exp4 fully converged at 1.060 — architecture ceiling with 4 blocks. Try moderate depth bump: n_blocks 4→6, hidden=256 (23M params, 1.5x exp4). bf16 gives 68ms/step budget.
- **Change:** Config.n_blocks=6 (train.py + predict.py). All else identical to exp4.
- **Result:** val/l2=**1.0430** @ epoch 38 (30.2 min). train=0.0091, 4.4GB peak. run wu0jb4c0.
- **Verdict:** KEPT — beats exp2 (1.0595) by 0.016. Confirms depth helps where width (exp3) didn't: more rounds of spatial mixing at same grid > larger features per point.
- **Notes:** Train loss still dropping (0.0091 vs exp4 0.0103), could benefit from longer training or an even deeper net. Next: try n_blocks=8 or multi-scale voxel.

### 2026-04-16 — exp4: exp2 arch + bf16 + SDF-to-airfoil
- **Hypothesis:** bf16 autocast at exp2 sizes halves step time → ~50 epochs vs exp2's 38. Add SDF-to-nearest-airfoil + is_airfoil binary as physics priors.
- **Change:** Revert to hidden=256/n_blocks=4/grid=32. Added bf16 autocast. Added `_geom_features()`: chunked cdist to 1024-sample airfoil subset → normalized SDF + is_airfoil indicator (+2 channels). Fixed VoxelMixer dtype bug (h.dtype not x.dtype) for autocast.
- **Result:** val/l2=1.0600 @ epoch 50 (27.6 min, fully converged — train=0.0103 matched exp2 exactly).
- **Verdict:** Tied exp2 (1.0595 vs 1.0600, within noise). Kept base changes (SDF + bf16 + dtype fix) for exp5 since they enable faster iteration.
- **Notes:** SDF + voxel grid seem redundant — grid already encodes geometry. Real gains need architectural change, not more features. Clear convergence plateau confirms arch ceiling at ~1.06.

### 2026-04-16 — exp3: scale up hidden=384/n_blocks=6 + bf16 (DISCARDED)
- **Hypothesis:** Exp2 underfit — scale hidden 256→384 and n_blocks 4→6 (52M params, 3.3x). Add bf16 so bigger model still trains in 30min.
- **Change:** Config defaults bumped + bf16 autocast. Fixed VoxelMixer dtype bug.
- **Result:** val/l2=1.1058 @ epoch 21 (30min timeout). train=0.0142 (vs exp2 0.010).
- **Verdict:** DISCARDED — worse than exp2 by 0.05. Larger model under-trained: bf16 gained speed but 82s/epoch still only yielded 21 epochs vs exp2's 38. Train loss was 50% higher than exp2's converged level, confirming not enough steps.
- **Notes:** Lesson: with a fixed 30min budget, "go bigger" must be paired with enough speedup. 3.3x params needed >3x speedup to equalize step count — bf16 only gave ~1.5x. For future scale-ups: combine bf16 + subsample points + lower batch iters.

### 2026-04-16 — exp1: residual + normalize + no-slip
- **Hypothesis:** Predicting delta from `velocity_in[-1]` should be much easier than absolute velocity (|delta|=1.17 vs |v|=14 raw). Normalizing by vel_std balances loss across Ux/Uy/Uz. Hard no-slip on airfoil is a physical constraint baseline ignores.
- **Change:** BaselineMLP: normalized v_in features, predicts delta_norm (zero-init head), denorms, adds to last frame, zeros airfoil indices. hidden=384, n_blocks=8 (~4.7M params). Loss is MSE on normalized error.
- **Result:** val/l2=1.3016 @ epoch 27 (30min hit timeout at epoch 33). train_loss=0.023. 8.1GB peak. run id 8bycg5j0.
- **Verdict:** Discarded as architecture direction. Marginal gain over last-round 1.33 baseline — confirms MLP-per-point is fundamentally limited without spatial context. Auto-predict failed due to predict.py import re-running train.py's sp.parse (fixed with __main__ guard in exp2).
- **Notes:** Residual + no-slip + normalization stack is still sound — keeping them in exp2. Clear plateau in train loss suggests architecture ceiling, not optimization issue.

### 2026-04-16 — exp2: voxel-grid spatial mixer + Fourier pos
- **Hypothesis:** Per-point MLP can't see neighbors → can't predict local turbulence. Pool features onto a per-sample 32³ voxel grid (bbox-normalized), mix with 3D conv, gather back via trilinear `F.grid_sample`. Fourier features on pos (8 freqs, sin+cos) help represent high-freq spatial structure. Alternate 4 ResBlock + 4 VoxelMixer.
- **Change:** New VoxelMixer module. Added `__main__` guard so predict.py import is clean. Config fields exposed (hidden/n_blocks/grid_size/n_fourier/grad_clip). hidden=256, 4 mixer blocks, 15.5M params. grad_clip=1.0.
- **Result:** val/l2=1.0595 @ epoch 37 (38 ran before 30min timeout). train=0.0103. 4.8GB peak. 48s/epoch. run 9a9gbsue. Auto-predict OK → /mnt/new-pvc/predictions/apr16/tanjiro/368fd11/val.pt. Would be rank #3 (leader alphonse=0.92, #2 thorfinn=1.07).
- **Verdict:** KEPT — huge gain over exp1 (1.30→1.06, -18%). Voxel mixer delivers the spatial context MLPs lacked. Train loss still dropping at timeout (0.010), suggesting capacity+time both still leave room.
- **Notes:** Peak only 4.8GB of 96GB — lots of headroom to scale. Train loss curve is smooth, no instability with grad_clip=1. Best epoch late (37/38) — more epochs would help. For exp3: go bigger (hidden, blocks, grid) AND faster (bf16 autocast).
