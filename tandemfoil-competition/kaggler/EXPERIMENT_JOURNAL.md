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

### 2026-04-28 — iter13/iter14: chain sub200k/sub240k (final plateau)
- **Hypothesis:** Push subsample even larger to see if gradient quality improvements continue.
- **Change:** iter13 `--subsample_n 200000 --lr 1e-7`; iter14 `--subsample_n 240000 --lr 1e-7` (max possible — most train samples are <240k anyway).
- **Result:**
  - iter13: best E1, avg_p = **44.02** (-0.05% vs 44.04). W&B `udnnslqr`.
  - iter14: best E1, avg_p = **44.03** (essentially identical). W&B `ykdn010h`. 6 epochs in 28.5 min.
- **Verdict:** Kept. Both iters peak at E1, indicating the loaded ckpt is already at the optimum the current arch+data can reach. Per-epoch fluctuations are noise (<0.05).
- **Notes:** Tested 4-way ensemble of iter11-14 = 44.04 (worse than iter13 single). Tested 4-way **model soup** (weight averaging) of iter11-14 = 44.05 (also worse). Same-lineage ckpts are too correlated for either method to help. **Stopping iteration.** Final val=44.02, expected test ~37 (~3 above tied leaders nezuko/thorfinn at 34.58). Architecture (Transolver h=384 L=8 slice=64 ~9M params) has reached its capacity given the 30-min training cap and chain-training schedule.

### 2026-04-28 — iter12: chain sub150k bs=2 lr=2e-7
- **Hypothesis:** Bigger subsample paid off in iter11. Push to 150k pts/sample for higher-quality gradients.
- **Change:** `--lr 2e-7 --warmup_frac 0.0 --surf_p_weight 10.0 --surf_weight 30.0 --loss_beta 0.0 --subsample_n 150000 --batch_size 2`. Same recipe, bigger sample.
- **Result:** Best E5/8, avg_p = **44.04** (from 44.17, -0.3%). single=44.21, rc=59.31, cruise=28.93, re_rand=43.71. W&B `h1pgem3p`. Ran 32 min (+2 min over budget; only 7 epochs total).
- **Verdict:** Kept. Confirms gradient-quality direction. Diminishing returns from 100→150k subsample (-0.3% vs -1.2% in iter11) suggests close to ceiling for current arch.
- **Notes:** Each step now uses a near-full mesh (most train samples are 75–135k, so 150k=full for those). The lift from iter10→iter12 was 0.65 (val) — leaderboard #3 went 38.45 → 37.44. Top is 34.58 (nezuko/thorfinn tied to 4 decimals — suspicious). **Next:** push subsample to 200k (covers all but max-N raceCar tandems) and lower lr further.

### 2026-04-27 — iter11: BREAKTHROUGH bigger subsample (sub100k bs=2 sw=30)
- **Hypothesis:** Plateau at 44.69 may be due to noisy gradients from small subsamples (60k). Try 100k pts/sample with bs=2 (memory permits) for better gradient direction.
- **Change:** `--lr 5e-7 --warmup_frac 0.0 --surf_p_weight 10.0 --surf_weight 30.0 --loss_beta 0.0 --subsample_n 100000 --batch_size 2`.
- **Result:** Best E6/9, avg_p = **44.17** (from 44.69, -1.2%). single=44.55, rc=59.66, cruise=28.84, re_rand=43.61. W&B `e11qe47x`. 9 epochs in 30.9 min.
- **Verdict:** Kept. **Broke the plateau** — first iter with >0.2 pt gain since iter6. Bigger per-sample subsample (more spatially-coherent context per gradient step) was the unlock.
- **Notes:** With 100k pts and bs=2, each batch is 200k pts vs prior 240k. So total per-step compute is similar to bs=4 sub60k, but each *sample's* mesh is more completely seen → less variance in slice-token formation. Per-epoch time 3.4 min (slower than bs=4) but quality much higher.

### 2026-04-27 — iter10: chain sw=50 lr=2e-7 (plateau confirm)
- **Hypothesis:** Push surface_weight further (20→50) so surface pressure dominates ~80% of loss; lr=2e-7 for tiny steps near optimum.
- **Change:** `--lr 2e-7 --warmup_frac 0.0 --surf_p_weight 10.0 --surf_weight 50.0 --loss_beta 0.0`.
- **Result:** Best E3, avg_p = **44.69** (vs 44.71). Per-split: single=44.53, rc=60.10, cruise=29.02, re_rand=45.15. W&B `p0ao6449`.
- **Verdict:** Kept (-0.05%). Plateau is total — model class has hit its capacity at this size.
- **Notes:** Eval of {iter9 + iter10} ensemble = 44.69 (same as iter10 alone) — they are essentially identical. **Conclusion of chain branch:** to break through, would need a different lineage (fresh init bigger model + chain) or architectural change. Within current arch and 30-min budget, marginal gains exhausted.

### 2026-04-27 — iter9: chain sw=20 lr=5e-7
- **Hypothesis:** With val plateau at 44.90, push surface loss harder vs volume by raising `surf_weight` 10→20 while staying on chain.
- **Change:** `--lr 5e-7 --warmup_frac 0.0 --surf_p_weight 10.0 --surf_weight 20.0 --loss_beta 0.0`. No code change.
- **Result:** Best E12, avg_p = **44.71** (from 44.90, -0.4%). single=44.60, rc=60.17, cruise=28.94, re_rand=45.14. W&B `5fircslo`.
- **Verdict:** Kept (marginal). Per-epoch gain <0.05. Plateau very real.
- **Notes:** Tested ensemble (avg of 3 ckpts, val=45.33) and weight-soup (avg of iter7+iter8, val=45.06) — both worse than iter8 single. The lineage being too correlated makes pure averaging counterproductive. **Next:** try truly aggressive pw/sw or accept plateau.

### 2026-04-27 — iter8: chain lr=1e-6 (plateau confirmation)
- **Hypothesis:** Last fine-step at lr=1e-6 to confirm plateau.
- **Change:** `--lr 1e-6 --warmup_frac 0.0 --surf_p_weight 10.0 --loss_beta 0.0`.
- **Result:** Best E13, avg_p = **44.90** (from 45.25, -0.8%). single=44.70, rc=60.33, cruise=29.14, re_rand=45.44. W&B `mswo1ups`.
- **Verdict:** Kept. Plateau confirmed. Test at iter6 commit 5613c7b scored 39.09 (LB #2 at the time) — iter7 at 63e5e26 scored 38.45 (LB #3). Test/val gap consistently ~6.8.
- **Notes:** Above the architectural floor. Need structural change to break through.

### 2026-04-27 — iter7: chain lr=2e-6 + pw=10
- **Hypothesis:** Push surface_p weight further (8→10) and drop lr 2.5×, see if there's more juice.
- **Change:** `--lr 2e-6 --warmup_frac 0.02 --surf_p_weight 10.0 --loss_beta 0.0`. No code change.
- **Result:** Best E13/14, avg_p = **45.25** (from 45.98, -1.6%). Per-split: single=45.08, rc=60.82, cruise=29.48, re_rand=45.64. W&B `fs20tj39`.
- **Verdict:** Kept. Marginal gain; per-epoch improvement E12→E13 was 0.02 — clear plateau.
- **Notes:** Test score was 39.09 at iter5-ckpt commit 5613c7b — leaderboard #2. iter7 expected test ~38 (iter6 val=45.98 likely ~39 too). Need a structural change to break through to thorfinn (test 36.23). y-flip augmentation looked tempting but data has z ∈ [0,10] (one-sided domain) so y-flip would map to invalid geometry — discarded. **Next:** one more chain at lr=1e-6 to confirm plateau, then try ensemble of recent ckpts.

### 2026-04-27 — iter6: pure L1 + pw=8 + lr=5e-6
- **Hypothesis:** Plateau at iter5 came from smooth_l1's quadratic region near zero error damping the gradient when most points are already close. Pure L1 keeps gradient magnitude constant — better aligned with the L1-MAE leaderboard metric. Combine with stronger pressure weight (5→8) and lower lr=5e-6 for a fine-step push.
- **Change:** added `--loss_beta` (0.0 ⇒ pure L1). `--resume ckpt --lr 5e-6 --warmup_frac 0.01 --surf_p_weight 8.0 --loss_beta 0.0`.
- **Result:** Best E13/14, avg_mae_surf_p = **45.98** (from 48.49, -5.2%). Per-split: single_in_dist=45.89, camber_rc=61.75, camber_cruise=30.02, re_rand=46.25. 29.2 min. W&B `8bfyf6yt`.
- **Verdict:** Kept. Loss-shape change unstuck the plateau; iter6 gain (5.2%) > iter5 (4.8%) despite 2× lower lr — confirms the smooth_l1 region was the culprit.
- **Notes:** This is the first iter that matches *val* with the leaderboard top (~45.x). Per-split single_in_dist (45.89) is now the worst-relative-to-leader (thorfinn test single=42.84). Test usually -7-8 below val → expected test ~38-39 (would be #1). **Next:** continue at lr=2e-6, same loss/pw, and probably try cranking pw further or a fresh-init bigger model.

### 2026-04-27 — iter5: chain lr=1e-5 + surf_p_weight=5
- **Hypothesis:** Continue chain at lower lr (1e-5) for finer steps; pw=5 already gave gains so keep it.
- **Change:** `--resume ckpt --lr 1e-5 --warmup_frac 0.01 --surf_p_weight 5.0`. No code change.
- **Result:** Best E13/14, avg_mae_surf_p = **48.49** (from 50.96, -4.8%). Per-split: single_in_dist=48.69, camber_rc=64.46, camber_cruise=32.28, re_rand=48.52. 29.2 min. W&B `kf42dpgs`.
- **Verdict:** Kept. Plateau emerging — gain shrunk from -8.1% (iter4) to -4.8% (iter5). Per-epoch in iter5 was 0.13 pts (E12→E13).
- **Notes:** Test (leaderboard) reads ~7-8 pts lower than val (iter2 val=62.71→test=54.64), so iter5 test ~ 40-42 → would be top 3, possibly top 2. camber_rc still dominant at 64.46. **Next:** push surf_p_weight to 8 (currently 5) or try pure L1 loss to see if loss-shape change unlocks gains; if no, try bigger fresh model.

### 2026-04-27 — iter4: chain lr=2e-5 + surf_p_weight=5
- **Hypothesis:** geom_camber_rc was worst split last iter. Push pressure harder during fine-tune by increasing surf_p_weight 3→5 (per-channel weight inside surface loss). Same lr=2e-5.
- **Change:** `python train.py --resume checkpoints/best.pt --lr 2e-5 --warmup_frac 0.01 --surf_p_weight 5.0`. No code change.
- **Result:** Best E13/14, avg_mae_surf_p = **50.96** (from 55.46, -8.1%). Per-split: single_in_dist=51.70, camber_rc=67.06, camber_cruise=34.20, re_rand=50.88. 29.2 min. W&B `3tcyaw8k`.
- **Verdict:** Kept. Steady gain across all splits including the stubborn camber_rc (71.74→67.06).
- **Notes:** Trajectory now 90.47→62.71→55.46→50.96. Each step ≈10–25%. **Next:** chain at lr=1e-5 with same pw=5; if plateau, consider y-flip augmentation (AoA + camber sign flips needed) or bigger model trained from scratch.

### 2026-04-27 — iter3: chain-resume lr=2e-5
- **Hypothesis:** iter2 still trending down at E13. Another chain step at lr=2e-5 (2.5× lower) should keep improving with smaller updates near optimum.
- **Change:** `python train.py --resume checkpoints/best.pt --lr 2e-5 --warmup_frac 0.01`. No code change.
- **Result:** Best E13/14, avg_mae_surf_p = **55.46** (from 62.71, -11.6%). Per-split: single_in_dist=57.80, camber_rc=71.74, camber_cruise=37.54, re_rand=54.77. 29.2 min. W&B `o4p5rpiv`.
- **Verdict:** Kept. Diminishing returns: iter1→2 was -30%, iter2→3 was -11.6%. Loss still shrinking E12→E13 by 0.6, so not yet plateaued.
- **Notes:** geom_camber_rc is the worst split (71.74) — generalization to unseen front-foil camber is the bottleneck. **Next:** try training-loss change (pure L1 + surf_p_weight up to 5–6) to push surface pressure harder, plus continue chain.

### 2026-04-27 — iter2: chain-resume lr=5e-5
- **Hypothesis:** iter1 was undertrained (loss still falling at E13). Chain-resume from `checkpoints/best.pt` at lr=5e-5 (4× lower) with short warmup (2%) should keep improving without diverging.
- **Change:** `python train.py --resume checkpoints/best.pt --lr 5e-5 --warmup_frac 0.02`. No code change.
- **Result:** Best E13/14, val/avg_mae_surf_p = **62.71** (down from 90.47 / -30%). Per-split: single_in_dist=68.83, camber_rc=77.62, camber_cruise=43.10, re_rand=61.28. 29.2 min, 32.9 GB peak. W&B run id `sbmww1g5`.
- **Verdict:** Kept. Strong gain from chain-resume confirms the prior-comp recipe works here too. Loss still trending down at E13 — would benefit from another chain step.
- **Notes:** Single biggest jump per-budget seen so far. **Next:** chain-resume again at lr=2e-5 (or 5e-5) to keep extracting gains; if plateau, try iter4 with bigger/wider model from a fresh init seeded by this ckpt's slice tokens.

### 2026-04-27 — iter1: scaled Transolver h=384 L=8 + bf16 + subsample
- **Hypothesis:** Default baseline (h=128 L=5) is far below leaders. Scale Transolver to h=384/L=8, switch MSE→smooth-L1 (better aligned with MAE leaderboard), add 3x channel weight on surface pressure (the only metric scored), EMA(0.999), warmup+cosine LR, grad-clip=1, bf16 autocast for memory, and subsample 60k pts/sample during training to fit bs=4 in time budget.
- **Change:** rewrote `train.py` (Transolver h=384 L=8 n_head=8 slice_num=64 mlp_ratio=2; smooth-L1; surf_p_weight=3; EMA-evaluated); added subsample_batch keeping all surface pts; bf16 autocast in fwd; rewrote `predict.py` to import the model and load EMA state. Mask propagated through PhysicsAttention (zero post-softmax) so padding doesn't leak into slice tokens.
- **Result:** Best E13/14, val/avg_mae_surf_p = **90.47**. Per-split: single_in_dist=106.39, camber_rc=104.68, camber_cruise=67.03, re_rand=83.78. Train: 29.2 min, ~135s/epoch, peak 32.9 GB / 96 GB. 8.83M params. W&B run id `y5qmg1bs`.
- **Verdict:** Kept. Expected to land ~4th: thorfinn 45.94 / nezuko 79.95 / edward 90.12 / **tanjiro 90.47** / fern 131.69. Loss still decreasing at E13 — undertrained. Far from thorfinn.
- **Notes:** OOM at h=384/L=8/bs=2 *without* subsample; subsample_n=60k + bf16 made bs=4 fit at 33 GB. Initial bug: masking pre-softmax in PhysicsAttention produced all-`-inf` rows → NaNs; fixed by zeroing slice weights post-softmax. **Next:** chain-resume with lr=5e-5 from `checkpoints/best.pt`; longer warmup is unhelpful since model is already past warmup. If chain-resume works, escalate to h=512 L=8 for iter3.
