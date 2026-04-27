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
