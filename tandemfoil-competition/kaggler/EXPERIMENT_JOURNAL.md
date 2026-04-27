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

### 2026-04-27 — iter1: scaled Transolver h=384 L=8 + bf16 + subsample
- **Hypothesis:** Default baseline (h=128 L=5) is far below leaders. Scale Transolver to h=384/L=8, switch MSE→smooth-L1 (better aligned with MAE leaderboard), add 3x channel weight on surface pressure (the only metric scored), EMA(0.999), warmup+cosine LR, grad-clip=1, bf16 autocast for memory, and subsample 60k pts/sample during training to fit bs=4 in time budget.
- **Change:** rewrote `train.py` (Transolver h=384 L=8 n_head=8 slice_num=64 mlp_ratio=2; smooth-L1; surf_p_weight=3; EMA-evaluated); added subsample_batch keeping all surface pts; bf16 autocast in fwd; rewrote `predict.py` to import the model and load EMA state. Mask propagated through PhysicsAttention (zero post-softmax) so padding doesn't leak into slice tokens.
- **Result:** Best E13/14, val/avg_mae_surf_p = **90.47**. Per-split: single_in_dist=106.39, camber_rc=104.68, camber_cruise=67.03, re_rand=83.78. Train: 29.2 min, ~135s/epoch, peak 32.9 GB / 96 GB. 8.83M params. W&B run id `y5qmg1bs`.
- **Verdict:** Kept. Expected to land ~4th: thorfinn 45.94 / nezuko 79.95 / edward 90.12 / **tanjiro 90.47** / fern 131.69. Loss still decreasing at E13 — undertrained. Far from thorfinn.
- **Notes:** OOM at h=384/L=8/bs=2 *without* subsample; subsample_n=60k + bf16 made bs=4 fit at 33 GB. Initial bug: masking pre-softmax in PhysicsAttention produced all-`-inf` rows → NaNs; fixed by zeroing slice weights post-softmax. **Next:** chain-resume with lr=5e-5 from `checkpoints/best.pt`; longer warmup is unhelpful since model is already past warmup. If chain-resume works, escalate to h=512 L=8 for iter3.
