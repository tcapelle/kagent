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

### 2026-04-27 — iter4 d192/L6/s64/h6 + L1 + 40K subsample + batch=8 + bf16 (KEPT)
- **Hypothesis:** Match prior agent's recipe (L1 loss in normalized space, batch=8 with random 40K-node subsampling per sample, surface always preserved). Should produce a sane baseline.
- **Change:** train.py — L1 loss for vol+surf (normalized), `subsample()` keeps all surface nodes + random fill to 40K, batch_size=8 (train) / 4 (val), bf16, surf_weight=10, epochs=30, cosine T_max=30, select on `avg_mae_surf_p`. predict.py uses bf16 autocast.
- **Result:** 30 epochs in 21.8 min, peak 19 GB. Val avg_surf_p improves monotonically: epoch1=246 → epoch10=132 → epoch20=110 → **epoch30=99.26**. Test **avg_surf_p=95.11**. Best epoch 30.
- **Verdict:** Kept — first run that uses the right recipe; ckpt saved at `checkpoints/best.pt` (commit 23b767d).
- **Notes:** Loss still decreasing at epoch 30 but cosine LR fully decayed → undertrained at end. Plateauing in epochs 24-30 (drop only 102→99). Big gap to prior thorfinn 42.9 — they likely got there through chain fine-tuning + a full-mesh fine-tune phase. Iter5 will load this ckpt and add a full-mesh fine-tune phase + more total epochs.

### 2026-04-27 — iter3 d256/L6/s64 + bf16 (DISCARDED)
- **Hypothesis:** Wider Transolver (d=256, L=6, s=64) + bf16 + select on `avg_mae_surf_p` would beat the 192/6/64 baseline.
- **Change:** train.py — model d=256/L=6/s=64/heads=8/mlp_ratio=2; bf16 autocast; MSE+MSE; surf_weight=10; mirror best ckpt to PVC; free GPU before auto-submit.
- **Result:** 9 epochs, peak 77 GB. Best epoch 9, **avg_mae_surf_p=149.94 (val)**, **128.75 (test)**. Auto-submit succeeded. Sanity check: prior thorfinn/26d8011 at 42.9 (test) was reachable from a checkpoint with d=192/L=6/s=64 (file `model-zwxdd9sm`) which I evaluated to ≈50 (val) / 42.9 (test). My iter3 run is a clear regression.
- **Verdict:** Discarded.
- **Notes:** Root-caused by reading the prior agent's session log (`/mnt/new-pvc/kagent/apr27/thorfinn/iter_1_*.jsonl`): the prior thorfinn used **L1 loss (vol+surf)**, **batch_size=8 with random 40 K-node subsampling per sample** (full-mesh fine-tune at batch 2), bf16 autocast. The current template ships MSE without subsampling — that's why even matching their `d=192` config in MSE never gets close to their 42.9 score. Iter4 will adopt L1 + 40 K subsample + batch=8.

### 2026-04-27 — iter2 d256/L8/s96 + bf16 (DISCARDED — OOM)
- **Hypothesis:** Bigger Transolver (frieren-like, n_hidden=256, n_layers=8, slice_num=96) + bf16 mixed precision should fit in 30 min and beat the 192/6/64 baseline.
- **Change:** train.py — model d256/L8/s96, autocast(bf16), MSE+MSE, surf_weight=10, select on `avg_mae_surf_p`, mirror best ckpt to PVC, free GPU before auto-submit subprocess.
- **Result:** OOM at epoch 1 step 0. Peak ~94.9 GB (94.97 GB total). Even with bf16 the activations from {B=4, N≈242K, slice_num=96, L=8} blow past the budget.
- **Verdict:** Discarded (`git reset --hard HEAD~1`).
- **Notes:** baseline d192/L6/s64 fp32 was 74 GB peak. Scaling factors that hurt most: slice_num 64→96 (≈1.5×) and L 6→8 (≈1.33×). For iter3, drop slice_num back to 64 and depth back to 6, only widen hidden to 256. Estimated peak: 74×(256/192)²/2 ≈ 66 GB (bf16). Comfortable.

### 2026-04-27 — iter1 SmoothL1 surface loss (DISCARDED)
- **Hypothesis:** Switching surface loss from MSE to SmoothL1 (β=1.0) would align training with the leaderboard's MAE metric and improve surface pressure accuracy.
- **Change:** train.py — Transolver d192/L6/s64 (n_head=6, mlp_ratio=2). Surface loss = SmoothL1(diff, β=1.0); volume loss = MSE. Selected best by `avg_mae_surf_p` (mean across 4 val splits) instead of `val/loss`. Refactored model classes into `model.py`. Added PVC checkpoint mirror.
- **Result:** 8 epochs in 32.1 min; train: vol=0.44 surf=0.09. Best epoch 7, **avg_mae_surf_p=137.7** — 3× worse than prior thorfinn baseline (42.90 with MSE+MSE+surf_weight=10). Auto-submit OOM'd because train.py kept the model on GPU when spawning predict.py subprocess.
- **Verdict:** Discarded (`git reset --hard HEAD~1`).
- **Notes:** SmoothL1 with β=1.0 has gradient |d| (≤1) inside the quadratic region — half of MSE's 2|d|. With surf_weight=10 unchanged, the *effective* surface gradient was halved, so training under-weighted the surface and the model learned a worse pressure field. To use SmoothL1 productively, double surf_weight or use β<1.0. Filed for later: also fix the auto-submit OOM by deleting the model and emptying CUDA cache before spawning predict.py.
