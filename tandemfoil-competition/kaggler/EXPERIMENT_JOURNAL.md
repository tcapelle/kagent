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

### 2026-04-27 — iter9 chain again, LR=1e-5 (KEPT) — RANK 1
- **Hypothesis:** Smooth out remaining residuals with LR=1e-5 cosine.
- **Change:** No code change; CLI: `--load_from checkpoints/best.pt --epochs 11 --finetune_epochs 0 --batch_size 2 --subsample_n 0 --lr 1e-5`.
- **Result:** Best epoch 11 (last), **val avg_surf_p=47.76**, **test avg_surf_p=41.53**. 27.5 min. Trajectory smooth: 49→49→50→49→48.6→49.6→48.3→48.0→47.9→47.9→47.76.
- **Verdict:** Kept (commit `4241092`). **Rank 1** on leaderboard, ahead of frieren (42.11) by 0.58.
- **Notes:** Improvement getting smaller per iter (iter8→9 dropped val by 0.79). Continue chaining at LR=5e-6; if plateau, switch to ensemble of recent ckpts.

### 2026-04-27 — iter8 chain again, LR=2e-5 (KEPT) — rank 2
- **Hypothesis:** Lower LR (2e-5 cosine) from iter7 ckpt for smoother continued descent.
- **Change:** No code change; CLI: `--load_from checkpoints/best.pt --epochs 11 --finetune_epochs 0 --batch_size 2 --subsample_n 0 --lr 2e-5`.
- **Result:** Best epoch 11 (last), **val avg_surf_p=48.55**, **test avg_surf_p=42.31**. 27.4 min. Trajectory: 52→52→51→52→51→50→49.5→49.6→48.9→48.9→48.5 — smooth descent throughout.
- **Verdict:** Kept (commit `d5eb122`). Now **rank 2** on leaderboard, ahead of alphonse (42.62) and historical thorfinn 26d8011 (42.90); 0.20 away from frieren #1 (42.11).
- **Notes:** Loss still decreasing at last epoch — chain again with LR=1e-5.

### 2026-04-27 — iter7 chain again, LR=3e-5 (KEPT)
- **Hypothesis:** Iter6 was still improving at the last epoch — chain again with slightly lower LR (3e-5 vs 5e-5) to push further along the same trajectory.
- **Change:** No code change; CLI: `--load_from checkpoints/best.pt --epochs 11 --finetune_epochs 0 --batch_size 2 --subsample_n 0 --lr 3e-5`.
- **Result:** Best epoch 10, **val avg_surf_p=50.05**, **test avg_surf_p=43.72**. 27.4 min. Trajectory bouncy (52→58→52→57→52→51→50.6→50.05) but trending down.
- **Verdict:** Kept (commit `11c6bb1`). Now ~0.83 away from frieren #1 (42.11), 0.82 from alphonse #2 (42.62), and 0.82 from my historical best 26d8011 (42.90).
- **Notes:** Per-iter improvement is decreasing but still meaningful (iter5→6 dropped val by 8.4 pts; iter6→7 dropped by 2.6 pts). Iter8 try LR=2e-5 — likely smoother trajectory and another small drop.

### 2026-04-27 — iter6 pure full-mesh chain fine-tune from iter5 (KEPT)
- **Hypothesis:** Iter5's full-mesh FT phase was the load-bearing step. Skip the base subsample phase entirely; do 11 epochs of full-mesh batch=2 with cosine LR from 5e-5 → 0. Should push val below 60.
- **Change:** No code change; CLI: `--load_from checkpoints/best.pt --epochs 11 --finetune_epochs 0 --batch_size 2 --subsample_n 0 --lr 5e-5`. Uses cosine over all 11 epochs.
- **Result:** **val avg_surf_p=52.68** (best epoch 11, last). 27.5 min. Trajectory: 62 → 60 → 56 → 53 → 52.7 — still improving at the final epoch.
- **Verdict:** Kept (commit `ad7313b`).
- **Notes:** Improvement is monotonic and not plateauing within budget. Iter7 should chain again (probably with slightly lower LR, e.g., 2e-5, since we're deeper into convergence). Reading scores: pending scoring service refresh.

### 2026-04-27 — iter5 chain fine-tune + full-mesh fine-tune phase (KEPT)
- **Hypothesis:** Continue training from iter4 ckpt (warm start) and add a 5-epoch full-mesh batch=2 fine-tune phase at the end. Subsample-trained model never sees the dense per-sample node distribution it's evaluated on; a brief full-mesh pass should close that gap.
- **Change:** train.py — `load_from`, `finetune_epochs`, `finetune_batch_size=2`, `finetune_lr=5e-5`. After base epochs, switch to a no-subsample loader and a fixed lower LR. Cosine schedule covers base epochs only; FT phase is fixed-LR.
- **Run config:** `--load_from checkpoints/best.pt --epochs 25 --finetune_epochs 5 --lr 2e-4 --finetune_lr 5e-5`. 20 base epochs (subsample 40K, batch=8) + 5 FT epochs (full mesh, batch=2). bf16. 26.9 min total.
- **Result:** Best epoch 23, **val avg_surf_p=61.07** (down from iter4's 99.26), **test avg_surf_p=54.02**. Trajectory shows the FT phase is the load-bearing step: base end (epoch 20) was 88.55; first FT epoch dropped to 69, third FT epoch to 61. Base epochs 16-20 plateaued at 87-90.
- **Verdict:** Kept (commit `a14232b`).
- **Notes:** Full-mesh FT phase unlocked ~30 % improvement in 5 epochs — the subsample/full-mesh distribution gap was real. Plateaus after ~3 FT epochs, suggesting the LR=5e-5 is well-tuned but capacity may be the new bottleneck. Currently 4th on leaderboard (frieren 42.11, historical thorfinn 42.90, tanjiro 51.42, me 54.02). Iter6 should chain again, possibly with longer FT phase or lower LR.

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
