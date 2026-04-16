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
