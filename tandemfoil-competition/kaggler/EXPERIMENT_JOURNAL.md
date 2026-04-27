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

### 2026-04-27 — iter4: 3-way ensemble of iter1+iter2+iter3 (weights 0.2/0.3/0.5)
- **Hypothesis:** Even chain-correlated models can give a small boost when blended (frieren saw 35.27→34.41 with 6-way ensembles). Worth a quick win while a fresh diverse model trains.
- **Change:** New `ensemble.py` reading per-split prediction tensors from N source commits and averaging. Submitted with weights 0.2/0.3/0.5 favoring iter3 (best single model).
- **Result:** Ensemble predictions written to `/mnt/new-pvc/predictions/apr27-4/edward/3a1cf7a/`. Best single model val/loss = 0.728; ensemble val score not measured directly (no val labels for test ensemble).
- **Verdict:** kept as a hedge; will compare scores once leaderboard refreshes.
- **Notes:** All three sources are warm-chained from iter1, so they're highly correlated — expect modest gain. Plan iter5 = train a fundamentally different model (n_hidden=256, slice_num=128, p_weight=1) for genuine ensemble diversity.

### 2026-04-27 — iter3: deeper warm-start chain (lr=5e-6) → val/loss=0.728, avg_surf_p=51.27 (-1%)
- **Hypothesis:** iter2's warm-start chain dropped surf_p only ~3% (53.46→51.87). Frieren's chain dropped ~5 surf_p per iter; mine is converging faster. Try one more chain step at `lr=5e-6` (4x lower) for 10 epochs to squeeze the last gains before pivoting.
- **Change:** `python train.py --warm_start /mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-yqqu10sg/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 5e-6 --p_weight 3.0 --epochs 10 --warmup_epochs 1`. Run `w2qvsfx1`.
- **Result:** epoch 10/10, val/loss=**0.7280**, **avg_surf_p=51.27** (single=0.70, rc=1.00, cruise=0.45, re_rand=0.77). 26.2 min wall, 29.3GB peak. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/5d05ebb/`. Auto-submit failed mid-run with OOM (something else briefly took 88GB on the GPU); ran predict.py manually after killing the orphan train workers.
- **Verdict:** kept — marginal 1% gain, chain has plateaued.
- **Notes:** Likely the Fourier features made iter1 already extract spatial info that frieren's chain unlocked between iters. To get to leaderboard-leader territory (~42 surf_p, possibly 35 with ensembles), need a fundamentally different angle. Plan iter4 = try a fresh, larger model (n_hidden=256, slice_num=128, n_layers=8) trained from scratch. If it lands near iter3, switch to ensembling iter1+iter2+iter3 predictions.

### 2026-04-27 — iter2: warm-start bs=2 full-mesh chain → val/loss=0.74, avg_surf_p=51.87 (-3%)
- **Hypothesis:** Frieren's apr27 iter2 found that warm-starting iter1 with `batch_size=2 + train_subsample=0` + `lr=2e-5` + `p_weight=3` for 10 epochs got 49.34 surf_p (from 54.26). The hypothesis is that the 40k subsample drops 60% of the volume mesh — refining on the full mesh at low LR teaches the model the spatial structure it missed.
- **Change:** `python train.py --warm_start /mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-77aydpcl/checkpoint.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --p_weight 3.0 --epochs 12 --warmup_epochs 1`. Run `yqqu10sg`.
- **Result:** stopped at epoch 11 (timeout 28.7 min, 29.3GB peak). Best val/loss=**0.7358**, **avg_surf_p=51.87** (single=0.71, rc=1.01, cruise=0.45, re_rand=0.77). All four splits improved a bit vs iter1. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/0fa22ab/`.
- **Verdict:** kept — modest 3% gain on surf_p, 2% on val/loss. Single-foil split still has the highest absolute val loss (0.71).
- **Notes:** The chain is converging slowly — frieren's chain dropped 4-5 surf_p per iter, mine dropped only 1.6. Might be that my Fourier features make iter1 already closer to convergence. Next: try a larger model (n_hidden=256, slice_num=128) or push lr down to 5e-6 and continue chaining for marginal gains.

### 2026-04-27 — iter1: proven recipe (Transolver 192/6/6 + Fourier features) → val/loss=0.75, avg_surf_p=53.46
- **Hypothesis:** Replicate frieren's apr27 proven recipe (Transolver n_hidden=192, n_layers=6, n_head=6, slice=64, L1 loss, p_weight=3, train_subsample=40k volume nodes, bf16, bs=4, lr=5e-4, surf_weight=10, warmup+cosine over 35 epochs). Add Fourier features for position (8 scales, max_freq=16) and attention masking on padded tokens — both unused by previous leaders. Ought to land near 50-55 surf_p with the Fourier features potentially helping high-frequency turbulence features.
- **Change:** New `train.py` + `model.py`. `model.py` exposes `Transolver` with `FourierFeatures` (sin/cos at 8 log-spaced freqs over normalized x,z) and `mask` plumbed through `PhysicsAttention` so padded tokens can't pollute slice softmax. `train.py` adds `SubsampledDataset` (keep all surface nodes, randomly subsample 40k volume), L1 channel-weighted loss with `p_weight=3` on the pressure channel, AdamW + warmup-cosine LR, grad-clip 1.0, bf16 autocast, automatic checkpoint mirror to `/mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-<id>/`, and to `checkpoints/best.pt`.
- **Result:** epoch 35/35, val/loss=0.7537, **avg_surf_p=53.46** (single=0.74, rc=1.02, cruise=0.47, re_rand=0.79). 28.2 min wall, 10.5GB peak. W&B run `77aydpcl`. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/79894e7/`.
- **Verdict:** kept — solid baseline, slightly better than frieren's apr27 iter1 (54.26).
- **Notes:** Loss still dropping at epoch 35 (cosine bottom). Plan iter2 = warm-start this checkpoint with bs=2, full mesh, lr=2e-5, p_weight=3, ~10 epochs. Frieren's apr27 iter2 dropped 54.26→49.34 with this recipe; expect similar gain. predict.py crashed first due to importing `train.py` at top-level (sp.parse fired); fixed by moving model into `model.py`.
