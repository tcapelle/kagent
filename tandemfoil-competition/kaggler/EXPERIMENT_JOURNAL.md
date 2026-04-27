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

### 2026-04-27 — iter1: proven recipe (Transolver 192/6/6 + Fourier features) → val/loss=0.75, avg_surf_p=53.46
- **Hypothesis:** Replicate frieren's apr27 proven recipe (Transolver n_hidden=192, n_layers=6, n_head=6, slice=64, L1 loss, p_weight=3, train_subsample=40k volume nodes, bf16, bs=4, lr=5e-4, surf_weight=10, warmup+cosine over 35 epochs). Add Fourier features for position (8 scales, max_freq=16) and attention masking on padded tokens — both unused by previous leaders. Ought to land near 50-55 surf_p with the Fourier features potentially helping high-frequency turbulence features.
- **Change:** New `train.py` + `model.py`. `model.py` exposes `Transolver` with `FourierFeatures` (sin/cos at 8 log-spaced freqs over normalized x,z) and `mask` plumbed through `PhysicsAttention` so padded tokens can't pollute slice softmax. `train.py` adds `SubsampledDataset` (keep all surface nodes, randomly subsample 40k volume), L1 channel-weighted loss with `p_weight=3` on the pressure channel, AdamW + warmup-cosine LR, grad-clip 1.0, bf16 autocast, automatic checkpoint mirror to `/mnt/new-pvc/kagent/apr27-4/edward/checkpoints/model-<id>/`, and to `checkpoints/best.pt`.
- **Result:** epoch 35/35, val/loss=0.7537, **avg_surf_p=53.46** (single=0.74, rc=1.02, cruise=0.47, re_rand=0.79). 28.2 min wall, 10.5GB peak. W&B run `77aydpcl`. Predictions at `/mnt/new-pvc/predictions/apr27-4/edward/79894e7/`.
- **Verdict:** kept — solid baseline, slightly better than frieren's apr27 iter1 (54.26).
- **Notes:** Loss still dropping at epoch 35 (cosine bottom). Plan iter2 = warm-start this checkpoint with bs=2, full mesh, lr=2e-5, p_weight=3, ~10 epochs. Frieren's apr27 iter2 dropped 54.26→49.34 with this recipe; expect similar gain. predict.py crashed first due to importing `train.py` at top-level (sp.parse fired); fixed by moving model into `model.py`.
