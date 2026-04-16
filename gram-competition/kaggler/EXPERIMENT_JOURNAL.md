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

### 2026-04-16 — iter4: per-timestep decoder + velocity tendency features
- **Hypothesis:** Model should know which output timestep it's predicting. Replace Linear(hidden, T_OUT*3) with per-step time-embedding + Linear(hidden, 3). Also add velocity tendency features (v[-1]-v[-2], (v[-1]-v[0])/T_IN).
- **Change:** train.py/predict.py — new decoder: time_emb[k] added to point features, shared Linear(hidden, 3) head applied per timestep. Added 6 velocity tendency channels to input.
- **Result:** val/l2=1.2040 at epoch 108. 3.34M params, 1.5GB VRAM, 14s/epoch, 120 epochs in 27min.
- **Verdict:** DISCARDED — significantly worse than iter2 (1.0751).
- **Notes:** Root cause: the new decoder has 1/5th the decoder-layer params of iter2. Iter2's Linear(hidden, T_OUT*3) is mathematically 5 independent heads; iter4's shared Linear(hidden,3)+time_emb bias is only a SHIFT of features per timestep before a single shared weight matrix. Takeaway: don't trade independent per-step weights for a bias-only conditioning. The velocity tendency features are likely redundant with raw v_in[t] inputs and couldn't be isolated as helpful.

### 2026-04-16 — iter3: scaled Transolver (d=384, 8 blocks, 128 slices)
- **Hypothesis:** Iter2 Transolver had huge headroom — 3.28M params fits in 1.4GB. Triple the capacity (10.5M params) + bigger slice pool (128 vs 64) should move the needle.
- **Change:** train.py — hidden 256→384, blocks 6→8, slices 64→128, subsample 16384→24000, dropout 0.0→0.05, lr 1e-3→7e-4, epochs 120→90.
- **Result:** val/l2=1.0860 at epoch 60 (ran out of 30min budget). 10.51M params, 2.3GB VRAM, 25s/epoch. Still decreasing at end — undertrained.
- **Verdict:** DISCARDED — worse than iter2 (1.0751). Scale is not the problem; budget is. A bigger model converges slower.
- **Notes:** Confirms 30-min wallclock is the binding constraint. Next iteration: keep iter2's small backbone (fits in budget) and add targeted architectural priors instead of capacity.

### 2026-04-16 — iter2: Transolver (soft-slice attention)
- **Hypothesis:** Per-point MLP (iter1) lacks spatial interaction; Transolver's O(N·M) slice attention should model wake/vortex long-range structure.
- **Change:** train.py — 6 Transolver blocks, d=256, M=64 slices, 8 heads. Kept residual head + no-slip BC + Fourier pos features + velocity normalization. Train subsample=16k, AMP bf16, lr=1e-3, 120 epochs, cosine decay.
- **Result:** val/l2_error=1.0751 at epoch 108. 3.28M params, 1.4GB VRAM, 13s/epoch. Train loss 5.40→1.57 (converged).
- **Verdict:** KEPT — massive jump from iter1 (1.27). Would be #1 on apr16 leaderboard (current #1 alphonse 1.32).
- **Notes:** Training loss still slowly decreasing at end (not overfit). Model is tiny — huge headroom to scale. Next: bigger model + more capacity; possibly full 100k train resolution.

### 2026-04-16 — iter1: residual MLP with Fourier pos + no-slip BC
- **Hypothesis:** A solid baseline — residual prediction from v_in[-1], no-slip BC enforcement, velocity normalization, Fourier pos encoding, and a deeper MLP should crush naive baseline.
- **Change:** train.py — 8-block ResMLP (512 hidden), Fourier pos features (10 freqs), velocity norm, airfoil indicator input, residual head zeroed-init, no-slip zero-out at airfoil indices. AMP bf16, 20k subsample during train, cosine LR.
- **Result:** plateau val/l2≈1.27 (killed at epoch 35/80). 8.47M params, 1.5GB VRAM.
- **Verdict:** DISCARDED — no spatial interaction means per-point MLP can't fit turbulent structure. Train loss also plateau-ed at ~3.3, saturated.
- **Notes:** Established that residual+normalization framing works (val/l2 dropped from 1.62→1.27 in 35 epochs). The MLP is capacity-bound on the per-point task, not architecture-bound.

