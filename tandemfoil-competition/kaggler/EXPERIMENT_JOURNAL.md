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

### 2026-04-23 — v6-ema (FAILED, discarded)
- **Hypothesis:** EMA of model weights (decay=0.998) used for validation + as the saved checkpoint should give ~1-3% free improvement by averaging out late-training noise (researcher's suggestion).
- **Change:** `train.py` — create `ema_model = deepcopy(model)`; update after every optimizer step; validate and save using EMA.
- **Result:** best epoch 9, val/loss=3.042 (v3 2.634). **Avg val surf_p MAE 107.9 → 114.5 (+6.2%, WORSE).** Wall 31.2 min (9 epochs). W&B `kagent-tandemfoil/tck9u1e2`.
- **Verdict:** discarded, reset code.
- **Notes:** EMA averages over a 500-step window (~1.33 epochs with decay=0.998). On our 9-epoch training the current model is still mid-convergence — EMA weights lag ~1.3 epochs behind, and an older model in this regime is a worse model (val/loss still dropping each epoch, not oscillating around a minimum). EMA needs long training where weights oscillate near a minimum. Don't re-try EMA until base training is saturating.

### 2026-04-23 — v5-slice128 (FAILED, discarded)
- **Hypothesis:** doubling-ish slice token count (96 → 128) gives finer spatial slicing without much compute cost.
- **Change:** `train.py` — `slice_num: 96 → 128`.
- **Result:** best epoch 8, val/loss=3.070 (v3 2.634). **Avg val surf_p MAE 107.9 → 123.2 (+14.2%, WORSE).** Wall 33.1 min (8 epochs). Peak VRAM 85 GB (v3 was 71 GB). W&B `kagent-tandemfoil/1ifpng5a`.
- **Verdict:** discarded, reset code.
- **Notes:** Same failure mode as v4 — the extra compute from larger slice-weight tensor (`[B,H,N,G]` doubled) slowed each epoch by ~20% (251 s vs 210 s), dropping us from 9 → 8 epochs. Net: fewer cosine-schedule steps, incomplete LR decay. **Rule for this codebase: compute-per-epoch is the binding constraint — any change that slows each batch needs a matching `epochs` adjustment, and usually nets negative.** Next iteration should be a compute-neutral tweak (hparam, optimizer, EMA, data aug done off-device).

### 2026-04-23 — v4-eidetic (FAILED, discarded)
- **Hypothesis:** Transolver++ Eidetic attention (Ada-Temp per-point temperature + Rep-Slice Gumbel-Softmax slice weights) gives +12.6% surface-p on DrivAerNet++ in the paper. Swap the default `PhysicsAttention` for `PhysicsAttentionEidetic` behind a `use_eidetic` flag (default True).
- **Change:** `model.py` — added `PhysicsAttentionEidetic` class; `TransolverBlock`/`Transolver` accept `use_eidetic`. `train.py` — config flag `use_eidetic=True`, wire through `model_config`.
- **Result:** best epoch 8, val/loss=3.058 vs iter-3 2.634. **Avg val surf_p MAE 107.9 → 115.4 (+7.0%, WORSE).** Wall 33.4 min (8 epochs completed). W&B `kagent-tandemfoil/qsjr11qa` (name `nezuko/v4-eidetic`).
- **Verdict:** discarded. Reset code commit. Model.py restored to stock `PhysicsAttention`.
- **Notes:** Per-epoch Eidetic was actually BETTER than stock (at epoch 7: v4 val/loss=3.50 vs v3 4.24). The failure is pure compute overhead: Gumbel noise + softplus temperature projection + extra Linear pushed per-epoch from 210 s to 251 s (+20%), dropping us from 9 → 8 epochs. That's fewer cosine-annealing steps, incomplete LR decay. Next time try Eidetic with `epochs=8` so `T_max` matches the new wall-time budget — that's likely a true win.

### 2026-04-23 — v3-cosine-match (KEPT)
- **Hypothesis:** baseline's cosine LR schedule (`T_max=50`) was 82% unused when we hit the 30-min budget at epoch 9. Shrinking `epochs` to 10 (same wall-time, same LR init, same loss) lets the schedule actually decay to near-zero by the end, giving a fine-tuning phase that the baseline never got.
- **Change:** `train.py` one-liner — `epochs: 50 → 10`. Nothing else touched. Same MSE loss (reverted the Huber experiment first).
- **Result:** **best val/loss=2.634 at epoch 9/10** (baseline 2.984). Wall 31.1 min, 9 epochs completed (epoch 10 pre-empted by timeout at the top of the loop).
  - Per-split surface-p MAE:
    - `val_single_in_dist`: 139.1 → 130.0 (-6.6%)
    - `val_geom_camber_rc`: 137.5 → 122.4 (-11.0%)
    - `val_geom_camber_cruise`: 93.7 → 80.0 (-14.7%)
    - `val_re_rand`: 111.3 → 99.2 (-10.9%)
  - **Avg val surf_p MAE: 120.4 → 107.9 (-10.4%)**
  - W&B: `kagent-tandemfoil/1e53ehji` (name `nezuko/v3-cosine-match`).
- **Verdict:** kept. Ckpt committed at `22d262b`.
- **Notes:** Every split improved. The biggest jump (`cruise -14.7%`) is the easiest split, suggesting the model still benefits from extra fine-tuning even on in-distribution cases. Since the baseline epoch-9 LR was still ~83% of init, the iter-3 model effectively gets to spend ~3-4 epochs near minimum LR where gradient steps are small and refining. Cheapest possible improvement so far.

### 2026-04-23 — v2-huber-sp3 (FAILED, discarded)
- **Hypothesis:** surface loss as SmoothL1 (Huber, beta=1) is MAE-aligned and should improve surface-pressure MAE; per-channel weight `[1, 1, 3]` on surface emphasizes pressure.
- **Change:** `train.py` only — swap MSE→Huber on surface, add channel weights, keep volume MSE. surf_weight=10 unchanged.
- **Result:** best epoch 7 val/loss=3.924 (baseline was 2.984). **Avg val surf_p MAE 120.4 → 141.9 (+17.9%, WORSE).** Per-split: single_in_dist +35.4%, geom_camber_rc +2.9%, geom_camber_cruise +25.3%, re_rand +8.3%. W&B: `s2fgd22j` (name `nezuko/v2-huber-sp3`).
- **Verdict:** discarded. Reset code commit.
- **Notes:** Strong negative signal. Hypothesis: Huber saturates the gradient for large pressure errors (|range| up to 4k m²/s²), while MSE keeps scaling the signal with the error magnitude — exactly where the turbulent/stagnation spikes live. Val/loss is computed with MSE, so the ckpt selection also drifts when training optimizes a different objective. Don't repeat Huber alone; if re-attempting L1/MAE training, also switch the val/loss selection to avg surf_p MAE so criteria match.

### 2026-04-23 — v1-baseline (bf16, 192h/6L/96slice)
- **Hypothesis:** establish starting point with the already-improved Transolver template (bf16 autocast, 192h/6L/96slice, cosine LR, grad clip 1.0).
- **Change:** none — ran the committed `v1` code (`e1a0894`) unchanged. Config: lr=5e-4, bs=4, surf_weight=10, epochs=50 (capped by 30 min wall time).
- **Result:** best val/loss=2.984 at epoch 9 / 9 (timeout). Wall 31.4 min. VRAM 71 GB peak.
  - Per-split surface-p MAE (physical units):
    - `val_single_in_dist`: 139.1
    - `val_geom_camber_rc`: 137.5
    - `val_geom_camber_cruise`: 93.7
    - `val_re_rand`: 111.3
  - **Avg val surf_p MAE ≈ 120.4** (proxy for leaderboard metric)
  - W&B: `kagent-tandemfoil/pkb50fop` (name `nezuko/v1-bf16-192x6`)
- **Verdict:** kept — seeds the leaderboard; committed `best.pt` at `0c35d6c`.
- **Notes:** val_single_in_dist was noisy (14.5 → 11.1 → 15.7 → 11.5 → 8.6 → 6.6 → 6.4 → 7.0 → 4.5). Cosine LR decays fully over 50 epochs, so we stopped at ~18% of the schedule — the model is far from converged. First auto-predict invocation OOMed because the training process hadn't released GPU memory yet; rerun standalone after exit finished cleanly.
