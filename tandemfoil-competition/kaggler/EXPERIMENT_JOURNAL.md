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
