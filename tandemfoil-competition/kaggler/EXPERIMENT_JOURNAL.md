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

### 2026-04-23 — baseline Transolver 128h/5L
- **Hypothesis:** establish a floor with the template model.
- **Change:** wired predict.py, guarded train.py under `main()`. Default config: n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2, lr=5e-4, bs=4, epochs=50, surf_weight=10.
- **Result:** val/loss=3.1000 at epoch 14 (30.7 min timeout). Splits: in_dist=5.10, camber_rc=3.54, camber_cruise=1.46, re_rand=2.30. Peak 42GB. Commit f65f434.
- **Verdict:** kept — first submission on leaderboard. Predictions at `/mnt/new-pvc/predictions/apr23/tanjiro/f65f434/`.
- **Notes:**
  - `val_single_in_dist` is the **worst** split (5.10) despite being the "easy" sanity check — likely because raceCar single has wide y ranges and surf_weight=10 amplifies surface errors.
  - Cosine LR scheduler was set for 50 epochs but training only reached 14 → LR never decayed near zero. Next run should set `epochs` to match the 30min budget so LR annealing finishes.
  - Predict auto-submit OOM'd because train process still held 93GB VRAM; fixed by freeing model + `empty_cache()` before subprocess (commit incoming).
