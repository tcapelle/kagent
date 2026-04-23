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

### 2026-04-23 — v1 baseline Transolver + AMP
- **Hypothesis:** Transolver with pressure-aware attention plus bf16 AMP should fit 6-8 epochs in 30min budget and beat the naive copy-baseline. Scoring metric is `avg/mae_surf_p` (surface pressure MAE across 4 test splits).
- **Change:** refactored `train.py` into a `main()` guard, added `apply_no_slip`, bf16 AMP, checkpoint selection by `avg_mae_surf_p` (was `val/loss`). `predict.py` implemented end-to-end with Transolver loading + config.yaml. `n_hidden=192, n_layers=6, slice_num=128, bs=2, 1.73M params`.
- **Result:** best ep8 `val/loss=3.73, avg_mae_surf_p=136.16` (val split means 186/140/97/120). Peak 42GB VRAM. Ran 29.9min. Commit `7d57563`. W&B: `askeladd/v1-baseline-fixed`.
- **Verdict:** kept — first complete submission. Leaderboard snapshot before submission: thorfinn 87.51, frieren 109.27. Baseline scored "incomplete" in first scoring pass (race condition between predict.py finishing + scorer polling).
- **Notes:**
  - **Critical bug found first:** `is_surface=True` is NOT only the airfoil — it includes inlet/outlet/walls where freestream velocity is non-zero. Initial attempt zero-ed Ux,Uy on all surface nodes and would have catastrophically hurt score; disabled no-slip BC enforcement.
  - Only 7-8 epochs fit in 30min at current size. VRAM is at 42GB / 96GB so we have 2x headroom for bigger model or bigger batch.
  - val_loss fluctuates (ep5=4.43, ep6=4.53) — need longer training or better LR schedule.
  - Next ideas: (v2) scale model to ~5M params and bs=4 with AMP; (v3) pressure-channel loss weighting (`p_weight`); (v4) warm-start + fine-tune ensemble.
