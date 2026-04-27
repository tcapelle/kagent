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

### 2026-04-27 — v2: chain warm-start from v1 (lr=1e-4, 30ep)
- **Hypothesis:** Chain warm-start at lower LR refines v1's solution by exploring nearby minima with slow cosine decay (thorfinn went 56→52 doing this).
- **Change:** Added `--warm_start` arg pointing to v1 checkpoint; lr=1e-4 (5x v1's final), epochs=30, no warmup, plain cosine decay.
- **Result:** Best val avg_surf_p=47.67 at epoch 29 (vs v1=55.07 → 13% improvement). 23.2 min total. Per-split: single_in_dist=43.16, geom_rc=64.42, geom_cruise=32.40, re_rand=50.68.
- **Verdict:** Kept — beats thorfinn's val (52.08) by 4.4 points.
- **Notes:** Disruptive at first (e3=56.11) then cosine decay rescued and pushed below v1 by epoch 10. geom_rc still the weak split (64). Run `0zkmlvv9`. Next: try ensemble v1+v2 OR train fresh diverse model for ensemble.

### 2026-04-27 — v1: thorfinn-clone baseline
- **Hypothesis:** Thorfinn's recipe (L1 + p_weight=3 + surf_weight=10 + sub40k + bs=4 + bf16 + 40ep) is the gold standard. Cloning it should land me at val avg_surf_p ≈ 50-60.
- **Change:** Wrote train.py from scratch with thorfinn-style config (192/L6/h6/mlp2/slice64); split model into model.py for predict.py importability; added auto-submit + best-checkpoint mirror to PVC.
- **Result:** Best val avg_surf_p=55.07 at epoch 39; train completed 40 epochs in 30.1 min (10.5 GB peak). Per-split: single_in_dist=52.73, geom_rc=72.73, geom_cruise=38.06, re_rand=56.77.
- **Verdict:** Kept — solid baseline, ~12 points above thorfinn's val (52.08) most likely just seed/sampling variance.
- **Notes:** geom_camber_rc is the worst split (~73, while others are 38-57). Cosine LR fully decayed by e39, last 5 epochs all near-tie. Test scoring pending. Run `it01dzgl`. Next: chain warm-start at lower LR (thorfinn went 56→52 with this).
