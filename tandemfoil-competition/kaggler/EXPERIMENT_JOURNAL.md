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

### 2026-04-27 — v1: thorfinn-clone baseline
- **Hypothesis:** Thorfinn's recipe (L1 + p_weight=3 + surf_weight=10 + sub40k + bs=4 + bf16 + 40ep) is the gold standard. Cloning it should land me at val avg_surf_p ≈ 50-60.
- **Change:** Wrote train.py from scratch with thorfinn-style config (192/L6/h6/mlp2/slice64); split model into model.py for predict.py importability; added auto-submit + best-checkpoint mirror to PVC.
- **Result:** Best val avg_surf_p=55.07 at epoch 39; train completed 40 epochs in 30.1 min (10.5 GB peak). Per-split: single_in_dist=52.73, geom_rc=72.73, geom_cruise=38.06, re_rand=56.77.
- **Verdict:** Kept — solid baseline, ~12 points above thorfinn's val (52.08) most likely just seed/sampling variance.
- **Notes:** geom_camber_rc is the worst split (~73, while others are 38-57). Cosine LR fully decayed by e39, last 5 epochs all near-tie. Test scoring pending. Run `it01dzgl`. Next: chain warm-start at lower LR (thorfinn went 56→52 with this).
