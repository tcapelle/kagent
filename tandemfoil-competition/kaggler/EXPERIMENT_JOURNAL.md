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

### 2026-04-27 — iter1 warm-start fine-tune (surf-p heavy L1)
- **Hypothesis:** Warm-start from prior thorfinn checkpoint (`model-e3itadc2`, leaderboard test 45.94) and fine-tune with a loss that weights surface pressure ~6× and surface velocity 1×, plus a small volume term, since the leaderboard ranks only by avg surface pressure MAE.
- **Change:** Rebuilt `model.py` (Transolver 192/6/6/slice=128/mlp_ratio=2), `train.py` (per-channel L1 in physical units divided by y_std, surf_p_weight=6, surf_uv_weight=1, vol weights 0.5; bf16 autocast; subsample 40k volume nodes/sample; warmup_frac=0; cosine over 50; lr=5e-5; grad_clip=1.0); `predict.py` reads `config.yaml` next to checkpoint.
- **Result:** val avg_surf_p best 54.81 at epoch 26 (per-split: single_in_dist=48.54, geom_rc=76.05, geom_cruise=37.39, re_rand=57.27). Warm-start val baseline (computed via `eval_ensemble.py`): 70.52. So ~22% relative drop on val. Run id `model-jbbynlph`. 29.4 min, 15.2 GB peak.
- **Verdict:** kept — clear val improvement and predictions auto-submitted to commit `c329256` (will be visible on leaderboard once scorer picks them up).
- **Notes:** Initial concern when epoch 1 hit 65.7 was misplaced — that's still better than warm-start's val of 70.5 (the leaderboard value 45.94 is on TEST, not val). Implication: my ratio of val→test for this checkpoint should put test in the 35–40 range, ahead of the prior 45.94.

