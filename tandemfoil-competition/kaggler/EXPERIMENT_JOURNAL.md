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

### 2026-04-27 — iter1: 192x6 L1 p_w=3 sub40K bs=8 (apr23 baseline port)
- **Hypothesis:** Port the apr23 frieren iter4/iter15 recipe verbatim — Transolver 192x6, slice=64, mlp_ratio=2, n_head=6, L1 loss with surface p up-weighted (p_w=3), bf16, AdamW betas=(0.9,0.95), warmup=3+cosine, sub40K volume nodes at bs=8, 35 epochs. Establishes a strong starting point for the chain ensembles that won apr23.
- **Change:** Created `model.py` (Transolver), rewrote `train.py` (apr23 frieren training loop with `--warm_start` flag, `MAX_TIMEOUT_MIN` env, mirror to PVC + `checkpoints/best.pt`, auto-submit), rewrote `predict.py` to load model from `config.yaml`. Added `ensemble.py` (still uncommitted; queued for later iters).
- **Result:** 35 epochs, 26.9 min, peak 20.8 GB. Best epoch 34: val/avg_surf_p=**81.37** (single=2.53, rc=2.79, cruise=1.08, re_rand=2.03 — split losses, not surf_p MAE). Run `zq0fst5n`. Predictions at commit `7ceb221` (still `incomplete` in scores at journal time).
- **Verdict:** Kept — trajectory is monotonic (314→81) and the cosine tail is still descending at e34, so warm-start chain should keep gaining.
- **Notes:** thorfinn currently #1 at test surf_p=45.94. The apr23 lesson is that bs=8+sub40K converges to a local minimum that bs=2+no_subsample warm-start can blow past (val 1.4 → 1.0 in iter93). That's the iter2 plan.

