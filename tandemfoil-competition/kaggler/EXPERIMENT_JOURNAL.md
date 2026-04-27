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

### 2026-04-27 — iter1: Transolver 192x6 + bf16 + L1 + 30k sub + pw3
- **Hypothesis:** A proven recipe from frieren's apr23 W&B runs — Transolver 192h x 6L slice_num=64, bf16 autocast, L1 loss, 30k node subsample (preserving all surface), pressure-channel weight 3x, sw=10, AdamW lr=5e-4 with 3-epoch warmup + cosine over 60 ep — should land near val/loss ~2 in one shot from scratch.
- **Change:** train.py rewritten with the full recipe (model unchanged from baseline shape, but with subsample dataset wrapper, bf16 autocast, weighted-channel L1, lambda LR scheduler, gradient clip 1.0). predict.py rewritten to load Transolver from the saved checkpoint with config.yaml. Refactored Transolver into model.py to avoid train.py CLI parsing during predict import.
- **Result:** 48 epochs in 28.2 min (timeout). Best epoch 31, val/loss=3.82, **avg_surf_p=91.88**. Per-split surf_p (best epoch): cruise=58.4, single_in_dist≈90, rc≈110, re_rand≈110. W&B run `thorfinn/iter1-192x6-bf16-sub30k-l1-pw3` (id 0ndqgt66).
- **Verdict:** Kept — first usable baseline. Predictions submitted to apr27-5/thorfinn/f745892. Far from frieren's apr27 score of 42.11 but a credible starting point for chained finetuning.
- **Notes:** First auto-predict failed because predict.py imported Transolver from train.py and triggered simple_parsing on the wrong argv; fixed by extracting the model into model.py. val/loss bounced epoch-to-epoch (likely surf-loss noise dominating). Next iter: chain finetune at bs=2, full mesh, low LR (1e-5 → 5e-6) following frieren's chain pattern.
