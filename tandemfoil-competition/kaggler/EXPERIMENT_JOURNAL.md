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

### 2026-04-27 — iter1: frieren-recipe Transolver fresh from scratch
- **Hypothesis:** Replicate frieren's apr23 winning recipe — 192×6 Transolver, slice=64, mlp_ratio=2, L1 loss, p_weight=3, surf_weight=10, bs=2 (no volume subsampling), bf16, AdamW lr=5e-4 with warmup+cosine. Skips frieren's earlier subsample-based pretraining and goes straight to bs=2/no-subsample.
- **Change:** Wrote `train.py` (`a6bb09c`), then refactored Transolver into `model.py` (`3b4188e`) because `predict.py` was triggering train.py's `simple_parsing.parse(Config)` at import time.
- **Result:** 12 epochs, ~2.5 min/epoch, peak 29.1 GB VRAM, total 29.9 min. **Best val/loss = 2.5526** at epoch 12 (last epoch — still improving). Train e12 vol=0.52 surf=0.43. Per-split val: single=4.54, rc=3.01, cruise=0.82, re_rand=1.83. Run [9ykedq8l](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/9ykedq8l). Predictions saved to `apr27-5/tanjiro/a6bb09c/`.
- **Verdict:** kept as warm-start seed for iter2. Val/loss is well above frieren's apr23 single-run iter15 (1.87) — likely because we used 12 epochs vs their 35. The model is still descending at ep12 (cosine ended at near-0 LR).
- **Notes:** Two issues to fix in iter2: (1) val/loss of 2.55 is much worse than expected; (2) cosine reached zero LR with the curve still declining. Strategy for iter2: warm-start from iter1 with a fresh cosine at lr=2e-4, 12 epochs. Should drop val/loss substantially. Then iter3 = warm-start at lower LR for further chain. Predictor refactor: model now lives in `model.py` so train and predict can both `from model import Transolver` without side effects.
