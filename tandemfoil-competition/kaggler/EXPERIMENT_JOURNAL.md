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

### 2026-04-27 — iter3: warm-start iter2 lr=2e-5 12ep (chain step 2)
- **Hypothesis:** Iter2 used lr=1e-4 with substantial early oscillation. Switching to lr=2e-5 avoids the perturbation phase and lets cosine decay over 12ep yield steady fine improvement. Following frieren's chain pattern (iter17→iter19→iter21 used 1e-4→5e-5→2e-5).
- **Change:** `python train.py --warm_start /tmp/iter2_best.pt --lr 2e-5 --epochs 12`. Placeholder commit `60c4364`. Also removed embedded auto-predict from `train.py` because the subprocess was leaving the parent process holding ~75 GB GPU memory after wandb.finish() and OOM-ing the next training run.
- **Result:** Best epoch 12, val/loss=**1.7301** (down from 1.8591). Per-split val: single=2.75, rc=2.04, cruise=0.59, re_rand=1.55. Train e12 vol=0.44 surf=0.32. Steady improvement: e1=1.85, e4=1.79, e8=1.73, e12=1.73 (cosine end). Predictions saved to `apr27-5/tanjiro/60c4364/`. Run [1cc63utu](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/1cc63utu).
- **Verdict:** kept. Smooth chain step, no oscillation. Train loss is much lower than val loss → some overfitting on single_in_dist (2.75 is the worst split).
- **Notes:** Cleanup pattern: must kill train.py process after predictions land (the subprocess.run+wandb.finish combo hangs). For iter4+, train.py no longer auto-predicts — must manually `python predict.py --checkpoint <path> --agent tanjiro`. Next: iter4 = warm iter3 lr=5e-6 12ep — should drop val/loss to ~1.65.

### 2026-04-27 — iter2: warm-start iter1 lr=1e-4 12ep (chain step 1)
- **Hypothesis:** iter1 val/loss=2.55 is undertrained (cosine ended at near-zero LR with the curve still descending). Frieren's recipe (iter17) showed warm-start at lr=1e-4 + 30 epochs gives big gains. With 30-min budget I get only 12 epochs, but cosine decay over those 12 should still improve val/loss substantially.
- **Change:** `python train.py --warm_start /tmp/iter1_best.pt --lr 1e-4 --epochs 12`. Placeholder commit `1a77b97`.
- **Result:** Best epoch 11, val/loss=**1.8591** (down from 2.55). Per-split val: single=3.40, rc=2.61, cruise=0.69, re_rand=1.61. Train e12 vol=0.45 surf=0.33. Predictions saved to `apr27-5/tanjiro/1a77b97/`. Run [5v9jw5ap](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/5v9jw5ap).
- **Verdict:** kept. ~27% improvement in val/loss confirms chain approach works. Early epochs 1-4 oscillated as expected (lr=1e-4 is high), then cosine decay drove improvement from epoch 5 onwards.
- **Notes:** Still well above frieren's apr23 1.0158, but chain is working. Next: iter3 = warm iter2 lr=2e-5 12ep — should land val/loss ~1.5.

### 2026-04-27 — iter1: frieren-recipe Transolver fresh from scratch
- **Hypothesis:** Replicate frieren's apr23 winning recipe — 192×6 Transolver, slice=64, mlp_ratio=2, L1 loss, p_weight=3, surf_weight=10, bs=2 (no volume subsampling), bf16, AdamW lr=5e-4 with warmup+cosine. Skips frieren's earlier subsample-based pretraining and goes straight to bs=2/no-subsample.
- **Change:** Wrote `train.py` (`a6bb09c`), then refactored Transolver into `model.py` (`3b4188e`) because `predict.py` was triggering train.py's `simple_parsing.parse(Config)` at import time.
- **Result:** 12 epochs, ~2.5 min/epoch, peak 29.1 GB VRAM, total 29.9 min. **Best val/loss = 2.5526** at epoch 12 (last epoch — still improving). Train e12 vol=0.52 surf=0.43. Per-split val: single=4.54, rc=3.01, cruise=0.82, re_rand=1.83. Run [9ykedq8l](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil5/runs/9ykedq8l). Predictions saved to `apr27-5/tanjiro/a6bb09c/`.
- **Verdict:** kept as warm-start seed for iter2. Val/loss is well above frieren's apr23 single-run iter15 (1.87) — likely because we used 12 epochs vs their 35. The model is still descending at ep12 (cosine ended at near-0 LR).
- **Notes:** Two issues to fix in iter2: (1) val/loss of 2.55 is much worse than expected; (2) cosine reached zero LR with the curve still declining. Strategy for iter2: warm-start from iter1 with a fresh cosine at lr=2e-4, 12 epochs. Should drop val/loss substantially. Then iter3 = warm-start at lower LR for further chain. Predictor refactor: model now lives in `model.py` so train and predict can both `from model import Transolver` without side effects.
