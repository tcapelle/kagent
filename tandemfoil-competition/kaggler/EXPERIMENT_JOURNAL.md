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

### 2026-04-23 — v2-hidden256-l8-amp (baseline submission)
- **Hypothesis:** Scaling the starter Transolver from hidden=128/layers=5 to hidden=256/layers=8 with bf16 AMP should materially reduce `val/l2_error` while still fitting in 96GB VRAM. AMP also gives more training steps in 30 min.
- **Change:** `train.py` — `n_hidden=256`, `n_layers=8`, `n_head=8`, `slice_num=64`, `mlp_ratio=2`; enabled bf16 autocast + gradient clip 1.0; added `val/l2_error` logging (mean sqrt(Ux² + Uy²) over all masked nodes). Filled in `predict.py` to load Transolver from saved checkpoint + config.yaml. Bumped `val/loss` to include the new l2 metric.
- **Result:** 8 epochs in 30 min. Best `val/l2_error = 5.84` at epoch 8, but checkpoint selection was by `val/loss` which picked epoch 7 (val/loss=3.74, l2=6.96). Peak VRAM 50.7 GB. W&B run `alphonse/v2-hidden256-l8-amp` (ce6hu0ux). Predictions submitted at commit `c2d3625`.
- **Verdict:** Kept — first real submission, sets baseline around l2≈6.96 with suboptimal checkpoint selection.
- **Notes:** Key finding — `val/loss` (dominated by surface pressure MSE) disagrees with `val/l2_error` (velocity only). Switching checkpoint selection to l2_error would have grabbed epoch 8 (l2=5.84, ~16% better) for free. Also found the `train.py` auto-submit crashes because `predict.py`'s `from train import Transolver` triggered train's argparse. Fixed by extracting the model into `model.py`. Cosine scheduler used T_max=50 so LR barely decayed across 8 epochs — shorten T_max to effective_epochs (~8) to let LR anneal.
