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

### 2026-04-27 — Transolver-192-6-6, bf16 AMP, L1 surf
- **Hypothesis:** Match the apr27 leader's smaller config (n_hidden=192, n_layers=6, n_head=6, slice_num=64). Use L1 on surface to better align with the MAE leaderboard metric, and bf16 autocast to fit a bigger model in 30 min.
- **Change:** `train.py` upsized model + bf16 forward+loss + L1 surf loss. `predict.py` loads `Transolver` from `train.py` and reads `config.yaml` next to the checkpoint.
- **Result:** Best epoch 9 (of 11 finished), `val/loss=3.187`. avg surf_p MAE = **119.7** (s=146.5, rc=134.5, cr=92.7, re=105.0). Trained 32 min, 58 GB peak. wandb run `sir5s034`.
- **Verdict:** Kept as starting point — first checkpoint to land on the apr27-5 leaderboard, but ~3× behind frieren's apr27 score (42.1). Surface pressure is the bottleneck.
- **Notes:** L1 in normalized space underweights pressure (since y_std differs across channels and pressure has the largest physical range). Next: switch surf loss back to MSE (sharper gradient on peaks), add per-channel pressure weight, raise surf_weight, possibly bigger model. The per-split spread (cr=93 vs s=147) tracks pressure-variance differences across domains, not generalization gap.
