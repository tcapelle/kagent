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

### 2026-04-27 — Iter 4 — second warm-start, lr=5e-5, surf_w=20
- **Hypothesis:** Iter 3 plateaued around `avg_surf_p≈100` with `val/loss` bouncing between 3.20 and 4.85; cosine LR from 2e-4 still seems to overshoot. Drop initial LR to 5e-5 and bump `surf_weight` from 15→20 to push the surface objective harder.
- **Change:** invocation only — `--resume model-aglmomxf/checkpoint.pt --lr 5e-5 --surf_weight 20`. No code changes. Wandb run `2v94v7an`.
- **Result:** Best epoch 9 of 11 finished. `val/loss=3.094`, avg surf_p MAE = **86.4** (vs iter 3 `99.7`), surf_Ux = 1.27. Train 32.6 min, 58 GB peak.
- **Verdict:** Kept — modest but real improvement, and the val curve was much steadier (3.4–4.0 range) than iter 3's, confirming LR was the overshoot culprit.
- **Notes:** Test scoring confirms the val→test correspondence is now sane (iter 3: val 99.7 → test 89.23, vs iter 1's broken val 119.7 → test 350.91). Whatever was specifically wrong with iter 1's predictions is gone — likely the L1 surface loss let one or two extreme high-Re/high-AoA samples ride free with very wild predictions, which iter 3's MSE penalized away.

### 2026-04-27 — Iter 3 — warm-start with MSE+pressure-weight
- **Hypothesis:** Iter 1 used L1 on surface; sharp pressure peaks need quadratic gradient (MSE) and explicit pressure weight to better match the leaderboard's avg surf_p MAE. Resume from iter 1's checkpoint with `lr=2e-4` to fine-tune.
- **Change:** `train.py` reverted model to T-192-6-6 (no subsample), surface loss back to MSE, added per-channel weight `[1,1,p_weight=2]` inside the squared error, raised `surf_weight=15`, plumbed `--resume <path>`. Wandb run `aglmomxf`.
- **Result:** Best epoch 7 of 11 finished. `val/loss=3.20` (matching iter 1) but the MAE-aligned axis is much better: avg surf_p MAE = **99.7** (vs iter 1 `119.7`), surf_Ux = 1.26 (vs `1.78`). Train 32.4 min, 58 GB peak.
- **Verdict:** Kept — strict improvement on the leaderboard axis. Per-channel pressure weighting + MSE on surface clearly beats uniform L1 surf, even with the same model and same total weight on surface.
- **Notes:** Iter 2 was killed mid-run: bigger T-224-7-8 + 50k train subsample + p_weight=4 converged ~2× slower (avg surf_p=195 at epoch 5 vs iter-1 epoch 5's 132); subsample appears to drop too much volume signal for the slice-based attention. Open puzzles: scorer reports `avg_surf_p=350.91` for iter 1, but my own per-node MAE on the same prediction file is `134.79` — fern's predictions match scorer exactly under the same code, so my predictions are getting scored differently for an unknown reason.

### 2026-04-27 — Transolver-192-6-6, bf16 AMP, L1 surf
- **Hypothesis:** Match the apr27 leader's smaller config (n_hidden=192, n_layers=6, n_head=6, slice_num=64). Use L1 on surface to better align with the MAE leaderboard metric, and bf16 autocast to fit a bigger model in 30 min.
- **Change:** `train.py` upsized model + bf16 forward+loss + L1 surf loss. `predict.py` loads `Transolver` from `train.py` and reads `config.yaml` next to the checkpoint.
- **Result:** Best epoch 9 (of 11 finished), `val/loss=3.187`. avg surf_p MAE = **119.7** (s=146.5, rc=134.5, cr=92.7, re=105.0). Trained 32 min, 58 GB peak. wandb run `sir5s034`.
- **Verdict:** Kept as starting point — first checkpoint to land on the apr27-5 leaderboard, but ~3× behind frieren's apr27 score (42.1). Surface pressure is the bottleneck.
- **Notes:** L1 in normalized space underweights pressure (since y_std differs across channels and pressure has the largest physical range). Next: switch surf loss back to MSE (sharper gradient on peaks), add per-channel pressure weight, raise surf_weight, possibly bigger model. The per-split spread (cr=93 vs s=147) tracks pressure-variance differences across domains, not generalization gap.
