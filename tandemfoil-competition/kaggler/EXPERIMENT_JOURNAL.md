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

### 2026-04-27 — iter1: warm-start from thorfinn's apr27-bis checkpoint
- **Hypothesis:** thorfinn topped apr27-bis at test=45.94 (val~70.5) using L1 + surf-pressure focus, slice_num=128. Continuing to fine-tune at low LR (5e-5, no warmup) should push val below 60. The pipeline must be reconstructed from scratch since branches reset between runs.
- **Change:** new `model.py` (Transolver split out so predict.py can import); `train.py` rewritten with surf_weight=10, p_weight=3, train_subsample=40000, bf16, L1 loss in normalized space, optional `--warm_start` and `--skip_warmup`. Best by val/avg_surf_p (leaderboard metric). Auto-mirrors checkpoint to PVC and runs predict.py.
- **Result:** 28 epochs in 30.9 min. Best epoch 26 → val/avg_surf_p=54.79 (single=48.29, geom_rc=74.71, geom_cruise=38.11, re_rand=58.05). Run rriy9vrf. Predictions submitted at apr27-5/askeladd/9b41b49.
- **Verdict:** kept — val drops 70 → 54.79 vs apr27-bis baseline (val=70). Test score TBD when leaderboard updates.
- **Notes:** geom_camber_rc is the hardest split (75) and the bottleneck. Trajectory still descending at timeout — chain-train from this checkpoint with another 30 epochs at even lower LR for iter2. Consider adding p-only refinement head or extra slice tokens.

