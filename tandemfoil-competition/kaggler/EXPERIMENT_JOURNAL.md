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

### 2026-04-27 — iter1: Transolver-192/6 + SmoothL1 surf loss (aborted)
- **Hypothesis:** Restoring the proven thorfinn-apr27 architecture (n_hidden=192, n_layers=6, n_head=6, slice=64) plus SmoothL1 (Huber) on surface (better aligned with leaderboard MAE) and surf_weight=15 would beat the prior best of 42.90 avg_surf_p.
- **Change:** train.py — switched to Huber loss for surface component, surf_weight 10→15, added grad clip 1.0, ckpt selected by avg_surf_p, added `__main__` guard. predict.py — load Transolver via import + read config.yaml from checkpoint dir.
- **Result:** Killed at epoch 3/50 after ~12 min. Trajectory: ep1=206, ep2=189, ep3=173 avg_surf_p. ~243s/epoch ⇒ only ~7 epochs in 30 min budget. Cosine T_max=50 means LR barely decays — model trained at near-constant 5e-4 throughout. Far from converging to ~42 baseline. VRAM peak 74GB.
- **Verdict:** Discarded. Cold-start from scratch in 7 epochs cannot match a prior thorfinn checkpoint that already exists at 42.90 avg_surf_p. Wrong starting point; need warm-start.
- **Notes:** Confirmed prior thorfinn checkpoint at `/mnt/new-pvc/kagent/apr27/thorfinn/checkpoints/model-eue2l8uf/checkpoint.pt` loads cleanly into the same architecture (1.71M params). Next: warm-start iter2 from that checkpoint with reduced LR and faster cosine decay.
