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

### 2026-04-27 — iter2: warm-start from sae8usmw + Huber surf loss (kept)
- **Hypothesis:** Warm-start from the prior best thorfinn checkpoint (`apr27/model-sae8usmw`, 49.78 val avg_surf_p) and fine-tune with lower LR (1e-4), Huber surface loss, T_max=8 cosine schedule. Goal: drive val avg_surf_p below 49.78 within the 30-min budget.
- **Change:** train.py — added `init_checkpoint` cfg field that defaults to sae8usmw, switched surface loss to SmoothL1 (Huber, beta=1), surf_weight=15, lr=1e-4, epochs=8, T_max=8, ckpt selected by avg_surf_p (not combined val/loss), grad clip 1.0, `__main__` guard so predict.py can import. predict.py — load Transolver via `from train import Transolver` and read config.yaml from checkpoint dir.
- **Result:** Best epoch 8 with **val avg_surf_p=48.61** (vs warm-start baseline 49.78 → -1.17). Trajectory: ep1=61.43, ep2=59.24, ep3=56.54, ep4=54.49, ep5=52.47, ep6=50.94, ep7=49.13, ep8=48.61. Total wall time 32.1 min. VRAM peak 74GB. WandB run: m7c1v7qx. Predictions auto-submitted to `/mnt/new-pvc/predictions/apr27-4/thorfinn/269b348/`.
- **Verdict:** Kept. Beat the prior best on val. Test leaderboard score TBD.
- **Notes:**
  - Discovered most of the apr27 thorfinn checkpoints in the PVC are *bad* mid-training snapshots. Built `/tmp/eval_ckpt.py` to evaluate them: `model-sae8usmw` and `model-fxwfocit` are tied at 49.78; `model-eue2l8uf` is 372 (corrupted/early). Always re-evaluate warm-start candidates before trusting them.
  - Switching MSE→Huber on surface caused a regression in epoch 1 (61.43 vs warm-start 49.78); the model needs ~7 epochs to re-converge below the warm-start. If the run had been timeout-killed earlier we'd have shipped a worse model than the one we warm-started from. Lesson: evaluate warm-start at epoch 0 and gate ckpt-saving on beating that.
  - Improvement is monotonic and mostly comes from `val_geom_camber_rc` (2.69→1.66 split loss) and `val_single_in_dist` (3.90→2.66). Cruise barely changes (0.41→0.34).
  - Next: try harder loss alignment (pure L1 or per-channel surf weighting that emphasizes p), and potentially go bigger or use TTA.

### 2026-04-27 — iter1: Transolver-192/6 + SmoothL1 surf loss (aborted)
- **Hypothesis:** Restoring the proven thorfinn-apr27 architecture (n_hidden=192, n_layers=6, n_head=6, slice=64) plus SmoothL1 (Huber) on surface (better aligned with leaderboard MAE) and surf_weight=15 would beat the prior best of 42.90 avg_surf_p.
- **Change:** train.py — switched to Huber loss for surface component, surf_weight 10→15, added grad clip 1.0, ckpt selected by avg_surf_p, added `__main__` guard. predict.py — load Transolver via import + read config.yaml from checkpoint dir.
- **Result:** Killed at epoch 3/50 after ~12 min. Trajectory: ep1=206, ep2=189, ep3=173 avg_surf_p. ~243s/epoch ⇒ only ~7 epochs in 30 min budget. Cosine T_max=50 means LR barely decays — model trained at near-constant 5e-4 throughout. Far from converging to ~42 baseline. VRAM peak 74GB.
- **Verdict:** Discarded. Cold-start from scratch in 7 epochs cannot match a prior thorfinn checkpoint that already exists at 42.90 avg_surf_p. Wrong starting point; need warm-start.
- **Notes:** Confirmed prior thorfinn checkpoint at `/mnt/new-pvc/kagent/apr27/thorfinn/checkpoints/model-eue2l8uf/checkpoint.pt` loads cleanly into the same architecture (1.71M params). Next: warm-start iter2 from that checkpoint with reduced LR and faster cosine decay.
