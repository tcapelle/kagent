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

### 2026-04-23 — v3 cosine-fit + ckpt-by-surf_p (DISCARDED)
- **Hypothesis:** v2 saved its best at epoch 32 of 35 with lr still ~3e-4 (cosine `epochs=60` only 58% through). Fitting cosine to the real time budget (`epochs=34`) plus selecting the checkpoint by `mean_surf_p` (the actual scored metric, not val/loss which is 80% volume) should push `mean_surf_p` below v2's 97.70.
- **Change:** `train.py` — `epochs 60→34`; added `val/mean_surf_p` log and switched the best-checkpoint criterion from `mean_val_loss` to `mean_surf_p`.
- **Result:** 34 epochs, final lr 0. `best mean_surf_p = 101.35` (single=76.17, rc=155.57, cruise=69.98, re_rand=103.69). Worse than v2's 97.70. W&B run `wujwly22`.
- **Verdict:** discarded — reset to v2 commit `5d93622a`. Geom-rc jumped (134→156) and dominated the average. Aggressive cosine-to-zero with the same peak lr (7e-4) over 34 epochs likely under-trained the middle phase; single+cruise improved but rc regressed hard.
- **Notes:** Run-to-run variance is large on rc split (150±25 across consecutive epochs). Future comparisons need 2+ seeds. Next hypothesis: don't touch the schedule — instead attack the rc weakness directly (heavier aug on camber, or more capacity, or longer training at the same peak lr).

### 2026-04-23 — v2-sub40k-sw25-h8-m4
- **Hypothesis:** Previous v1b (192/L6/H4/mlp2, sw=10, no subsample) fit only 13/50 epochs in the 30-min cap and was still descending. Leader thorfinn uses `train_subsample=40000` (all surface + random volume), `surf_weight=20`, `n_head=8`, `mlp_ratio=4`, lr=7e-4 with warmup+grad_clip. Match that and push `surf_weight=25` since the leaderboard scores surface-pressure only.
- **Change:** `train.py` — added `SubsampleDataset` wrapper keeping all surface nodes + random volume sample to target 40k total; model `n_head=4→8`, `mlp_ratio=2→4`; optimizer `lr=5e-4→7e-4`, `weight_decay=1e-4→1e-5`, 500-step linear warmup + per-step cosine via `LambdaLR`, grad-clip 1.0; `surf_weight=10→25`; epochs 50→60.
- **Result:** 35 epochs in 30 min (best at ep32). val/loss=4.896. **avg mae_surf_p = 97.70** (single=84.83, rc=134.40, cruise=74.44, re_rand=97.11). Peak VRAM 12.5 GB. W&B run `f1gjwn3z`.
- **Verdict:** kept — avg surf_p dropped 147→98 vs v1b, pushing from 4th (134.78) to near 2nd (thorfinn 87.51, frieren 109.27).
- **Notes:** Subsampling was the dominant lever — ~2.8× throughput (145s→52s per epoch). Geom-rc is still the weakest split (134 surf_p) — unseen high-camber foils. Cosine with epochs=60 didn't fully decay in the time budget (best at 3.25e-4 peak·0.47).
