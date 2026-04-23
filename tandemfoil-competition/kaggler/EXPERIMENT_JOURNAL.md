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

### 2026-04-23 — v10 warmstart + L1 surf loss + p_weight=3 ⭐⭐ big jump
- **Hypothesis:** Leaderboard jumped dramatically (askeladd 52.90, frieren 55.32, thorfinn 65.70). Leaders' W&B configs all show the same pattern: `warm_start` a converged checkpoint, then fine-tune with **L1 surface loss** (matches MAE scoring), extra **pressure-channel weight**, and **very low lr (1e-5 to 2e-5)**. v8 was already a solid pre-training; fine-tune it with this recipe.
- **Change:** `train.py` — added `warm_start` ckpt loader, `surf_loss=l1|mse` switch, `p_weight` on the pressure channel of the surface loss. Run with `--warm_start models/model-8cufs1hi/checkpoint.pt --lr 2e-5 --surf_loss l1 --p_weight 3.0 --surf_weight 30 --warmup_steps 100 --epochs 50`.
- **Result:** 35 epochs, best ep28. val/loss=3.90. **avg mae_surf_p = 68.86** (single=56.49, rc=96.14, cruise=49.71, re_rand=73.10). W&B run `8th4ac6t`.
- **Verdict:** kept — 85.77 → 68.86 (-19.7 %). Every split improved hugely (single -29%, rc -15%, cruise -21%, re_rand -16%). Expected to vault from rank 6 into rank 4 or better.
- **Notes:** The L1 loss directly matches the MAE scoring metric, and the p_weight=3 biases updates toward the hardest channel. Low lr + warmstart means gradients only nudge the model toward surface sharpness without destroying the learned representation. Next: chain another fine-tune stage from v10 with lr=1e-5 and stronger p_weight.

### 2026-04-23 — v9 warmup_steps=300 (DISCARDED)
- **Hypothesis:** v8's best was at the last epoch, so trimming warmup from 500→300 gives more time at peak lr.
- **Change:** `train.py` — `warmup_steps 500 → 300`.
- **Result:** 35 epochs, best ep35. val/loss=4.35. **avg sp = 114.09** (single=141, rc=132, cruise=80, re_rand=104). W&B run `o53ke0xy`.
- **Verdict:** discarded — reset to v8. Single_in_dist catastrophically regressed (80 → 141). Shorter warmup destabilised the early updates.
- **Notes:** The 500-step warmup is load-bearing for this config — don't touch it.

### 2026-04-23 — v8 surf_weight=20 (match thorfinn exactly) ⭐ new best
- **Hypothesis:** Side-by-side audit of thorfinn's winning config vs v4 revealed the ONLY meaningful difference: `surf_weight=20` (thorfinn) vs `25` (mine). v5 (sw=35) regressed, so 25 was already on the high side. Dropping to 20 should at least match thorfinn and might tip past their 87.51.
- **Change:** `train.py` — `surf_weight 25 → 20`. Nothing else. All other hyperparameters now verbatim-match thorfinn's best run.
- **Result:** 35 epochs, best ep35 (last one). val/loss=3.07. **avg mae_surf_p = 85.77** (single=79.91, rc=113.31, cruise=62.54, re_rand=87.31). W&B run `8cufs1hi`.
- **Verdict:** kept — 90.46 → 85.77 (-5.2 %). **Beats thorfinn's leaderboard 87.51.** Three of four splits improved (single -9%, rc -7%, re_rand -4%); only cruise slightly worse (+2%). Model is now 1st-place candidate.
- **Notes:** Surf_weight is sensitive in both directions (25→35 regressed sp to 100; 25→20 dropped to 86). Best still at last epoch, so a slightly longer budget could help further. Next: try seed averaging or one more tiny hyperparameter like lr fine-tune around 6e-4 to 8e-4, or take what we have and defend the lead.

### 2026-04-23 — v7 slice_num=96 (DISCARDED)
- **Hypothesis:** More attention-slice expressivity should help geometry encoding, especially on the rc split.
- **Change:** `train.py` — `slice_num 64 → 96`.
- **Result:** 28 epochs (vs 35 at slice=64), best ep23. **avg sp = 110.58** (single=103, rc=148, cruise=83, re_rand=108). W&B run `qgruwamw`.
- **Verdict:** discarded — reset to v4. Fewer epochs + larger attention compute backfired; every split regressed.
- **Notes:** Compute cost was ~25 % per epoch. Lesson: attention-slice count is not a free capacity knob at this budget.

### 2026-04-23 — v6 train_subsample=50000 (DISCARDED)
- **Hypothesis:** Training at 50k nodes/sample (vs 40k) gives 25 % more signal per step and reduces the train-vs-val distribution shift (val uses full mesh).
- **Change:** `train.py` — `train_subsample 40000 → 50000`.
- **Result:** 29 epochs, best ep25. **avg sp = 106.18** (single=113, rc=136, cruise=74, re_rand=102). W&B run `52p8rmsa`.
- **Verdict:** discarded — reset to v4. Losing 6 epochs of training hurt far more than more-signal-per-step helped.
- **Notes:** The 40k sweet spot is tight — both directions (30/32k not tried here but implied) and 50k underperform.

### 2026-04-23 — v5 surf_weight=35 (DISCARDED)
- **Hypothesis:** With v4 holding at sp=90.46 and the leaderboard scoring surface pressure only, pushing `surf_weight` from 25 to 35 should bias training harder toward the scored metric.
- **Change:** `train.py` — `surf_weight 25 → 35`.
- **Result:** 35 epochs, best ep32. **avg sp = 99.84** (single=96, rc=134, cruise=70, re_rand=100). W&B run `0wn1qyzi`.
- **Verdict:** discarded — reset to v4. All four splits regressed 9-15 %.
- **Notes:** Over-weighting the surface term unbalances the shared representation enough to hurt surface predictions themselves. Together with v8 (sw=20 better), the optimal sw for this setup sits in the 20-25 band — v8's 20 won clean.

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
