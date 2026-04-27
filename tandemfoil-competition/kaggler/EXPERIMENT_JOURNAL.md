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

### 2026-04-27 — iter3: warm-start iter2 ckpt @ lr=1e-4, wd=3e-4

- **Hypothesis:** Iter2 hit val=87.57 then bounced/overfit on geom_camber_rc (129→152). Drop LR by 2× and 3× weight decay to slow overfit while letting other splits keep improving.
- **Change:** `--warm_start checkpoints/best.pt --lr 1e-4 --weight_decay 3e-4 --epochs 25 --warmup_epochs 0`. Else identical to iter2.
- **Result:** 21 epochs, best val avg_mae_surf_p=**86.36 at E10** (only ~1.4% better than iter2). Splits: single=59.7, geom_rc=136.0, geom_cruise=58.8, re_rand=90.9.
- **Verdict:** kept — small but real win. Single_in_dist did improve (70→60) but geom_camber_rc barely moved.
- **Notes:** Plateau is real; ~86 looks like the ceiling for this architecture+config under heavy warm-start. geom_camber_rc dominates avg now (136 vs ~60 elsewhere). Tried inspecting data for safe mirror aug — z is always positive (floor at z≈0), so vertical flip would create OOD data. x-flip changes foil orientation (not symmetric). Need a real architectural change next.

### 2026-04-27 — iter2: warm-start iter1 ckpt @ lr=2e-4, 25 epochs (cosine to 0)

- **Hypothesis:** Iter1 cosine schedule was set for 200 epochs but only ran 21 → LR barely decayed and val bounced 103–128. Warm-starting and giving the schedule full decay over a 25-epoch budget should let the model fine-tune.
- **Change:** Added `--warm_start <path>` arg; loaded `checkpoints/best.pt` (15.8 MB, 256/L8/sl96/h8). lr=2e-4 (was 5e-4), warmup_epochs=0, epochs=25 so cosine fully decays in budget.
- **Result:** 21 epochs in 30 min. Best val avg_mae_surf_p=**87.57 at E9** (improved 16% from 103.14). Splits: single=70.1, geom_rc=129.4, geom_cruise=60.8, re_rand=90.0. After E9, geom_camber_rc kept worsening (129→152) while easy splits kept improving — clear OOD overfitting.
- **Verdict:** kept — biggest single-iter improvement so far.
- **Notes:** geom_camber_rc (M=6–8 unseen camber) is now ~2× worse than the other splits and drives the average. Other agents likely beat me here via mirror augmentation or larger-scale training. Next iter: another warm-start at even lower lr (1e-4), or add architectural changes (Fourier features, FiLM on Re).

### 2026-04-27 — iter1: leader-class Transolver + bf16 + smooth_L1 + sub40k

- **Hypothesis:** Default Transolver was 128/L5/sl64 → 112.94 surf_p (rank 7/7). Combine leader sizing (256/L8/sl96/h8 from frieren/thorfinn) with alphonse's speed/loss tricks (bf16, smooth_L1+p_weight=3, train_subsample=40000, warmup, grad_clip=1.0). Get many more epochs into the 30-min budget.
- **Change:** Refactored model into `model.py`. New `--train_subsample` collate keeps all surface nodes + uniform vol subsample. Added bf16 autocast, smooth_L1(beta=0.1) with `[1,1,3]` channel weights, 3-epoch warmup + cosine, grad clip 1.0, batch_size=4. Switched checkpoint selection from `val/loss` to `avg_mae_surf_p` (the leaderboard metric). Mirror best to `checkpoints/best.pt`. Removed ill-fated no-slip BC enforcement after inspecting actual data — `is_surface` near-wall cells have non-zero Ux/Uy (mean abs 43.66), so forcing them to 0 hurts.
- **Result:** 21 epochs in 30 min (5.4 it/s vs prior 1.7). Best val avg_mae_surf_p=103.14 at E18. Splits: single=114.5, geom_rc=141.6, geom_cruise=62.3, re_rand=94.2. Peak VRAM 20.5GB.
- **Verdict:** kept — first iteration that beats prior commit (112.94 → ~103 val). Still rank ~5–6 vs leader (42.11). geom_camber_rc is the hardest split.
- **Notes:** Variance in last few epochs (E15–E20: 113, 128, 107, 103, 121, 115, 121) suggests model is bouncing around — LR likely a touch too high near saturation. Next iter: warm-start from this checkpoint with lower LR + full cosine to see if it fine-tunes further. After that try Fourier features and Re-FiLM. The "no-slip BC" claim in README disagrees with the data — likely `is_surface` marks adjacent boundary cells, not strict wall vertices. Don't waste time on that constraint.
