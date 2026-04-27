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

### 2026-04-27 — iter1: leader-class Transolver + bf16 + smooth_L1 + sub40k

- **Hypothesis:** Default Transolver was 128/L5/sl64 → 112.94 surf_p (rank 7/7). Combine leader sizing (256/L8/sl96/h8 from frieren/thorfinn) with alphonse's speed/loss tricks (bf16, smooth_L1+p_weight=3, train_subsample=40000, warmup, grad_clip=1.0). Get many more epochs into the 30-min budget.
- **Change:** Refactored model into `model.py`. New `--train_subsample` collate keeps all surface nodes + uniform vol subsample. Added bf16 autocast, smooth_L1(beta=0.1) with `[1,1,3]` channel weights, 3-epoch warmup + cosine, grad clip 1.0, batch_size=4. Switched checkpoint selection from `val/loss` to `avg_mae_surf_p` (the leaderboard metric). Mirror best to `checkpoints/best.pt`. Removed ill-fated no-slip BC enforcement after inspecting actual data — `is_surface` near-wall cells have non-zero Ux/Uy (mean abs 43.66), so forcing them to 0 hurts.
- **Result:** 21 epochs in 30 min (5.4 it/s vs prior 1.7). Best val avg_mae_surf_p=103.14 at E18. Splits: single=114.5, geom_rc=141.6, geom_cruise=62.3, re_rand=94.2. Peak VRAM 20.5GB.
- **Verdict:** kept — first iteration that beats prior commit (112.94 → ~103 val). Still rank ~5–6 vs leader (42.11). geom_camber_rc is the hardest split.
- **Notes:** Variance in last few epochs (E15–E20: 113, 128, 107, 103, 121, 115, 121) suggests model is bouncing around — LR likely a touch too high near saturation. Next iter: warm-start from this checkpoint with lower LR + full cosine to see if it fine-tunes further. After that try Fourier features and Re-FiLM. The "no-slip BC" claim in README disagrees with the data — likely `is_surface` marks adjacent boundary cells, not strict wall vertices. Don't waste time on that constraint.
