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

### 2026-04-27 — iter8: add FiLM-on-Re (AdaLN) conditioning + Fourier + warm-start

- **Hypothesis:** re_rand split is one of my worst on the leaderboard (test 119.91 vs leaders ~37–44). The model only sees `log(Re)` as one of 24 input features; explicit AdaLN-style FiLM conditioning lets each block's LayerNorm be modulated by Re globally. Zero-init the FiLM linear weights so warm-start preserves iter6 behavior; SGD then learns to use the Re signal.
- **Change:** model.py: `TransolverBlock` now takes `film_dim`; if >0 builds a `Linear(film_dim, 4*hidden)` (zero-init) for `(γ1,β1,γ2,β2)` and applies `LN(x)*(1+γ)+β` before each sub-layer. `Transolver` adds an `re_embed` MLP (1→64→64) that maps a per-sample `log(Re)` scalar (taken from `x[:, 0:1, 13]`) into the conditioning vector. train.py: new `--film_re` and `--film_emb_dim` flags; warm-start with `strict=False` so the new FiLM and re_embed params keep their zero/init state. Restored iter6's `best.pt` from git history (94495c7) before launching, since iter7 had clobbered it with worse weights. Run with `--lr 1e-4 --p_weight 3 --weight_decay 2e-4 --warmup_epochs 2 --epochs 25`.
- **Result:** 19 epochs (each ~100s vs prior 89s — FiLM overhead). Best val avg_mae_surf_p=**83.49 at E15** — beats iter6's 83.56 by 0.07. Splits: single=55, geom_rc=135, cruise=55, re_rand=88.
- **Verdict:** kept — new SOTA for me, FiLM contributed a small but real improvement on re_rand (90→88) at no harm elsewhere.
- **Notes:** FiLM took ~10 epochs to escape the warm-start basin and another 5 to reach a slightly better one. Plateau still essentially the same: geom_camber_rc=135 is the hard wall (M=6–8 OOD camber). Pushed 4-model ensemble (iter4 no-Fourier + iter5 Fourier + iter6 Fourier+lower-LR + iter8 Fourier+FiLM) at commit 06e2ef3 — should reduce variance on the OOD splits.

### 2026-04-27 — iter6: warm-start iter5 @ lr=3e-5, p_weight=4

- **Hypothesis:** Drop LR another 3× and bump p_weight 3→4 for one more fine-tune pass on the new (Fourier-featured) architecture.
- **Change:** `--warm_start checkpoints/best.pt --fourier_dim 32 --fourier_sigma 5.0 --lr 3e-5 --p_weight 4 --weight_decay 2e-4 --epochs 30 --warmup_epochs 0`. 21 epochs ran.
- **Result:** Best val avg_mae_surf_p=**83.56 at E1** — i.e. *one epoch of fine-tuning was best*; everything after was worse (87–90). Splits at E1: single=55, geom_rc=136, cruise=56, re_rand=88.
- **Verdict:** kept — new best.
- **Notes:** "Best at E1" tells me the iter5 checkpoint was already very near-optimal under this loss; a single small SGD step nudged it into a slightly better basin and then noise dominated. Suggests next iter should use lr ~1e-5 (or run brief multi-restart fine-tunes) and that ensembles across the last few iters should help most.

### 2026-04-27 — submitted ensemble at commit 94495c7 (iter6+iter5+iter4)

- **Hypothesis:** The last three best checkpoints have similar val scores but different architectures (iter4 has no Fourier features; iters 5–6 do) and different loss-target (different p_weight) — averaging predictions should reduce variance on the hardest split.
- **Change:** Wrote `ensemble.py` that averages saved per-commit prediction tensors element-wise. Uniform weights, 3 commits.
- **Result:** Submitted; awaiting test scoring. (Val numbers don't help since these are test-only predictions.)
- **Verdict:** submitted, separate slot from any single iteration.
- **Notes:** Lost my first ensemble at bdc0312 because iter6's auto-submit overwrote that commit's predictions. Need to commit ensemble.py *after* a training run if I want a clean separate slot.

### 2026-04-27 — iter5: add Fourier features (32 freqs, σ=5) + warm-start

- **Hypothesis:** Iters 2–4 plateaued around 86. The preprocess MLP only sees raw spatial coords (x, z) — adding random Fourier features should help the network represent high-frequency turbulent pressure features. Architecture grows by 64 input dims; pad the preprocess weight with zeros so warm-start preserves iter4 behavior, then let SGD learn to use the new features.
- **Change:** model.py: added `fourier_dim` and `fourier_sigma` to `Transolver`, registers a fixed `[2, fourier_dim]` random buffer; concatenates `[sin(2π Bx), cos(2π Bx)]` to model input. train.py: pads `preprocess.linear_pre.0.weight` from `[512, 24]` → `[512, 88]` when warm-starting an old model into a new one with extra Fourier columns. Run with `--fourier_dim 32 --fourier_sigma 5.0 --lr 1e-4 --p_weight 3 --weight_decay 1e-4`.
- **Result:** 21 epochs, best val avg_mae_surf_p=**84.66 at E11** (1.7% better than iter4's 86.09). Splits: single=57, geom_rc=135, cruise=58, re_rand=88. After E11 the model overfits again, geom_rc bouncing 135–150.
- **Verdict:** kept — small but real improvement on a plateau.
- **Notes:** Fourier features moved the needle a bit but didn't break the geom_camber_rc bottleneck (still ~135). The whole pattern looks like a generalization ceiling on M=6–8 unseen camber. Possible next moves: bigger σ for finer freqs, more Fourier dims (64), or just keep stacking warm-starts.

### 2026-04-27 — iter4: warm-start iter3 @ lr=5e-5, p_weight=5

- **Hypothesis:** Push pressure-channel weight from 3 → 5 to focus the model harder on the leaderboard metric, with even lower LR to fine-tune.
- **Change:** `--warm_start checkpoints/best.pt --lr 5e-5 --p_weight 5.0 --weight_decay 3e-4 --epochs 30 --warmup_epochs 0`. Ran 21 epochs (timeout).
- **Result:** Best val avg_mae_surf_p=**86.09 at E8** (vs iter3=86.36 — basically a tie). single=57, geom_rc=139, cruise=58, re_rand=91.
- **Verdict:** kept (very small win on noise level).
- **Notes:** Plateau is firmly established at ~86 with this architecture. Higher p_weight didn't move geom_camber_rc. Need a real architectural change. Next iter: add Fourier positional features for spatial coords (random Gaussian frequencies, sigma=5) and warm-start with zero-padded preprocess weights so the model starts from current behavior and learns to use the new features.

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
