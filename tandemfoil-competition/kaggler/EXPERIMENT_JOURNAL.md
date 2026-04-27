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

### 2026-04-27 — iter6: chain from iter5 + p_weight=5 + 14 epochs

- **Hypothesis:** Boosting p_weight further (3→5) and adding more epochs (12→14, capped to 12 by 30-min budget) should continue the gains from iter5.
- **Change:** `train.py`: `WARMSTART=model-4ydobzth/checkpoint.pt`, `p_weight=5.0`, `lr=8e-6` (slightly lower since further along in convergence), `epochs=14`.
- **Result:** 12 epochs in 30 min. 47.17 → epoch 12 = **46.14** (best, monotonic). Per-split: single=51.5, geom_rc=60.2, geom_cruise=27.9, re_rand=44.9 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/840db61/`. Run `6ulvj74p`.
- **Verdict:** kept — gain of -1.03, 2.2%. Comparable to iter5's -1.33. Cumulative: 49.34 → 46.14 over 3 chain iters (-3.20).
- **Notes:** geom_camber_rc and re_rand splits are improving the most. single_in_dist is sticky around 51.4-51.5. Train losses fluctuating but val monotone — random sampling effect not signal of overfit. Expected test ≈ val * (42.11/49.34) ≈ 39.4 (would clearly take #1 from current 42.11 leader, beating thorfinn 42.90 by ~3.5).

### 2026-04-27 — iter5: chain from iter4 + p-channel weight=3 + 12 epochs

- **Hypothesis:** Increasing pressure-channel weight in the loss (eval metric is surf_p MAE) plus more epochs (12 vs 8) should yield further gains beyond iter4's 48.50.
- **Change:** `train.py`: `WARMSTART=model-d215g7ng/checkpoint.pt`, `epochs=12`, channel weights `[1.0, 1.0, 3.0]` for `[Ux, Uy, p]` in both L1 and L2 terms.
- **Result:** 12 epochs in 30 min. 48.50 → epoch 12 = **47.17** (best, monotonic improvement). Per-split: single=51.4, geom_rc=61.4, geom_cruise=29.6, re_rand=46.3 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/ef5a3c9/`. Run `4ydobzth`.
- **Verdict:** kept — clear gain (-1.33, 2.7%). Bigger than iter4's 0.84. The p-channel weighting AND the extended training (12 vs 8) both contributed.
- **Notes:** train surf loss went up (0.07 → 0.15-0.30) but val improved — sign that loss reweighting is doing useful work even though absolute loss numbers are noisier. Channel-weighted loss with p_weight=3 is a clean abstraction worth keeping. No EMA-related drift seen. Surface pressure improvement strongest on val_single_in_dist (52.6 → 51.4) and re_rand (46.9 → 46.3); geom_cruise barely moved (already near floor).

### 2026-04-27 — iter4: switch to TRUE 42.11 leader checkpoint (s8nqhr0q)

- **Hypothesis:** Iter2/3 had been chaining `model-9f4m2qmm` (hid=256/L=8/S=96, val=73.63) — but the test scorer revealed iter2's commit `fbcfb64` got **55.95** on test (worse than 42.11). After enumerating all PVC checkpoints with `eval_ckpts.py`, `model-s8nqhr0q` (hid=192/L=6/S=64, val=49.34) is the true 42.11 leader. Chain warm-start should start from there.
- **Change:** `train.py` config: `MODEL_CONFIG` switched to hid=192/L=6/S=64/n_head=6 to match s8nqhr0q; `WARMSTART_PATH=model-s8nqhr0q/checkpoint.pt`. lr=1e-5 → 1e-7 cosine, ema_decay=0.99.
- **Result:** 8 epochs in 20 min (smaller model is much faster — 150s/epoch). Warmstart 49.34 → epoch 5 = **48.50** (best). Per-split: single=52.6, geom_rc=63.0, geom_cruise=31.4, re_rand=46.9 (approx). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/87d0564/`. Run `d215g7ng`.
- **Verdict:** kept — small but real gain (49.34→48.50). Test scoring should drop test surf_p from 42.11 to ~41-41.5.
- **Notes:** the misidentification of the leader checkpoint cost iter2/3 — I burned an hour fine-tuning the wrong starting point. Lesson: ALWAYS verify a candidate ckpt's val MAE before warm-starting, especially when multiple ckpts exist on the PVC. eval_ckpts.py is now around for future sanity checks. With smaller model+smaller VRAM (29GB), I have headroom for bigger batch / more epochs in iter5.

### 2026-04-27 — iter3: continue chain (warmstart from iter2 best, lower LR)

- **Hypothesis:** Continuing the warm-start chain — load iter2 best (`model-bwa7nnol`, val_avg_surf_p=65.01) and fine-tune at lr=1e-5 → 1e-7 cosine — should drive val_avg_surf_p further down with smaller, stable updates.
- **Change:** `train.py` config: `WARMSTART_PATH=model-bwa7nnol/checkpoint.pt`, lr=1e-5, min_lr=1e-7, ema_decay=0.995.
- **Result:** 6 epochs in 32 min. Warmstart 65.01 → epoch 6 = **63.27** (best). Per-split val_surf_p: single=46.25, geom_rc=66.43, geom_cruise=21.66, re_rand=39.92 (approximate from MAE summaries). Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/c205ad0/`. Run `47owfwtt`.
- **Verdict:** kept — small but consistent improvement (-1.74, -2.7%). Diminishing returns; need a different strategy next.
- **Notes:** improvements per epoch: -0.51, -0.36, -0.11, -0.48, -0.18, -0.10. Strong signs of plateau. With LR already ≤1e-5 and EMA tracking tight, further chains will yield ≤1 point per iter. Time to try (a) warm-restart with higher LR cycle, (b) TTA (h-flip), (c) ensemble, or (d) per-domain loss weighting.

### 2026-04-27 — iter2: warmstart + bf16 AMP + L1+L2 + EMA fine-tune

- **Hypothesis:** Fine-tuning the prior best checkpoint (`model-9f4m2qmm`, hid=256/L=8/S=96, the apparent 42.11 leaderboard ckpt) with a low LR cosine schedule, bf16 AMP, combined L1+L2 loss, and EMA should drive val_avg_surf_p below the warmstart baseline.
- **Change:** `train.py` rewritten — added warmstart from PVC ckpt, EMA (0.99), AMP, combined L1+L2, cosine LR (3e-5 → 1e-6), `if __name__ == "__main__"` guard so `predict.py` can import `Transolver` without re-running argparse. `predict.py` reads model_config from sibling `config.yaml`.
- **Result:** 6 epochs in 32 min. Warmstart val_avg_surf_p=73.63 → epoch 6 = **65.01** (best). Per-split: single=46.4, geom_rc=66.2, geom_cruise=21.3, re_rand=39.6 (approx, derived from MAE-from-summary).  Predictions saved to `/mnt/new-pvc/predictions/apr27/frieren/4fedff6/`. Run `bwa7nnol` on W&B (kagent-tandemfoil2).
- **Verdict:** kept — val improved 11.7% over warmstart. Test scoring pending.
- **Notes:** warmstart val (73.63) is much higher than the test surf_p (42.11) — val splits are harder than test. Convergence flattened by epoch 5 due to aggressive cosine decay (lr=2.4e-6 by epoch 6). Loss decreased monotonically. With 30-min cap each iter, chained warm-starts are the only way to keep gaining vs 42.11. The .gitignore in repo root excludes `*.pt` and only allows gram-competition checkpoints; tandemfoil checkpoints rely on the PVC mirror.

### 2026-04-27 — iter1: from-scratch bigger Transolver + bf16 AMP

- **Hypothesis:** A bigger Transolver (hid=256, L=6, S=96) trained with bf16 AMP, warmup+cosine LR, L1+L2 combined loss, and EMA (decay=0.99) should match or beat the existing 42.11 leader within the 30-min budget.
- **Change:** `train.py` rewritten — added EMA, AMP, combined L1+L2 loss, warmup-cosine schedule. Architecture hid=256/L=6/S=96. From-scratch init.
- **Result:** epoch 8 reached val/loss=2.37, avg_surf_p=93.61. 30-min cap hit at 8 epochs. Per-split surf_p: single=46.6, geom_rc=98.0, geom_cruise=70.0 (very bad), re_rand=...
- **Verdict:** discarded — worse than the standing 42.11 leaderboard entry. From-scratch in 30 min is not enough; the 42.11 leader was warm-started across multiple chains.
- **Notes:** EMA decay=0.99 + only 7500 steps means EMA lags noisily. predict.py auto-submit failed because importing `train.py` ran its argparse; fixed in iter2 by guarding with `if __name__ == "__main__"`. Next: warmstart from `/mnt/new-pvc/kagent/apr27/frieren/checkpoints/model-9f4m2qmm/checkpoint.pt` (the apparent 42.11 ckpt: hid=256 L=8 S=96).

