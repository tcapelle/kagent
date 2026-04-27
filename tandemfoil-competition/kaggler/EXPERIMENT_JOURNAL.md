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

### 2026-04-27 — iter2 warmstart-cosine9-lr3e4

- **Hypothesis:** Resume from iter1 ckpt with a fully-completed cosine schedule sized to actual training time (epochs=9 ≈ 27 min) and a lower peak LR (3e-4) to fine-tune without destabilising. Iter1's cosine never decayed (sized for 50 epochs but only ran 11), so we never benefited from lr annealing.
- **Change:** Same model + recipe as iter1; resumed `checkpoints/best.pt`. CLI: `--resume checkpoints/best.pt --batch_size 4 --train_max_nodes 80000 --lr 3e-4 --epochs 9 --warmup_steps 50`. EMA reset on resume (cold-starts averaging from current weights).
- **Result:** 9 epochs in 26.6 min. Best `mean_mae_surf_p=82.48` at epoch 9 (live; EMA was close behind). Per-split: single=81.42, geom_rc=104.40, geom_cruise=61.36, re_rand=82.75. Submission `apr27/fern/d4221b0`.
- **Verdict:** kept — 27% improvement (113.26 → 82.48). Still 2× behind leader (frieren=42).
- **Notes:** Completed cosine made the difference — train loss kept dropping until last epoch. `geom_camber_rc` (unseen front foil camber, raceCar) is the bottleneck split (104) while cruise is easy (61). Live beat EMA in late epochs, suggesting EMA was lagging behind a still-improving live model. Idea for iter3: another warm-start chain at lower lr (1e-4) over 9 more epochs to keep dropping.

### 2026-04-27 — iter1 baseline-256h8L-bf16-ema-huber-sub80k

- **Hypothesis:** A bigger Transolver (hidden=256, L=8, slice_num=96, h=8) trained with bf16 AMP, warmup+cosine LR, EMA, Huber loss, and 80k-node training subsampling for speed should reach roughly frieren's ballpark (val mean surf_p ~50-100) in 30 min.
- **Change:** Refactored model into `model.py`. New `train.py` with: bigger Transolver, bf16 AMP autocast, warmup(100)+cosine LR, EMA(decay=0.999), Huber loss (β=1.0), grad_clip=1.0, training subsample to 80k nodes (keeps all surface), TF32, val on full mesh, best-by-mean-mae-surf-p, EMA-vs-live picked per epoch. Auto-submit at end.
- **Result:** 11 epochs × 177s training in 32.5 min wall-clock. Best `val/mean_mae_surf_p=113.26` at epoch 11 (EMA). Per-split surf_p: single=128.96, geom_rc=133.65, geom_cruise=85.94, re_rand=104.49. Submission: `apr27/fern/e34b551`.
- **Verdict:** kept — first valid fern submission this run.
- **Notes:** The cosine schedule was sized for `epochs=50` but we only ran 11, so LR only decayed ~5%. iter2 should size epochs to actual training time so cosine fully anneals. Trajectory was monotonically improving — more epochs would help. Frieren's iter1 (35 epochs) reached 54 surf_p, so our 11-epoch run is reasonably on-pace. VRAM was 40.8GB at bs=4 / 80k nodes — plenty of headroom.
