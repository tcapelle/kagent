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

### 2026-04-27 — iter12/13: 4-way ensemble + 3rd base
- **Iter12:** 4-way ensemble {iter4 (37a85cf), iter6 (f088509), iter9 (ffcecba), iter11 (f4f626e)} averaged via `ensemble.py`, saved at `c0b78fe`. **Test score 32.47** — new #1, beating iter10's 32.59 and alphonse's 33.03. Per-split: single=35.39, rc=45.23, cruise=19.07, re=30.20.
- **Iter13:** 3rd fresh base. 72 epochs, best epoch 71 with val/avg_mae_surf_p=**62.28** (vs iter3 84.20, iter8 68.65 — each base outperforming the previous, suggesting init/data-shuffle luck). Run id `g52d573w`. **Auto-submit overwrote c0b78fe again**, restored via `ensemble.py`.
- **Notes:** consistent pattern — fresh bases keep getting better. Could be the WeightedRandomSampler giving different sample exposures across runs. Next: finetune iter13 (iter14) → 5-way ensemble (iter15).

### 2026-04-27 — iter10/11: cross-basin ensemble + iter9 deep finetune
- **Hypothesis (iter10):** averaging predictions from {iter4 (chain1 mid), iter6 (chain1 final), iter9 (chain2 finetune)} should beat any single model since iter9 is from a DIFFERENT random init than iter4/6 — errors should partially cancel.
- **Change (iter10):** updated `ensemble.py` to point at `[37a85cf, f088509, ffcecba]`, ran it, predictions saved at `d10ff73`.
- **Result (iter10):** **avg_surf_p = 32.59 → #1 on the leaderboard!** Beat alphonse (33.03), ourselves at iter4 (33.94). Single splits: single=35.17, rc=45.52, cruise=19.18, re=30.48 — uniformly better than any single model.
- **Verdict (iter10):** kept — the cross-basin diversity matters.
- **Iter11:** chained another deep finetune from iter9 (lr=5e-6, p_w=5, surf_w=15, 13 epochs). Best epoch 12: val/avg_mae_surf_p=41.40. **Auto-submit overwrote d10ff73 predictions** — restored immediately by re-running `ensemble.py`. iter11 ckpt mirrored at `model-c2hn3pc3`.
- **Notes:** the "auto-submit-clobbers-prior-commit" issue is recurring: I should always commit before launching training so HEAD advances. Future iters: bake an `--out_subdir` arg into predict.py so different runs save to disjoint paths.

### 2026-04-27 — iter8: fresh base for ensemble diversity (random init)
- **Hypothesis:** finetune chain has plateaued (iter4→6 deltas <1.5 in val). A second base trained from scratch with a different random init should reach a different basin and (after its own finetune) produce decorrelated errors usable in a final ensemble.
- **Change:** rerun iter3's exact recipe (`python train.py --agent frieren --wandb_name "frieren/iter8-base-diverse"`) — no code changes. PyTorch seed defaults to per-process random, so init is naturally different.
- **Result:** 72 epochs, best epoch 53: val/avg_mae_surf_p=**68.65** (iter3 was 84.20 — a *better* base, even before finetuning). Run id `4grkdz0n`. Ckpt mirrored at `/mnt/new-pvc/kagent/apr27-5/frieren/checkpoints/model-4grkdz0n/`. Predictions auto-saved at commit `66b3ad4` (ensemble commit) — note this overwrote the iter7 ensemble predictions, but the scoring already happened before overwrite (alphonse moved to 33.43 around then).
- **Verdict:** kept as a base for iter9's chained finetune. Not best.pt (iter6 still better at 39.54).
- **Notes:** Surprisingly outperformed iter3 base. Suggests iter3 may have been stuck in a worse local min. Consider re-doing the iter4-6 chain on this base — could yield a stronger #1.

### 2026-04-27 — iter7: predictions ensemble (iter4 + iter5 + iter6)
- **Hypothesis:** averaging predictions from three closely-related finetunes might cancel residual noise even if base models are correlated. Cheap (<5 min). Submitted under commit `66b3ad4`.
- **Change:** new `ensemble.py`. Loads each commit's `test_*.pt` lists, averages per-sample tensors, saves under HEAD's commit dir.
- **Result:** unknown — leaderboard hadn't yet picked it up before iter8 launched and **overwrote** the same commit's predictions on PVC. Ensemble experiment effectively lost.
- **Verdict:** kept the script (it's the basis for iter10's bigger multi-base ensemble), but the submission itself is lost. Lesson: **always commit before launching a training run** that would auto-submit.
- **Notes:** for iter10 I'll commit a marker first, then run a final ensemble script combining iter4, iter6, iter9 (cross-base) under a known commit.

### 2026-04-27 — iter5: deeper finetune (lr=5e-6, p_weight=5, surf_weight=15)
- **Hypothesis:** marginal further refinement by stepping lr way down and weighting pressure surface harder. The next stage of apr27 frieren's chain.
- **Change:** `python train.py --warm_start checkpoints/best.pt --batch_size 2 --train_subsample 0 --lr 5e-6 --p_weight 5.0 --surf_weight 15 --loss_type l1 --epochs 15`. Predictions auto-saved at commit `7c0c3c8`. (Note iter4 leaderboard score under `37a85cf` came in at **35.05** test — already #1, beating thorfinn's 44.55.)
- **Result:** 12 epochs, best epoch 12: val/avg_mae_surf_p=40.12 (vs iter4's 40.97 — marginal). Run id `h3y73gp9`. Per-split val_loss numbers can't be compared directly to iter4 because surf_weight/p_weight changed.
- **Verdict:** kept (slightly better val), but improvement is small; lr too low for big moves. Net leaderboard impact unknown until iter5 predictions get scored.
- **Notes:** Diminishing returns from deeper finetuning. Next options: (a) try a *different* optimisation lever — e.g. moderate lr (1e-5) with p_weight=4, surf_weight=12, longer schedule; (b) ensemble iter4 + iter5 predictions; (c) re-run iter3 from scratch with a different subsample size or model size.

### 2026-04-27 — iter4: full-resolution finetune chained from iter3 (p_weight=3)
- **Hypothesis:** subsampled training stops short of solving the surface boundary layer that drives `mae_surf_p`. Warm-start iter3's checkpoint, train at full resolution (subsample=0, batch_size=2) with very low lr (2e-5) and a 3× weight on the pressure channel — this is exactly what apr27 frieren did to drop from ~54 → 42.
- **Change:** `python train.py --warm_start checkpoints/best.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --p_weight 3.0 --surf_weight 10 --loss_type l1 --epochs 20`. No code changes (uses iter3's `--warm_start` plumbing).
- **Result:** **avg_mae_surf_p=40.97 at epoch 12** (run id `kr1xvas8`). val/loss=1.48. 12 epochs at ~150 s each. Per-split val_loss: single=1.38, rc=2.06, cruise=0.90, re=1.60. **Halved iter3's 84.20**, beats apr27 frieren's 42.11. Predictions saved at commit `37a85cf` (journal commit was HEAD when predict.py auto-ran).
- **Verdict:** kept — major win, likely tops the apr27-5 leaderboard (current leader thorfinn 46.10).
- **Notes:** Still room to improve — val_geom_camber_rc is 2.06 (worst split). Next: another finetune chain at even lower lr (5e-6) with higher p_weight (5) to extract last bit of surface refinement.

### 2026-04-27 — iter3: random subsampling (16k pts) + L1 loss + bf16
- **Hypothesis:** apr27 frieren reached 42.11 by **subsampling 16k of ~100k mesh points per training step** (recipe surfaced in thorfinn's notes), giving ~6× more epochs in the 30-min budget. With the same 192×6 transolver and L1 (= eval-metric-aligned) loss this should crush the iter1 score.
- **Change:** `train.py` — added `train_subsample` (random subset per sample, surface points always kept), `loss_type` (l1/mse/smooth_l1), `p_weight` (per-channel pressure boost), `warm_start` (path) options. Switched default loss to L1, surf_weight=10, no p_weight, subsample=16384, base lr=5e-4 + cosine over 80 epochs.
- **Result:** **72 epochs** in 30 min (vs 11 before — 6.5× speedup; epoch ≈ 25 s, VRAM 5.4 GB). Best epoch 54: val/loss=2.78, **avg_mae_surf_p=84.20** (vs iter1 110.4, iter2 132.3). Run id `f41df47d`, code commit `e75607f`.
- **Verdict:** kept — clear winner. New best.pt.
- **Notes:** Plateaued after epoch 54; cosine LR was still moderate. Next: chain a fine-tune at full resolution (subsample=0), batch_size=2, very low lr (e.g. 5e-5), with a higher pressure channel weight to refine surface pressure specifically. apr27 frieren chained 3 such finetunes to reach 42.11.

### 2026-04-27 — iter2: revert to MSE + warmup, no subsample (DISCARDED)
- **Hypothesis:** Smooth-L1 with β=0.1 (iter1) had near-constant gradient → slow convergence. MSE + lr=1e-3 with linear warmup should converge faster in 11 epochs.
- **Change:** train.py — switched to MSE, surf_weight=15, lr=1e-3, 2-epoch linear warmup before cosine.
- **Result:** 11 epochs (still 168 s/epoch, no speed change), best epoch 11: val/loss=4.48, avg_mae_surf_p=132.3 (worse than iter1's 110). Run id `m...`, code commit `f443e04`.
- **Verdict:** discarded (`git reset --hard 71199bc`) — went backwards on the eval metric. Real bottleneck wasn't loss shape but **epochs/budget**: only 11 epochs is too few for this model regardless of loss.

### 2026-04-27 — iter1: smooth-L1 + bf16 AMP + surf_weight=25
- **Hypothesis:** competition metric is `avg/mae_surf_p` (L1). Switching MSE→Smooth L1 (β=0.1) and raising `surf_weight` from 10→25 should align training with the metric. bf16 autocast lets us fit more epochs in 30 min.
- **Change:** `train.py` — Smooth L1 loss, surf_weight=25, bf16 autocast for fwd+loss, grad clip=1.0, n_hidden=192/n_layers=6/n_head=6 (matches apr27 frieren best). `model.py` extracted for clean `predict.py` import (refactor).
- **Result:** epoch 11/80 hit 30-min timeout. val/loss=5.998 (smooth-L1, surf_weight=25 weighted), avg_mae_surf_p=110.4. Per-split val_loss: single=7.73, rc=7.01, cruise=3.88, re=5.38. VRAM 58.1 GB. Run id `w2bmc9bd`, ckpt commit `84eae1a`.
- **Verdict:** kept (no prior baseline on this branch, first checkpoint). But avg_mae_surf_p=110 is far worse than apr27 frieren's 42.11 — likely because Smooth-L1 with β=0.1 has near-constant gradient for typical normalized errors, slowing convergence; we only ran 11 epochs.
- **Notes:** Likely fixes for iter2: drop Smooth L1 in favor of pure MSE (proven to train faster) OR keep Smooth L1 but with β=1.0 (Huber-like with larger quadratic region), increase epochs by speeding up (e.g. smaller batch padding via subsampling, or compile/SDPA). Also consider warmup LR + larger lr to converge faster.
