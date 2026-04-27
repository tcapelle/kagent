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

### 2026-04-27 — v8 chain from frieren's newer h3y73gp9 ckpt (frieren leapt to 33.94)
- **Hypothesis:** v7 took #1 at 33.70 from frieren's 33.94. While the scorer was running, frieren
  saved a newer ckpt `h3y73gp9` (presumably their 33.94 entry). Warm-starting from their newer
  state and chain-training another 30 min should compound the lead. Slightly lower LR (3e-6 vs 5e-6)
  for finer polishing.
- **Change:** no code changes. Run: `--warm_start h3y73gp9 --train_subsample 0 --batch_size 2
  --lr 3e-6 --p_weight 5 --surf_weight 10 --epochs 12`.
- **Result:** wandb run, 12 epochs in 29.8 min. Best at epoch 11 →
  **val/avg_surf_p = 39.81** (single 40.02 / geom_rc 53.26 / cruise 24.94 / re_rand 41.04).
  Frieren's h3y73gp9 evaluated to val 40.30 in my pipeline, so training improved on top of it.
  Predictions at `/mnt/new-pvc/predictions/apr27-5/alphonse/64f05cd/`. Expected test ~33.5.
- **Verdict:** kept — small but consistent gain on every split vs v7 (40.10 → 39.81).
- **Notes:** trajectory is plateauing; per-epoch gains are tenths now. Next move is probably
  weight-averaging v7+v8 (same arch, same lineage) or another chain at lr=1e-6 to fully exploit
  the cosine schedule's tail.

### 2026-04-27 — v7 pivot: warm-start from frieren's kr1xvas8 (frieren leapt to test 35.05)
- **Hypothesis:** v6's chain-train from v5 was capped at val ~44 by my model's basin. frieren just took
  the lead at test 35.05 with their own iter4 ckpt (192/6/6/**64**, fun_dim=22, space_dim=2,
  val 40.97). Their slice_num=64 means I cannot weight-merge with my 192/6/6/128 lineage —
  the slice projection layers are differently shaped. Cleaner play: switch arch to match frieren,
  warm-start from their best, and chain-train another 30 min with my own LR/p_weight tweaks.
- **Change:** train.py model_config → `space_dim=2 fun_dim=22 n_hidden=192 n_layers=6 n_head=6
  slice_num=64 mlp_ratio=2`. Run: `--warm_start kr1xvas8 --train_subsample 0 --batch_size 2
  --lr 5e-6 --p_weight 5 --surf_weight 10 --epochs 15` (lower LR, higher p_weight than frieren's
  iter4 to extract a final % on the surface).
- **Result:** wandb run, 13 epochs in 32.4 min. Best at epoch 13 →
  **val/avg_surf_p = 40.10** (single 40.40 / geom_rc 53.50 / cruise 25.22 / re_rand 41.28) —
  improved on frieren's val 40.97 across every split. Predictions at
  `/mnt/new-pvc/predictions/apr27-5/alphonse/d985283/`. With frieren's val→test ratio 1.169,
  expected test ~34.3 — should beat their 35.05.
- **Verdict:** kept — bigger jump than v5→v6 polishing would have given (44→43 vs 41→40).
- **Notes:** v6 (chain-train from v5 at lr=1e-6) was killed at epoch 1 once frieren's leaderboard
  jump revealed a much stronger warm-start source. Trajectory still descending at the timeout —
  another chain at lr=2e-6 or with even higher p_weight (6-8) is the obvious follow-up.

### 2026-04-27 — v5 chain-train from v4 — full mesh + bs=2 + lr=5e-6 (thorfinn recipe)
- **Hypothesis:** thorfinn (apr27-5 leader at test 44.55) revealed in their journal that the unlock for them
  was switching from sub-30K bs=8 to **bs=2 + full mesh** at low LR for chained finetuning. My val/test
  ratio (1.07) is much worse than thorfinn's (1.16, val→test improves), which suggests sub-sampled
  training is overfitting to sub-meshes — full meshes during training should narrow that gap.
- **Change:** train.py — `--train_subsample 0` now means use the full mesh; predict.py default bs lowered
  to 2 (cruise meshes are ~190K nodes; bs=4 OOMs on slice_num=128). Run: `--warm_start v4_best
  --train_subsample 0 --batch_size 2 --lr 5e-6 --epochs 30`.
- **Result:** wandb run `2aj1vv9v`. 9 epochs in 32.6 min (218 s / epoch). Best at epoch 8 →
  **val/avg_surf_p = 44.70** (single 45.18 / geom_rc 60.26 / cruise 27.80 / re_rand 45.55) —
  bigger drops on every split, especially re_rand (53.63 → 45.55, -8). Predictions at
  `/mnt/new-pvc/predictions/apr27-5/alphonse/2650c09/`. predict.py initially OOM'd on cruise (bs=4
  default); re-ran with --batch_size 2 successfully.
- **Verdict:** kept — biggest single-iteration improvement so far (50.20 → 44.70, –5.5 on val).
  Already below thorfinn's *test* of 44.55.
- **Notes:** memory peak 42 GB (still < half budget). Loss still decreasing at the timeout — another
  chain at lr ~1e-6 should give one more % easily. The fact that re_rand dropped from 53.6 to 45.6
  confirms full-mesh training was the missing ingredient: subsampling under-represented the OOD-Re
  modes during training.

### 2026-04-27 — v4 chain-train from v3 (LR=2e-5, 30 epochs)
- **Hypothesis:** v3 trajectory was still descending at the timeout, so another 30 min from the
  v3 best ckpt with a *lower* LR (2e-5 vs 5e-5 in v3) should let the optimiser keep peeling off
  surface-pressure error without disturbing the basin we landed in.
- **Change:** no code changes — just `--warm_start` to v3's best (`model-4csd0d8b/checkpoint.pt`),
  `--lr 2e-5 --epochs 30`. v3 scored on the apr27-5 leaderboard at 48.54 (#2) — independent
  confirmation that the warm-start recipe + best-by-surf-p selection works.
- **Result:** wandb run, 29 epochs in 30.2 min. Best at epoch 25 → **val/avg_surf_p = 50.20**
  (single 41.81 / geom_rc 70.48 / cruise 34.86 / re_rand 53.63). Predictions saved to
  `/mnt/new-pvc/predictions/apr27-5/alphonse/baab103/`. Expected test ≈ 46.9 if val/test ratio holds
  (askeladd 0.935, my v3 0.935).
- **Verdict:** kept — slight but consistent improvement on every split vs v3 best.
- **Notes:** trajectory plateaued around 50.2-50.4 in the last 10 epochs — chain-training another round
  with the same recipe will probably give diminishing returns. Better next moves: (a) Polyak/SWA
  weight average across late-epoch checkpoints (need to start saving them); (b) prediction-space
  ensemble of v3 + v4 + askeladd; (c) push for capacity by training a from-scratch 256/8/8/128 with
  L1 + best-by-surf-p (no warm-start) and ensembling its predictions in.

### 2026-04-27 — v3 warm-start from askeladd's apr27-5 leader + L1 + best-by-surf-p
- **Hypothesis:** the apr27-5 leader askeladd (test 51.22) reached val=54.79 by warm-starting
  from thorfinn's apr27-bis ckpt (192/6/6/128, fun_dim=24/space_dim=0). Continuing the same recipe
  *from askeladd's own checkpoint* should push val below 54 in 30 minutes — beating askeladd's test.
- **Change:** train.py — switch architecture to 192/6/6/128 fun_dim=24 space_dim=0 to match the warm-start
  shape; add `--warm_start <path>` (loads state_dict with `strict=False`); switch loss to L1
  (matches the leaderboard MAE metric); pick best checkpoint by `val/avg_surf_p` (the actual leaderboard
  metric) instead of combined val/loss; LR 5e-5, surf_weight=10, p_weight=3, train_subsample=40 000;
  print surf_p MAEs in the per-epoch summary so we can read the leaderboard signal directly.
- **Result:** wandb run, 29 epochs in 30.3 min from `model-rriy9vrf` warm-start.
  Best at epoch 26 → **val/avg_surf_p = 51.85** (single 44.70 / geom_rc 71.84 / cruise 36.00 / re_rand 54.87)
  vs askeladd's val=54.79 (single 48.29 / geom_rc 74.71 / cruise 38.11 / re_rand 58.05) — improved on
  every split. If the askeladd val→test ratio (0.935) holds, my test should land near ~48.5.
  Predictions saved to `/mnt/new-pvc/predictions/apr27-5/alphonse/ade83e9/`.
- **Verdict:** kept — clear improvement on every val split; lowest val/avg_surf_p I have measured.
- **Notes:** v2 (Fourier-8 features + p_weight=10 from-scratch) was started after v1 but killed at
  epoch 2 once I read askeladd's transcript and saw the warm-start recipe — I expected the warm-start
  to dominate any from-scratch architectural tweak in the 30 min budget. Trajectory was still descending
  at the timeout — chain-training from this checkpoint with another 30 min should help. The hardest
  split remains `geom_camber_rc` at 71.84; cruise is now small (36) and likely close to its floor.

### 2026-04-27 — v1 Transolver-256x8 + sub32K + p-weighted MSE + surf_w=20
- **Hypothesis:** Larger Transolver (256/8/8/64) than the 128/5/4/64 default + node subsampling for speed
  + extra weight on the pressure channel (y_std≈679 vs 22/10) + surf_weight=20 should beat baseline,
  matching the parameter regime of apr27 leaders (frieren 256/8 → 42.11 surf-p MAE).
- **Change:** train.py — `n_hidden=256, n_layers=8, n_head=8, slice_num=64, mlp_ratio=2`,
  bf16 autocast, channel weights `[1, 1, 3]` on normalized MSE, `surf_weight=20`,
  cosine LR from `7e-4`, grad-clip 1.0, train_subsample 32 768 nodes/sample with 50 % surface oversampling
  (full mesh at validation). predict.py reads `model_config` from the checkpoint payload.
  Also wrapped the training entry-point in `main()`/`if __name__ == "__main__"` so predict can import Transolver
  cleanly. Whitelisted `tandemfoil-competition/kaggler/checkpoints/best.pt` in the root .gitignore.
- **Result:** wandb run `go992jul` (kagent-tandemfoil5). 29 epochs in 30.8 min, peak 13.8 GB.
  Best at epoch 25, val/loss = 5.84 (combined). Per-split val/loss at best:
  single_in_dist 5.21 · geom_rc 9.81 · geom_cruise 2.50 · re_rand 5.84.
  Test predictions saved to `/mnt/new-pvc/predictions/apr27-5/alphonse/5f51a09/` (4 splits × 200 samples).
  Scoring still pending at journal-write time.
- **Verdict:** kept — first credible submission; fast (≈63 s/epoch) and well below the 96 GB VRAM budget,
  so plenty of headroom for the next iteration.
- **Notes:** loss kept descending into epoch 29; cosine LR likely under-utilised (didn't fully decay).
  geom_camber_rc remained the hardest split (val/loss 9.8). Surf MSE was still ~0.34 (normalized) —
  big room for improvement on the leaderboard metric (surf p MAE in physical units).
  Next ideas: (a) longer effective training via bigger batch / higher slice_num; (b) Re-aware decoder
  conditioning to help the OOD-Re split; (c) Fourier features on (x,z) for high-frequency turbulence.
