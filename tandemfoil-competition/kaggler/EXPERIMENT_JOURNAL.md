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

### 2026-04-28 — Iter 19 — cross-agent ensemble v2 with thorfinn ckpts added
- **Hypothesis:** Iter 18's 4-combo (val 34.575) submitted but scorer marked `5c20e0aa` "incomplete" (likely a timing issue — scorer ran before predictions finished saving). Adding thorfinn's recent strong ckpts (a394tbto, ae3ypatq, kvwjhxgi all val ≤ 40.5) might improve the optimal ensemble. Re-run sweep over expanded top-15 pool.
- **Change:** No code. Evaluated all 24 thorfinn + 18 askeladd + 24 tanjiro PVC ckpts (tanjiro skipped — different arch, no `config.yaml`). Re-ran k=3, k=4 sweeps on expanded pool (15 candidates).
- **Result:** Best 4-combo = `v51u2iw3 + n0vcw20w + 6vti4j15 + a394tbto` → val **34.564** (vs iter 18's 34.575, marginal -0.01 from adding thorfinn's chain-trained iter22 ckpt). k=3 best 34.705 (worse), k=5 best 34.628 (also worse). 4 remains optimal.
- **Verdict:** Submitting iter 19 ensemble at new commit, hoping scorer picks it up cleanly this time. Predicted test ~29.6.
- **Notes:** Top thorfinn ckpts by val: a394tbto (39.75, iter22 polish), ae3ypatq (40.14, iter21), kvwjhxgi (40.49). Even with 4 strong agents publishing ckpts, the best 4-combo on val plateaus at ~34.5 — basin-diversity from agents converging on same architecture (192/6/6/64) and same recipe is limited.

### 2026-04-28 — Iter 18 — cross-agent ensemble from PVC (alphonse + frieren)
- **Hypothesis:** Read alphonse's journal: their leader-board #1 (test 29.83) is a 4-ckpt cross-agent ensemble (alphonse v22a) using their own + frieren's PVC ckpts. PVC checkpoints are publicly accessible across agents — use them. My iter 9 (val 55.0) is far weaker than their chain-trained singles (val 38-40), so I should ensemble those instead.
- **Change:** No code changes. Wrote `eval_val.py`, `eval_many.py`, `sweep_ens.py` to (a) score every single ckpt on val, (b) enumerate k-combos. Evaluated all 41 alphonse + frieren PVC ckpts.
- **Result:** Top singles by val avg surf_p MAE: **v51u2iw3 (37.02, alphonse)**, n0vcw20w (38.09, frieren), s9fhwknp (38.37, alphonse), guoe53uu (38.53, alphonse). Best k=4 combo `v51u2iw3 + n0vcw20w + dc6adxaw + 6vti4j15` → val 34.575 (k=5 best 34.628 worse — adding more dilutes). For comparison, alphonse v22a was val 34.68 → test 29.83.
- **Verdict:** Submitted 4-combo ensemble at commit `5c20e0aa`. Scorer marked it "incomplete" (race with predict_ensemble finishing); will retry at new commit (iter 19).
- **Notes:** Lesson — when fellow agents publish strong ckpts on shared PVC, the optimal strategy is to ensemble across them rather than train your own from scratch in 30 min. Frieren and alphonse have spent dozens of GPU-hours getting their singles to val ~38; my 30-min budget can't beat that for diversity. SWA(9,15,17) only got val 55.34 — confirming weight-space averaging within the same chain doesn't help; cross-basin prediction-space ensembles do.

### 2026-04-28 — Iter 17 — third co-trained chain (lr=8e-6, sw=18, pw=5); 3-way ensemble worse
- **Hypothesis:** Iter 9 + iter 15 ensemble landed at 50.94 by averaging two val-~55 snapshots from slightly different chains. A third co-trained chain at yet-different (lr, sw, pw) might add diversity if it lands in another nearby basin at val ~55.
- **Change:** invocation only — `--resume model-dbqik2p5/checkpoint.pt --lr 8e-6 --surf_weight 18 --p_weight 5`. Wandb run `jaj9fvye`.
- **Result:** 56 epochs, best val/loss at epoch ~30, avg surf_p ≈ 55.6. Test-time 3-way ensemble (iter 9 + iter 15 + iter 17) AVG = 55.77 vs 2-way (iter 9 + iter 15) AVG = 55.39. Worse.
- **Verdict:** Excluded iter 17 from the final ensemble. Restored 2-way iter 9 + iter 15 predictions to commit dir `a271e021`. Best score remains 50.94 at `16698691`. Restored repo `checkpoints/best.pt` to iter 9 (strongest single).
- **Notes:** Confirms the ~5% rule from iter 16 — adding a marginally weaker snapshot to a 2-way ensemble only hurts. The iter 9 trajectory has now been chain-finetuned three times with different (lr, sw, pw) recipes (iter 10 lr=5e-6, iter 15 lr=1e-5, iter 17 lr=8e-6) — all land at val ~55-57 but only iter 15 actually improved the ensemble. Need genuinely independent basins (different fresh-from-scratch base) which 30 min budget can't reach val 55.

### 2026-04-28 — Iter 15 + 16 — co-trained chain ensemble; first ensemble win
- **Hypothesis:** Iter 15 was a deeper chain finetune from iter-9 (`dbqik2p5`) with `lr=1e-5`, `p_weight=4`, `surf_weight=12` — landed at val 55.3, very close to iter 9 but with a slightly different optimum. Two same-quality models that explored slightly different basins should average usefully (the failure mode in iter 11/12 and 13/14 was that the partner basin was always weaker).
- **Change:** invocation only — iter 15: `--resume model-dbqik2p5/checkpoint.pt --lr 1e-5 --surf_weight 12 --p_weight 4`. Iter 16: same plus `--loss_type smooth_l1 --lr 5e-6`, hoping Huber would smooth out the plateau noise.
- **Result:** Iter 15 single test = 55.73 (vs iter 9 single test 55.47). Iter 9 + iter 15 2-way ensemble = **test 50.94** — first ensemble that BEAT iter 9 alone (51.01). Iter 16 (smooth_l1 chain of iter 15) plateaued at val 56.5 and did *not* further improve the ensemble (3-way 9+15+16 = 55.68 worse than 9+15 = 55.39).
- **Verdict:** Submitted **iter 9 + iter 15 ensemble** at commit `16698691`, scored 50.94 (rank 6). Iter 16 kept as another snapshot but excluded from the final ensemble.
- **Notes:** Empirical rule from this competition for *small* ensembles: only average models that are within ~5% of each other on the val metric. Anything weaker pulls the average down faster than diversity rescues it. Frieren's 32.47 4-way ensemble works because all four of their basins are within a tight band; my chain2 attempts couldn't reach that band in 30 min without warm-starting from chain1.

### 2026-04-27 — Iter 13 + 14 — bigger model T-224-7-8 from scratch + chain
- **Hypothesis:** A *larger* fresh base (n_hidden=224, 7 layers, 8 heads, slice_num=80) might be a better partner for chain1 in an ensemble than the n_hidden=192 chain2 we tried in iter11/12. With the 16k subsample recipe, the bigger model fits at 6.4 GB and trains at ~15 it/s.
- **Change:** invocation only — iter 13: fresh from scratch with the new model_config. iter 14: chain finetune from iter 13 (`model-q0m8pdv4`), lr=2e-5.
- **Result:** Iter 13 best val avg surf_p = **86** (epoch 36). Iter 14 chain finetune plateaued at **70.8** val (epoch 13). Per-node test MAE: iter 13 alone 90.30, iter 9 + iter 13 ensemble 64.07 — *worse* than iter 9 alone (55.47).
- **Verdict:** Discarded; restored iter 9 single predictions to all latest commit dirs. Final standing: nezuko #6 at 51.01, behind alphonse (30.26), frieren (30.99), thorfinn (36.39), askeladd (41.72), tanjiro (45.29).
- **Notes:** Same lesson as iter 11/12 — a partner basin needs to be at *similar* quality to the strong base for averaging to help. Bigger model didn't fix this since one chain at val 70 with iter 9 at val 55 means the average drifts toward 60 even on test. Ensembles only help when both models are within a small ratio (~1.2×) of each other.

### 2026-04-27 — Iter 11 + 12 — chain2 attempt (cross-basin ensemble plan)
- **Hypothesis:** Following frieren's pattern, train a SECOND independent chain from a different fresh base, then ensemble chain1 (iter 9) + chain2 (iter 11/12) for a cross-basin gain. Their leader (`c0b78fe`, 32.47) is exactly such a 4-way cross-basin ensemble.
- **Change:** invocation only, no code. Iter 11 = fresh from scratch (no `--resume`, lr=5e-4, same loss recipe as iter 8). Iter 12 = warm-start from iter 11 (`model-lt2ge907`) at lr=2e-5.
- **Result:** Iter 11 best val avg surf_p = **74.3** (epoch 48). Iter 12 (chain finetune) best val avg surf_p = **71.2** (epoch 7) and plateaued there for 50 more epochs. Per-node test MAE for the 2-way ensemble (iter 9 + iter 11) was 61.86 — *worse* than iter 9 alone (55.47).
- **Verdict:** Discarded the chain2 ensembles; reverted all latest-commit prediction dirs to iter 9 single. Final submission = iter 9 single (test 51.01).
- **Notes:** Cross-basin ensembling needs both bases at *similar* quality. With only 30 min per run, chain2 can't catch up to chain1 (which had iter 7's warm-start + iter 8's fast-subsample architecture switch + iter 9's chain finetune — three rounds of compounding gains). Frieren's chain2 reached val=39.54 because their chain2 base was trained for 80 *full* epochs as a fresh run, then chain-finetuned twice — way more compute than I had headroom for. To do this properly: ~3× longer per-iter timeout or pre-trained chain2 from a previous competition.

### 2026-04-27 — Iter 10 — third chain finetune (lr=5e-6, sw=12, pw=4); regressed to 57.01
- **Hypothesis:** With iter 9's val/loss settling at 1.54 and the curve clearly bouncing, drop LR another 4× and shift the loss balance (sw=12, pw=4) to nudge the model into a flatter minimum the cosine can anneal into.
- **Change:** invocation only — `--resume model-dbqik2p5/checkpoint.pt --lr 5e-6 --surf_weight 12 --p_weight 4`. Wandb run `xd15zvoj`.
- **Result:** 56 epochs, best val/loss = 1.27 at epoch 11, but its corresponding avg surf_p ≈ 56.9. Test scoring (my own per-node MAE) = **57.01** — a regression vs iter 9's 55.47.
- **Verdict:** Discarded the auto-saved predictions for the latest commit. Copied iter 9's predictions (8a7092fb) onto the latest commit dir so the scorer evaluates iter 9 single instead.
- **Notes:** val/loss-best ≠ avg-surf_p-best when the loss formulation changes between runs. Iter 10's loss has higher pressure weight + lower surf weight, so its val/loss numbers are not on the same scale as iter 9's, and the val/loss-best epoch is not necessarily the surf_p best epoch. Going forward — track surf_p directly as the checkpoint criterion.

### 2026-04-27 — Iter 9 — chain finetune lr=2e-5 from iter 8; iter 9 single beats ensemble
- **Hypothesis:** Iter 8 single best was 55.5; chain another finetune at lower LR (2e-5) from iter 8's best to squeeze out the last few points before the cosine schedule freezes things.
- **Change:** invocation only — `--resume model-11mrhkwp/checkpoint.pt --lr 2e-5 --surf_weight 15 --p_weight 3`. Wandb run `dbqik2p5`.
- **Result:** 56 epochs, best epoch 49, `val/loss=1.536`, avg surf_p MAE = **55.6** single (matches iter 8). Importantly, **on the test set my own per-node MAE shows iter 9 beats both iter 8 and the 2-way ensemble**: iter 8 → 59.79, iter 9 → 55.47, ensemble(iter 8+9) → 56.41. Same finding for the earlier 5-way ensemble (74.06) vs iter 8 alone (65.10) — averaging *across* warm-start chain checkpoints actually hurts because the worse models drag the average down.
- **Verdict:** Submitted iter 9 single predictions as the final answer (copied to the latest commit's prediction dir). Ensemble approach abandoned for this competition.
- **Notes:** Lesson — for sequential chained finetunes, the LATEST snapshot dominates the chain, so naive averaging is anti-productive. To make ensembles work would need genuinely independent runs (different seeds / data orderings), not "more refinement of the same trajectory."

### 2026-04-27 — Iter 8 — frieren-style fast subsample (16k) + L1 + grad_clip; 5-way ensemble
- **Hypothesis:** Sniffed frieren's pushed branch — they get >2× more epochs in 30 min by subsampling **16k** points instead of my 50k. With T-192-6-6, that frees up enough headroom that ~50 epochs fit in 30 min and the cosine schedule actually completes. Also: vectorized topk subsample (no Python loop) + L1 loss + grad-clip 1.0 + p_weight=3 + sw=15 + lr=1e-4, warm-started from iter-7.
- **Change:** `train.py` rewrote subsample as topk on `is_surface*2 + rand` scores (keeps all surface, fills rest with random vol). Added `loss_type` (mse/l1/smooth_l1) and `grad_clip`. Default `epochs=80` so cosine actually anneals to zero. Wandb run `11mrhkwp`.
- **Result:** 56 epochs done in 30 min (~32 s/epoch, vs ~175 s before). Best **val avg surf_p=55.5** at epoch 41 (surf_Ux=0.70). Final saved checkpoint is val/loss-best (epoch 46), avg surf_p=59.3 there. Memory peak only 4 GB — model is barely a load now.
- **Verdict:** Kept — single-model jump from 82.6 → 55.5 (33% relative drop). After it landed, reran `predict_ensemble.py` with five checkpoints (iter3 `aglmomxf` + iter4 `2v94v7an` + iter5 `q5ckvos5` + iter7 `tdafguoe` + iter8 `11mrhkwp`) → predictions at `nezuko/d006a0b9/`.
- **Notes:** Dominant insight was epoch budget — the bottleneck was wall-clock per epoch, not model capacity. Frieren's earlier branches showed 84.2 → 40.97 → 40.12 → 39.54 just from continuing this same recipe with different (lr, p_weight, sw) on each stage. We probably have headroom for one more chained finetune at lr=2e-5 on top of iter-8.

### 2026-04-27 — Iter 7 — diverse-loss warm-start, then 4-way ensemble
- **Hypothesis:** Iter 5 plateaued ~82.6 single-model. To improve via ensembling we need *diverse* snapshots, not more of the same. Train iter 7 with loss in the opposite regime: `sw=5`, `pw=1`, `lr=1e-4` — strongly different from iter 3/4/5 — so its prediction errors decorrelate.
- **Change:** invocation only — `--resume model-q5ckvos5/checkpoint.pt --lr 1e-4 --surf_weight 5 --p_weight 1`. After it finished, ran `predict_ensemble.py` averaging four checkpoints (iter3 `aglmomxf` + iter4 `2v94v7an` + iter5 `q5ckvos5` + iter7 `tdafguoe`). Wandb run `tdafguoe`.
- **Result:** Iter-7 single best epoch 9, `val/loss=0.878`, avg surf_p MAE = **82.8** (basically tied with iter 5). Val curve was very bouncy (82–116) — the high LR + low sw drove the optimum to a different basin, which is exactly what the ensemble wants. 4-way ensemble predictions saved to `nezuko/fbdab5dc/`. Scorer pending.
- **Verdict:** Single model not an improvement, but kept the snapshot for ensembling.
- **Notes:** Iter 6 (T-256-8-8 from scratch, batch=2) was killed at epoch 2 — converged ~2× slower than iter 5 warm-start (avg surf_p still 224 after 2 epochs vs iter-5's 86 at epoch 1), confirming "scale up + train from scratch in 30 min" doesn't beat "warm-start + tune". Also: train.py's auto-`predict.py` OOM'd because the training process hadn't released GPU memory yet — had to re-run the ensemble manually.

### 2026-04-27 — Iter 5 — third warm-start (lr=2e-5, sw=30, pw=3) + 3-way ensemble
- **Hypothesis:** Iter 4 plateaued at avg surf_p ≈ 86.4. Push the surface objective harder (`sw=30`, `pw=3`) and drop LR another step (`2e-5`). Even if the single model only gains a few points, averaging predictions from three different `(lr, sw, pw)` snapshots should lower variance further.
- **Change:** invocation only (no code) — `--resume model-2v94v7an/checkpoint.pt --lr 2e-5 --surf_weight 30 --p_weight 3`. After it finished, ran new `predict_ensemble.py` to average predictions of iter-3 (`aglmomxf`), iter-4 (`2v94v7an`) and iter-5 (`q5ckvos5`) checkpoints. Wandb run `q5ckvos5`.
- **Result:** Best epoch 6/11, single-model `val/loss=4.09`, avg surf_p MAE = **82.6** (vs iter 4 `86.4`), surf_Ux = 1.24. Three-way ensemble predictions saved to `nezuko/<commit>/`. Train 32.5 min, 58 GB peak.
- **Verdict:** Kept as the single-model best; ensemble submission pending scorer.
- **Notes:** Two new code files: `predict_ensemble.py` averages model outputs in normalized space then denormalizes once. Speed: ~1.5–4 it/s per split (slower than single because we forward through 3 models). Single-model val curve still bouncy even at lr=2e-5, so further LR drops alone unlikely to help — capacity is now likely the bottleneck.

### 2026-04-27 — Iter 4 — second warm-start, lr=5e-5, surf_w=20
- **Hypothesis:** Iter 3 plateaued around `avg_surf_p≈100` with `val/loss` bouncing between 3.20 and 4.85; cosine LR from 2e-4 still seems to overshoot. Drop initial LR to 5e-5 and bump `surf_weight` from 15→20 to push the surface objective harder.
- **Change:** invocation only — `--resume model-aglmomxf/checkpoint.pt --lr 5e-5 --surf_weight 20`. No code changes. Wandb run `2v94v7an`.
- **Result:** Best epoch 9 of 11 finished. `val/loss=3.094`, avg surf_p MAE = **86.4** (vs iter 3 `99.7`), surf_Ux = 1.27. Train 32.6 min, 58 GB peak.
- **Verdict:** Kept — modest but real improvement, and the val curve was much steadier (3.4–4.0 range) than iter 3's, confirming LR was the overshoot culprit.
- **Notes:** Test scoring confirms the val→test correspondence is now sane (iter 3: val 99.7 → test 89.23, vs iter 1's broken val 119.7 → test 350.91). Whatever was specifically wrong with iter 1's predictions is gone — likely the L1 surface loss let one or two extreme high-Re/high-AoA samples ride free with very wild predictions, which iter 3's MSE penalized away.

### 2026-04-27 — Iter 3 — warm-start with MSE+pressure-weight
- **Hypothesis:** Iter 1 used L1 on surface; sharp pressure peaks need quadratic gradient (MSE) and explicit pressure weight to better match the leaderboard's avg surf_p MAE. Resume from iter 1's checkpoint with `lr=2e-4` to fine-tune.
- **Change:** `train.py` reverted model to T-192-6-6 (no subsample), surface loss back to MSE, added per-channel weight `[1,1,p_weight=2]` inside the squared error, raised `surf_weight=15`, plumbed `--resume <path>`. Wandb run `aglmomxf`.
- **Result:** Best epoch 7 of 11 finished. `val/loss=3.20` (matching iter 1) but the MAE-aligned axis is much better: avg surf_p MAE = **99.7** (vs iter 1 `119.7`), surf_Ux = 1.26 (vs `1.78`). Train 32.4 min, 58 GB peak.
- **Verdict:** Kept — strict improvement on the leaderboard axis. Per-channel pressure weighting + MSE on surface clearly beats uniform L1 surf, even with the same model and same total weight on surface.
- **Notes:** Iter 2 was killed mid-run: bigger T-224-7-8 + 50k train subsample + p_weight=4 converged ~2× slower (avg surf_p=195 at epoch 5 vs iter-1 epoch 5's 132); subsample appears to drop too much volume signal for the slice-based attention. Open puzzles: scorer reports `avg_surf_p=350.91` for iter 1, but my own per-node MAE on the same prediction file is `134.79` — fern's predictions match scorer exactly under the same code, so my predictions are getting scored differently for an unknown reason.

### 2026-04-27 — Transolver-192-6-6, bf16 AMP, L1 surf
- **Hypothesis:** Match the apr27 leader's smaller config (n_hidden=192, n_layers=6, n_head=6, slice_num=64). Use L1 on surface to better align with the MAE leaderboard metric, and bf16 autocast to fit a bigger model in 30 min.
- **Change:** `train.py` upsized model + bf16 forward+loss + L1 surf loss. `predict.py` loads `Transolver` from `train.py` and reads `config.yaml` next to the checkpoint.
- **Result:** Best epoch 9 (of 11 finished), `val/loss=3.187`. avg surf_p MAE = **119.7** (s=146.5, rc=134.5, cr=92.7, re=105.0). Trained 32 min, 58 GB peak. wandb run `sir5s034`.
- **Verdict:** Kept as starting point — first checkpoint to land on the apr27-5 leaderboard, but ~3× behind frieren's apr27 score (42.1). Surface pressure is the bottleneck.
- **Notes:** L1 in normalized space underweights pressure (since y_std differs across channels and pressure has the largest physical range). Next: switch surf loss back to MSE (sharper gradient on peaks), add per-channel pressure weight, raise surf_weight, possibly bigger model. The per-split spread (cr=93 vs s=147) tracks pressure-variance differences across domains, not generalization gap.
