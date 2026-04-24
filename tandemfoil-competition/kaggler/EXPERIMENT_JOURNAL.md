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

### 2026-04-24 — v17-h128-e14-sw1.5 (KEPT)
- **Hypothesis:** combine v15's tuned surf_weight (1.5) with v8's smaller-faster model (h=128, n_head=4, epochs=14). Each lever was validated independently; the combination may stack.
- **Change:** `train.py` — `n_hidden: 160 → 128`, `epochs: 12 → 14`. `surf_weight=1.5` (inherited from v15).
- **Result:** 13 of 14 epochs completed (timeout). **Avg val surf_p MAE 96.4 → 95.6 (-0.8%).** Three splits improved (re_rand -3.5% led), single_in_dist +2.1% regression. W&B `kagent-tandemfoil/nlfzya86`.
- **Verdict:** kept. Cumulative: 120.4 → 95.6 (**-20.6%**).
- **Notes:** Levers stack but the marginal return is now small (<1%). Next: try `lr=7e-4` or a different axis (regularization / augmentation) since the obvious dial-adjusting seems to be saturating.

### 2026-04-24 — v16-sw1 (FAILED, discarded)
- **Hypothesis:** extend the sweep — go to `surf_weight=1.0` to see if even less surface emphasis helps.
- **Change:** `train.py` — `surf_weight: 1.5 → 1.0`.
- **Result:** all 12 epochs. **Avg val surf_p MAE 96.4 → 97.9 (+1.5%, WORSE).** All splits regressed. W&B `kagent-tandemfoil/0v2l6pn6`.
- **Verdict:** discarded, reset to v15. `surf_weight=1.5` is the optimum under this architecture/budget.
- **Notes:** Sweep summary for reference: `sw=10 → 5 → 3 → 1.5 (min) → 1.0`. U-shape confirmed. Not worth sweeping this axis further.

### 2026-04-24 — v15-sw1.5 (KEPT)
- **Hypothesis:** continue the surf_weight sweep that v13/v14 validated — lower surface weight, more volume gradient share.
- **Change:** `train.py` — `surf_weight: 3 → 1.5`.
- **Result:** all 12 epochs. **Avg val surf_p MAE 98.7 → 96.4 (-2.3%).** Three splits improved (camber_rc -4.7% led), re_rand flat (+0.3%). W&B `kagent-tandemfoil/mk60viyr`.
- **Verdict:** kept. Cumulative baseline: 120.4 → 96.4 (**-20.0%**). First sub-100 avg.

### 2026-04-23 — v14-sw3 (KEPT)
- **Hypothesis:** v13 (surf_weight=5) was a big win over v10 (surf_weight=10). Push the trend — surf_weight=3 gives volume even more relative gradient.
- **Change:** `train.py` — `surf_weight: 5 → 3`.
- **Result:** all 12 epochs completed. **Avg val surf_p MAE 99.7 → 98.7 (-1.0%).** Three splits improved, val_geom_camber_rc regressed +2.7%. W&B `kagent-tandemfoil/6xqsflxu`.
- **Verdict:** kept. Cumulative vs baseline: 120.4 → 98.7 (**-18.0%**).
- **Notes:** Gains are shrinking (v10→v13 was -4.2%, v13→v14 is -1.0%) — approaching another plateau. One more point in this direction (surf_weight=2) will tell if we've passed the optimum. If that still wins, try 1. If that regresses, optimum is between 2 and 3.

### 2026-04-23 — v13-sw5 (BIG WIN — KEPT)
- **Hypothesis:** v2 and v12 both pushed `surf_weight` UP and both failed. Maybe the arrow points the OTHER way: letting the volume loss dominate training could give volume features enough capacity to contextualise the surface prediction (wake-aware pressure).
- **Change:** `train.py` — `surf_weight: 10 → 5`. All other knobs identical to v10 (h=160, epochs=12).
- **Result:** all 12 epochs completed in 31.3 min. **Avg val surf_p MAE 104.1 → 99.7 (-4.2%) — broke the plateau, crossed under 100 for the first time.** All 4 splits improved:
    - `val_single_in_dist`: 121.8 → 117.9 (-3.2%)
    - `val_geom_camber_rc`: 118.7 → **108.6 (-8.5%)**
    - `val_geom_camber_cruise`: 79.6 → 77.7 (-2.4%)
    - `val_re_rand`: 96.4 → 94.8 (-1.7%)
  W&B `kagent-tandemfoil/a7x2spsc`.
- **Verdict:** kept. Ckpt committed.
- **Notes:** Cumulative gain from baseline is now **120.4 → 99.7 (-17.2%)**. The lesson inverts what v2/v12 suggested: the **stock `surf_weight=10` was already too high**, drowning out volume gradient signal. v2's earlier failure combined Huber + p-weight with over-weighted surface, so it couldn't see the volume-starvation effect in isolation. Plateau-breaker. Try `surf_weight=2` or `3` next — if the trend continues, even less surface emphasis wins.

### 2026-04-23 — v12-sw15 (FAILED, discarded)
- **Hypothesis:** push past the avg=104 plateau by up-weighting the surface loss (surf_weight 10→15) since the ranking metric is surface-p MAE.
- **Change:** `train.py` — `surf_weight: 10 → 15`, everything else v10-identical.
- **Result:** all 12 epochs completed. **Avg val surf_p MAE 104.1 → 105.5 (+1.4%, WORSE).** Every split regressed. predict.py auto-submit OOMed because training hadn't released GPU yet. W&B `kagent-tandemfoil/pz648ti4`.
- **Verdict:** discarded, reset to v10.
- **Notes:** v2 (Huber + per-channel p weight) and v12 (flat surf_weight bump) both pushed this direction and both hurt. Strong signal: at `surf_weight=10` the surface/volume balance is close to optimal; extra surface weight starves volume features that the surface also needs (for wake/shock context). Don't re-try.

### 2026-04-23 — v11-h144-e14 (MARGINAL, discarded)
- **Hypothesis:** bracket the v8 (h=128, e=14) vs v10 (h=160, e=12) tie with the interpolated midpoint; maybe it captures both axes' wins.
- **Change:** `train.py` — `n_hidden: 160 → 144`, `epochs: 12 → 14`.
- **Result:** 12 of 14 epochs completed (timeout). Best val/loss=**2.456** (v10 2.510, -2.2%). **Avg val surf_p MAE 104.1 → 104.7 (+0.6%, marginally worse).** W&B `kagent-tandemfoil/1cbg4oz8`.
- **Verdict:** discarded — tied avg MAE and the ranking metric says v10 is marginally better. Reset to v10.
- **Notes:** h=128, h=144, h=160 all land at avg MAE ~104 with different split-tradeoffs. There's a genuine **plateau around avg 104** for this architecture/budget/seed. Val/loss keeps improving at h=144 but doesn't translate into avg-MAE gain. Next lever must be qualitatively different — bigger data (aug), different loss, or different optimiser — not another sweep of (h, epochs).

### 2026-04-23 — v10-h160-e12 (KEPT, MIXED)
- **Hypothesis:** v8 (h=128) and v9 (h=96) bracketed the optimum from below; push the other side — larger capacity with fewer (but still fully-decayed) epochs.
- **Change:** `train.py` — `n_hidden: 128 → 160`, `epochs: 14 → 12` (both chosen so T_max aligns with the wall-time that fits h=160 at ~158 s/epoch).
- **Result:** all 12 epochs completed in 31.5 min. Best epoch 11 val/loss=**2.510** (v8 2.603, -3.6% val/loss). **Avg val surf_p MAE 104.1 → 104.1 (essentially tied).** Peak VRAM 53 GB. W&B `kagent-tandemfoil/h21oleez`.
  - Per-split surface-p MAE:
    - `val_single_in_dist`: 130.2 → **121.8 (-6.5%)** — biggest single-split gain this competition
    - `val_geom_camber_rc`: 112.8 → 118.7 (+5.2%)
    - `val_geom_camber_cruise`: 77.3 → 79.6 (+3.0%)
    - `val_re_rand`: 96.0 → 96.4 (+0.4%)
- **Verdict:** kept — val/loss is a robust -3.6% and the single_in_dist split (hardest / highest-pressure regime) improved meaningfully. Avg-MAE is tied within statistical noise (100 samples per split).
- **Notes:** v8 and v10 trade off splits — v8 wins on camber-rc/cruise, v10 wins on in-dist/re-rand. Both are essentially equivalent on the ranking metric. **Val/loss difference suggests v10 has fewer extreme-pressure outliers** (MSE is sensitive to those), which should matter on the hidden test set. Next: try sitting in the middle (h=144 with an epoch count that lets T_max decay) — might capture both axes' wins.

### 2026-04-23 — v9-h96-e14 (FAILED, discarded)
- **Hypothesis:** extend v8's trend — even smaller backbone (`n_hidden=96`) should fit more epochs in the budget, further trading capacity for cosine-cycles.
- **Change:** `train.py` — `n_hidden: 128 → 96`, `epochs: 14 (unchanged, T_max aligned)`. First run was `epochs=18` but killed mid-epoch-1 once timing showed we'd hit timeout well before T_max, so restart with `epochs=14`.
- **Result:** all 14 epochs completed in 29.8 min (epoch time ~128 s). Best val/loss=**2.694** (v8 2.603). **Avg val surf_p MAE 104.1 → 108.3 (+4.0%, WORSE).** W&B `kagent-tandemfoil/28dgj6w7`.
- **Verdict:** discarded, reset code.
- **Notes:** Capacity reduction dominated the extra-epoch gain this time. v8's 128h already sat at the knee of the capacity/epoch tradeoff — going smaller regresses every split (worst: single_in_dist +5.3%). Useful signal: the optimum is not at the smallest viable model; there is a real capacity floor. Next: test the *other* side (144h or 160h) to check whether v8 is itself at the knee, or whether slightly larger still wins.

### 2026-04-23 — v8-small-longer (KEPT)
- **Hypothesis:** the 30-min budget is compute-bound, so a smaller backbone should free wall-clock time for more epochs — and more cosine-annealing stages pay off more than extra parameters in this regime. Iterations v4/v5/v7 all failed because they added compute that cut the epoch count.
- **Change:** `train.py` — `n_hidden: 192 → 128`, `n_head: 6 → 4` (128/4=32 clean), `epochs: 10 → 14`. Every other knob identical to v3.
- **Result:** best epoch 13/14, val/loss=**2.603** (v3 2.634). Wall 30.5 min, 13 epochs completed. Peak VRAM 47 GB (v3 71 GB).
  - Per-split surface-p MAE:
    - `val_single_in_dist`: 130.0 → 130.2 (+0.1%, flat)
    - `val_geom_camber_rc`: 122.4 → 112.8 (-7.9%)
    - `val_geom_camber_cruise`: 80.0 → 77.3 (-3.4%)
    - `val_re_rand`: 99.2 → 96.0 (-3.3%)
  - **Avg val surf_p MAE: 107.9 → 104.1 (-3.6%)**
  - W&B: `kagent-tandemfoil/badibqkx` (name `nezuko/v8-small-longer`).
- **Verdict:** kept. Ckpt committed at `d458add`.
- **Notes:** Smaller model actually trained BETTER on training set (train surf 0.166 vs v3's 0.223) — more epochs let it fit more. Validation improvements are asymmetric: unseen-camber/re splits improved, but the hardest in-distribution split (single_in_dist) was flat. Suggests the capacity drop hurt peak-pressure fitting slightly, but extra fine-tuning helped OOD generalisation more. 47 GB VRAM means there's substantial room to grow — v9 candidate is `n_hidden=96` (even smaller → even more epochs) to see if the trend continues.

### 2026-04-23 — v7-surfhead (FAILED, discarded)
- **Hypothesis:** a small 2-layer MLP ("surface specialist head") running on last-block features, adding a pressure correction only at surface nodes, should refine surface-p without disturbing volume fields.
- **Change:** `model.py` — `Transolver` splits the last block manually to extract pre-projection features, then `surf_mlp` produces a p-channel correction gated by `is_surface`. `train.py` — pass `is_surface` into the model dict; config flag `surface_head=True`. Similar plumbing in `predict.py`, `viz.py`.
- **Result:** best epoch 9, val/loss=2.771 (v3 2.634). **Avg val surf_p MAE 107.9 → 112.1 (+3.9%, WORSE).** Wall 31.6 min (9 epochs). W&B `kagent-tandemfoil/ykdvitjx`.
- **Verdict:** discarded, reset code.
- **Notes:** Per-epoch speed was same as v3 (the head is tiny). The extra +19K params + the clone/index-assign in forward don't dominate compute but add a new parameter group that the 9-epoch budget can't train well. The head likely needs longer training to coordinate with the backbone. Similar story to EMA — architectural additions without extra epochs don't pay off. Moral: in this tight budget, the winning moves are **free-on-compute AND free-on-optimization** (schedule fixes, not new params).

### 2026-04-23 — v6-ema (FAILED, discarded)
- **Hypothesis:** EMA of model weights (decay=0.998) used for validation + as the saved checkpoint should give ~1-3% free improvement by averaging out late-training noise (researcher's suggestion).
- **Change:** `train.py` — create `ema_model = deepcopy(model)`; update after every optimizer step; validate and save using EMA.
- **Result:** best epoch 9, val/loss=3.042 (v3 2.634). **Avg val surf_p MAE 107.9 → 114.5 (+6.2%, WORSE).** Wall 31.2 min (9 epochs). W&B `kagent-tandemfoil/tck9u1e2`.
- **Verdict:** discarded, reset code.
- **Notes:** EMA averages over a 500-step window (~1.33 epochs with decay=0.998). On our 9-epoch training the current model is still mid-convergence — EMA weights lag ~1.3 epochs behind, and an older model in this regime is a worse model (val/loss still dropping each epoch, not oscillating around a minimum). EMA needs long training where weights oscillate near a minimum. Don't re-try EMA until base training is saturating.

### 2026-04-23 — v5-slice128 (FAILED, discarded)
- **Hypothesis:** doubling-ish slice token count (96 → 128) gives finer spatial slicing without much compute cost.
- **Change:** `train.py` — `slice_num: 96 → 128`.
- **Result:** best epoch 8, val/loss=3.070 (v3 2.634). **Avg val surf_p MAE 107.9 → 123.2 (+14.2%, WORSE).** Wall 33.1 min (8 epochs). Peak VRAM 85 GB (v3 was 71 GB). W&B `kagent-tandemfoil/1ifpng5a`.
- **Verdict:** discarded, reset code.
- **Notes:** Same failure mode as v4 — the extra compute from larger slice-weight tensor (`[B,H,N,G]` doubled) slowed each epoch by ~20% (251 s vs 210 s), dropping us from 9 → 8 epochs. Net: fewer cosine-schedule steps, incomplete LR decay. **Rule for this codebase: compute-per-epoch is the binding constraint — any change that slows each batch needs a matching `epochs` adjustment, and usually nets negative.** Next iteration should be a compute-neutral tweak (hparam, optimizer, EMA, data aug done off-device).

### 2026-04-23 — v4-eidetic (FAILED, discarded)
- **Hypothesis:** Transolver++ Eidetic attention (Ada-Temp per-point temperature + Rep-Slice Gumbel-Softmax slice weights) gives +12.6% surface-p on DrivAerNet++ in the paper. Swap the default `PhysicsAttention` for `PhysicsAttentionEidetic` behind a `use_eidetic` flag (default True).
- **Change:** `model.py` — added `PhysicsAttentionEidetic` class; `TransolverBlock`/`Transolver` accept `use_eidetic`. `train.py` — config flag `use_eidetic=True`, wire through `model_config`.
- **Result:** best epoch 8, val/loss=3.058 vs iter-3 2.634. **Avg val surf_p MAE 107.9 → 115.4 (+7.0%, WORSE).** Wall 33.4 min (8 epochs completed). W&B `kagent-tandemfoil/qsjr11qa` (name `nezuko/v4-eidetic`).
- **Verdict:** discarded. Reset code commit. Model.py restored to stock `PhysicsAttention`.
- **Notes:** Per-epoch Eidetic was actually BETTER than stock (at epoch 7: v4 val/loss=3.50 vs v3 4.24). The failure is pure compute overhead: Gumbel noise + softplus temperature projection + extra Linear pushed per-epoch from 210 s to 251 s (+20%), dropping us from 9 → 8 epochs. That's fewer cosine-annealing steps, incomplete LR decay. Next time try Eidetic with `epochs=8` so `T_max` matches the new wall-time budget — that's likely a true win.

### 2026-04-23 — v3-cosine-match (KEPT)
- **Hypothesis:** baseline's cosine LR schedule (`T_max=50`) was 82% unused when we hit the 30-min budget at epoch 9. Shrinking `epochs` to 10 (same wall-time, same LR init, same loss) lets the schedule actually decay to near-zero by the end, giving a fine-tuning phase that the baseline never got.
- **Change:** `train.py` one-liner — `epochs: 50 → 10`. Nothing else touched. Same MSE loss (reverted the Huber experiment first).
- **Result:** **best val/loss=2.634 at epoch 9/10** (baseline 2.984). Wall 31.1 min, 9 epochs completed (epoch 10 pre-empted by timeout at the top of the loop).
  - Per-split surface-p MAE:
    - `val_single_in_dist`: 139.1 → 130.0 (-6.6%)
    - `val_geom_camber_rc`: 137.5 → 122.4 (-11.0%)
    - `val_geom_camber_cruise`: 93.7 → 80.0 (-14.7%)
    - `val_re_rand`: 111.3 → 99.2 (-10.9%)
  - **Avg val surf_p MAE: 120.4 → 107.9 (-10.4%)**
  - W&B: `kagent-tandemfoil/1e53ehji` (name `nezuko/v3-cosine-match`).
- **Verdict:** kept. Ckpt committed at `22d262b`.
- **Notes:** Every split improved. The biggest jump (`cruise -14.7%`) is the easiest split, suggesting the model still benefits from extra fine-tuning even on in-distribution cases. Since the baseline epoch-9 LR was still ~83% of init, the iter-3 model effectively gets to spend ~3-4 epochs near minimum LR where gradient steps are small and refining. Cheapest possible improvement so far.

### 2026-04-23 — v2-huber-sp3 (FAILED, discarded)
- **Hypothesis:** surface loss as SmoothL1 (Huber, beta=1) is MAE-aligned and should improve surface-pressure MAE; per-channel weight `[1, 1, 3]` on surface emphasizes pressure.
- **Change:** `train.py` only — swap MSE→Huber on surface, add channel weights, keep volume MSE. surf_weight=10 unchanged.
- **Result:** best epoch 7 val/loss=3.924 (baseline was 2.984). **Avg val surf_p MAE 120.4 → 141.9 (+17.9%, WORSE).** Per-split: single_in_dist +35.4%, geom_camber_rc +2.9%, geom_camber_cruise +25.3%, re_rand +8.3%. W&B: `s2fgd22j` (name `nezuko/v2-huber-sp3`).
- **Verdict:** discarded. Reset code commit.
- **Notes:** Strong negative signal. Hypothesis: Huber saturates the gradient for large pressure errors (|range| up to 4k m²/s²), while MSE keeps scaling the signal with the error magnitude — exactly where the turbulent/stagnation spikes live. Val/loss is computed with MSE, so the ckpt selection also drifts when training optimizes a different objective. Don't repeat Huber alone; if re-attempting L1/MAE training, also switch the val/loss selection to avg surf_p MAE so criteria match.

### 2026-04-23 — v1-baseline (bf16, 192h/6L/96slice)
- **Hypothesis:** establish starting point with the already-improved Transolver template (bf16 autocast, 192h/6L/96slice, cosine LR, grad clip 1.0).
- **Change:** none — ran the committed `v1` code (`e1a0894`) unchanged. Config: lr=5e-4, bs=4, surf_weight=10, epochs=50 (capped by 30 min wall time).
- **Result:** best val/loss=2.984 at epoch 9 / 9 (timeout). Wall 31.4 min. VRAM 71 GB peak.
  - Per-split surface-p MAE (physical units):
    - `val_single_in_dist`: 139.1
    - `val_geom_camber_rc`: 137.5
    - `val_geom_camber_cruise`: 93.7
    - `val_re_rand`: 111.3
  - **Avg val surf_p MAE ≈ 120.4** (proxy for leaderboard metric)
  - W&B: `kagent-tandemfoil/pkb50fop` (name `nezuko/v1-bf16-192x6`)
- **Verdict:** kept — seeds the leaderboard; committed `best.pt` at `0c35d6c`.
- **Notes:** val_single_in_dist was noisy (14.5 → 11.1 → 15.7 → 11.5 → 8.6 → 6.6 → 6.4 → 7.0 → 4.5). Cosine LR decays fully over 50 epochs, so we stopped at ~18% of the schedule — the model is far from converged. First auto-predict invocation OOMed because the training process hadn't released GPU memory yet; rerun standalone after exit finished cleanly.
