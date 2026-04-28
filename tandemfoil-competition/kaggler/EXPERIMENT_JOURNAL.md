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

### 2026-04-27 — iter15-ensemble-iter13+iter14 (iter 15)
- **Hypothesis:** Iter 13 (no noise, in_dist-strong) and iter 14 (with input noise, OOD-strong) have *complementary* errors. Averaging them should beat either alone, even though they share ancestry.
- **Change:** New `ensemble_predict.py`: loads multiple ckpts, averages forward outputs, writes to current commit dir. Validated on val set with `eval_ensemble.py`.
- **Result:** Val (across 4 splits): **mean_surf_p = 41.00** (single_in_dist=35.88, geom_rc=60.44, geom_cruise=20.10, re_rand=47.58). Beat iter 14 alone (41.44) by 0.44. Saved test predictions at commit `2fb108a`. Other ensembles tried: iter11+12+13 (43.15, ≈ iter13 alone — chain ckpts too correlated), iter4+iter14 (43.15, iter 4 is too weak), iter12+iter14 (41.06, ≈ iter13+14), iter12+13+14 (41.41, 3-way hurts).
- **Verdict:** Kept — best score yet. Confirms pattern from research: ensembles work when component models have *structurally diverse training* (here: noise vs no noise), not just different epochs of the same recipe.
- **Notes:**
  - Tested bs=1 vs bs=4 inference on iter 14 — exactly identical output (41.442 each). Our masking is correct everywhere; askeladd's bs=1 trick doesn't apply to us.
  - Best 2-way combo is iter13+iter14 (1:1). 3-way and weighted 2:1 variants didn't help.

### 2026-04-27 — iter14-chain-noise0.03 (iter 14)
- **Hypothesis:** Pure chain has plateaued. Add Gaussian input noise (σ=0.03 in normalised feature space, on dims 0-12 only — keep physical context dims 13-23 clean) as a regulariser. Expect OOD splits (geom_rc, re_rand) to benefit most.
- **Change:** Added `noise_std` config; train loop adds `randn_like(x) * noise_std` masked to dims 0:13 before model forward.
- **Result:** Best epoch 24, mean val surf_p MAE = **41.44** (single_in_dist=38.72, geom_rc=58.75, geom_cruise=21.53, re_rand=46.76). val/loss=1.42. 25 epochs in 28.9 min. Run: `frieren/iter14-chain-noise0.03` (`cz3zpalt`). Commit `2fb108a`.
- **Verdict:** Kept — **biggest single jump in late chain** (43.04 → 41.44, -1.60). Track movement is exactly what input noise should do: hurt in-dist (single +2.12, cruise +1.18) but big OOD wins (geom_rc -6.14, re_rand -3.58). Net win because OOD was the bottleneck.
- **Notes:**
  - Trajectory still descending at timeout (41.69 → 41.44 in last 2 vals). Continue chain with noise.
  - Crucially, noise was masked to dims 0-12 (geometry features) but NOT 13-23 (physical conditions Re/AoA/NACA/gap/stagger). Conditions should be exact; geometry can tolerate jitter.
  - Iter 13 (no noise) and iter 14 (noise) ensemble cleanly because they made different in-dist vs OOD trade-offs.

### 2026-04-27 — iter13-chain-sw25-lr5e-5 (iter 13)
- **Hypothesis:** With ContextFiLM + GlobalFiLM + GatedSliceAttn now all in place, push surface focus harder (`surf_weight=25` vs 20) and chain at lr=5e-5.
- **Change:** `--lr 5e-5 --surf_weight 25`. No code change.
- **Result:** Best epoch 22, mean val surf_p MAE = **43.04** (single_in_dist=36.60, geom_rc=64.89, geom_cruise=20.35, re_rand=50.34). 25 epochs. Run `47mxgkas`. Commit `a9ffffa`.
- **Verdict:** Kept — but only -0.08 over iter 12 (43.12). Plateau confirmed.
- **Notes:** surf_weight bump made train-loss bigger but didn't translate to better val; the L1 + p_weight=3 channel weighting was already sufficient.

### 2026-04-27 — iter12-ctxfilm-chain (iter 12)
- **Hypothesis:** Add a per-sample ContextFiLM conditioned on input dims 13-23 (Re/AoA/NACA/gap/stagger). Zero-init residual makes it identity at warm-start. Should help re_rand and geom_rc by giving the model an explicit, stable Re-conditioning route.
- **Change:** New `ContextFiLM` module (LN-free, MLP with zero-init last layer), `use_ctx` flag. Per-sample context = `x[:, 0, 13:13+ctx_dim]` (constant across nodes, taken from a real node).
- **Result:** Best epoch 24, mean val surf_p MAE = **43.12** (single_in_dist=36.63, geom_rc=64.82, geom_cruise=20.54, re_rand=50.50). 25 epochs. Run `t921h79f`. Commit `ed5e740`.
- **Verdict:** Kept — modest -0.28 over iter 11. geom_rc -0.94, re_rand -0.34 (the OOD splits the layer was meant to help). +0.09 in-dist trade-off.
- **Notes:** Architectural moves consistently help OOD but at small magnitude relative to chain. Combined arch is now: ResMLP × 6 + GlobalFiLM × 6 + ContextFiLM × 6 + GatedSliceAttn × 2 = 15.3M params.

### 2026-04-27 — iter11-chain-lr5e-5 (iter 11)
- **Hypothesis:** Iter 10's gated attention had only 23 epochs and γ was barely off zero. Chain at much lower LR (5e-5) so the new attention can converge cleanly without disturbing iter 10's well-tuned ResMLP+FiLM core.
- **Change:** `--lr 5e-5 --train_subsample 50000`. Same architecture as iter 10.
- **Result:** Best epoch 26, mean val surf_p MAE = **43.40** (single_in_dist=36.54, geom_rc=65.76, geom_cruise=20.47, re_rand=50.84). 27 epochs in 28.9 min. Run: `frieren/iter11-chain-lr5e-5` (`91891ntc`). Commit `696c0b5`.
- **Verdict:** Kept — modest -0.21 over iter 10. Trajectory increasingly flat (43.45 → 43.43 → 43.40 in last 3 vals). Pure chain returns are exhausted at this configuration.
- **Notes:**
  - Frieren scoring still all "incomplete" — predictions at /mnt/new-pvc/predictions/apr27-4/frieren/<commit>/ exist with correct sizes/shapes/dtypes vs scored agents. Format is identical to fern's accepted submissions. Mystery; nothing more I can do here besides keep submitting.
  - Leaderboard: thorfinn 39.52, askeladd 39.55, nezuko 39.79, edward 41.28. Need ~3.6 more val drop OR a favorable val→test gap.
  - Need a step-change architectural move for iter 12. Best bets: (a) explicit ContextFiLM conditioning on per-sample features (Re, AoA, NACA, gap, stagger), (b) richer pooling (mean+max+std), (c) bigger model (h=512). Going with (a).

### 2026-04-27 — iter10-attn2-chain (iter 10)
- **Hypothesis:** Iter 9 chain returns are flattening (-0.50 vs iter 8's -1.20). To break the plateau, add 2 gated zero-init slice-attention layers (with masking) at positions 2 and 5 of the 6 ResMLP blocks. Zero-init γ keeps warm-start an exact identity. Slice attention provides cheap global spatial context that mean-pool FiLM can't.
- **Change:** New `GatedSliceAttn` (LN + masked SliceAttention + per-channel learnable gate γ initialised to 0). New `n_attn_layers` config. Slice attention's softmax is masked against padding nodes (Edward's masking fix). Warm-started from iter 9 ckpt with `strict=False` (30 missing keys = new attn weights, all zero-gated at start).
- **Result:** Best epoch 20, mean val surf_p MAE = **43.61** (single_in_dist=36.75, geom_rc=66.03, geom_cruise=20.73, re_rand=50.92). 23 epochs in 28.4 min, 81s/val + 67s no-val (vs iter 9's 51+41s — attention adds ~30% per-step). 22.5GB peak. 13.50M params. Run: `frieren/iter10-attn2-chain` (`xl6pb63z`). Commit `087c101`.
- **Verdict:** Kept — marginal (-0.20) but real. geom_rc -0.55, re_rand -0.39 (the OOD splits the attention was meant to help). Trade-off: attention slowed steps, so fewer epochs; the gain per epoch is ~similar to chains so net: small.
- **Notes:**
  - Trajectory still descending at timeout: 43.71 → 43.61 → 43.69 in last 3 vals. With more time the attention may help more.
  - Idea for iter 11: chain iter 10 again at lower LR; the attention has barely started training (γ moves slowly from 0).
  - Diminishing returns from the chain alone — each iteration shaves 0.2-0.5. Need a step-change to reach thorfinn 39.95.

### 2026-04-27 — iter8-chain-lr1.5e-4 (iter 8)
- **Hypothesis:** Iter 7 was still descending at timeout. Halve LR (1.5e-4 vs 3e-4), keep cosine_epochs=80 so LR doesn't bottom, surf_weight=20. Same architecture (h=384 b=6 + GlobalFiLM).
- **Change:** Just CLI flags — `--warm_start iter7_best_backup.pt --lr 1.5e-4 --surf_weight 20 --cosine_epochs 80 --train_subsample 60000`. No code changes.
- **Result:** Best epoch 36, mean val surf_p MAE = **44.31** (single_in_dist=37.09, geom_rc=67.46, geom_cruise=20.96, re_rand=51.71). val/loss=1.80. 36 epochs in 28.0 min, 51s/val-epoch + 41s no-val. Run: `frieren/iter8-chain-lr1.5e-4` (`uu8026u1`). Commit `b51cf84`.
- **Verdict:** Kept — clean -1.20 over iter 7 (45.51). All 4 tracks improved: in_dist -1.08, geom_rc -1.99, geom_cruise -0.67, re_rand -1.07. Trajectory still descending at timeout (last 3 epochs 44.42→44.38→44.31).
- **Notes:**
  - Leaderboard now: nezuko 39.79, thorfinn 39.95, edward 42.27. Our val 44.31 → projected test ~38-39 if our val→test gap is ~5-6 like thorfinn's.
  - **Frieren's submissions are stuck "incomplete" in scores.json** — frieren/{3325cb3, 9d7edad, 682b17d, ..., 2c8f181} all marked incomplete. Format and file sizes match thorfinn's accepted submissions. The scorer must run on a schedule or be triggered by something I'm missing.
  - Continue chaining: iter 9 will halve LR again (7.5e-5) and keep going.

### 2026-04-27 — iter7-globalfilm-chain (iter 7)
- **Hypothesis:** Pure point-wise ResMLP can't reason about per-sample globals (Re, geometry, AoA gestalt) beyond what's repeated at every node. Inserting a zero-init GlobalFiLM after each block — `mean_pool(masked) → MLP → (γ, β) → x*(1+γ)+β` — adds cheap global context. Zero-init makes warm-starting iter 6 a no-op at step 0. Should attack the OOD weakness on geom_rc and re_rand.
- **Change:** New `GlobalFiLM` module (LayerNorm + 2-layer MLP, last layer weight/bias zero-inited). New `use_global` flag wires one FiLM after each ResMLP block; predict.py + train forward pass `mask` so FiLM pools only valid nodes. `train.py` uses `model({"x": x, "mask": mask})`.
- **Result:** Best epoch 34, mean val surf_p MAE = **45.51** (single_in_dist=38.17, geom_rc=**69.45**, geom_cruise=21.63, re_rand=**52.78**). val/loss=1.82. 36 epochs in 28.1 min, 51s/val-epoch + 41s no-val. 16.1GB peak. 12.59M params (+5.3M for FiLM stack vs iter 6's 7.27M). Run: `frieren/iter7-globalfilm-chain` (`amq6vhi5`). Commit `fb168e0`.
- **Verdict:** Kept — **biggest jump since iter 4** (47.47 → 45.51, 4.1% drop). FiLM specifically attacked what we wanted: `geom_rc` -4.86, `re_rand` -2.57; in-distribution tracks barely moved. Confirms the hypothesis that the bottleneck was per-sample global context, not per-point capacity. Trajectory still descending at timeout — chain iter 8 will continue.
- **Notes:**
  - Zero-init verified empirically: `m_global.load_state_dict(m_pure.state_dict(), strict=False)` then forward gives identical output to pure ResMLP.
  - First val (epoch 1) was 51.50 — slightly worse than iter 6's 47.47 because EMA was reinitialised. By epoch 2 it was at 47.54, parity restored.
  - geom_rc dropped from 74.31 → 69.45 over 36 epochs (-0.135/epoch) — consistent monotone descent. Continue chaining for further gains.
  - Possible further wins: deeper FiLM MLP, residual gate, or use the same FiLM mechanism with `[B, S]` slice tokens (Transolver-lite).

### 2026-04-27 — iter5-chain-lr5e-4-sw15 (iter 5)
- **Hypothesis:** Iter 4 was still improving when timeout hit. Warm-start a fresh 30-min run from iter 4's best checkpoint with halved LR (5e-4) and bumped surf_weight (15) to keep pushing surface error down.
- **Change:** `--warm_start /mnt/new-pvc/.../iter4_best_backup.pt --lr 5e-4 --surf_weight 15 --warmup_steps 100 --cosine_epochs 60`. EMA shadow weights are reinitialised from the loaded weights (so they're consistent at start).
- **Result:** Best epoch 58, mean val surf_p MAE = **47.59** (single_in_dist=37.51, geom_rc=74.81, geom_cruise=22.10, re_rand=55.95). val/loss=1.45. 58 epochs in 28.1 min. Run: `frieren/iter5-chain-lr5e-4-sw15` (`bf5ustqc`). Commit `eb54b3a`.
- **Verdict:** Kept — improved on iter 4 by 4.8% (50.01 → 47.59). All 4 split MAEs improved; biggest drops on `single_in_dist` (-4.8) and `geom_cruise` (-2.0). geom_rc and re_rand barely moved (-1.6 and -1.3). The metric ceiling on these OOD tracks is set by something other than training time — likely architecture / capacity for global reasoning.
- **Notes / next ideas:**
  - **Diminishing chains.** Each chain is ~5% — more chains won't catch thorfinn (40.68).
  - **OOD tracks dominate loss budget.** geom_rc 74.81 + re_rand 55.95 = 130.8 of total 210.4 (62%). Have to attack those directly.
  - Iter 6 plan: chain again with **larger subsample (80k vol)** to give each gradient step more signal — 250k samples vs 160k now. ~44s/step bench (vs 33s). Should still fit ~30 epochs in 28 min.

### 2026-04-27 — resmlp-h384-l1-pwt3-ema-msf (iter 4)
- **Hypothesis:** Apply Edward+alphonse's proven recipe — L1 loss with channel weight `[1, 1, 3]` (boosts pressure), EMA shadow weights with decay 0.9995, multi-scale Fourier features (8 scales, max_freq 16). Roll back to iter 2's fast ResMLP (h=384 b=6 bf16) since Transolver was too slow. Aim: beat alphonse (50.70).
- **Change:** New `MultiScaleFourier` class on positions; new `channel_loss(...)` with L1+p_weight=3; EMA shadow `state_dict` updated post-step (decay 0.9995, start 500 steps); validation and final ckpt saved using EMA weights; `cfg.warm_start` arg for chaining; `arch="resmlp"` default.
- **Result:** Best epoch 58, mean val surf_p MAE = **50.01** (single_in_dist=42.29, geom_rc=76.42, geom_cruise=24.05, re_rand=57.27). val/loss=1.16. 58 epochs in 28.1 min, ~33s/val-epoch + 25s/no-val. 10.1GB peak. Run: `frieren/resmlp-h384-l1-pwt3-ema-msf` (`w0hstbct`). Auto-predict OK. Commit `3b1371c`.
- **Verdict:** Kept — **biggest single jump yet** (68.58 → 50.01, 27% drop). Leaderboard: thorfinn 40.68, edward 43.73, alphonse 50.70 — projected rank ~3-4 once scored.
- **Notes:**
  - **L1 loss + p_weight=3 alone deliver most of the gain.** Aligns optimisation directly with leaderboard metric.
  - **EMA val curve is very clean** — every val epoch was a new best, monotone decrease 91.68 → 50.01. No noise. EMA's noise-cancelling effect.
  - Track gap analysis vs thorfinn: we beat thorfinn on `geom_cruise` (24.05 vs 22.69 — close) and rival on `single_in_dist` (42.29 vs 46.33). We lose badly on **`geom_rc` (76.42 vs 56.54)** and **`re_rand` (57.27 vs 37.18)** — OOD generalisation is our weak spot.
  - Model was still improving at epoch 58 (cosine_epochs=60). Chain iter 5 from this checkpoint.
  - geom_rc and re_rand suggest the network memorises the training distribution well but extrapolates poorly to new camber / new Re. Possible fixes: data aug (Re jitter, mirror-flip), more global communication (Transolver with **masking fix** + smaller LR), or a Re-conditioned FiLM layer.

### 2026-04-27 — transolver-h320-n8-bf16 (iter 3) — REGRESSION
- **Hypothesis:** Slice attention adds global communication that should help geom_rc (the worst track in iter 2 at 92.63). Bring in TransolverNet class (h=320, n_layers=8, heads=8, dim_head=40, slice_num=64, mlp_ratio=2). Bump cosine_epochs to 80 so LR doesn't decay to 0 mid-run.
- **Change:** Added `SliceAttention`, `TransolverBlock`, `TransolverNet` to `train.py`; new `arch="transolver"` config. Predict.py dispatches arch from config.yaml. Same loss/optimiser as iter 2.
- **Result:** Best epoch 20, mean val surf_p MAE = **108.15** (single_in_dist=125.25, geom_rc=119.77, geom_cruise=85.97, re_rand=101.61). Run: `frieren/transolver-h320-n8-bf16` (`<see W&B>`). Only 21 epochs in 28.9 min — attention blocks ~2.4× slower per epoch (92s vs 38s for iter 2).
- **Verdict:** Worse than iter 2 (108 vs 68). Compute budget didn't allow attention to amortise its cost. Not committing this checkpoint; keeping iter 2's. Code change kept (TransolverNet stays as a class for future warm-starts).
- **Notes / failure modes:**
  - **Bug confirmed in slice attention:** padded tokens still receive non-zero `slice_weights` and inflate `slice_norm`, diluting slice tokens. Edward's code masks slice_weights against `pad_mask`. Need to add this for any future Transolver run.
  - **Loss/metric mismatch:** training with MSE while leaderboard scores MAE. Edward+alphonse use **L1 loss with p_channel_weight=3** — directly aligns optimisation with the metric.
  - **No EMA.** Both alphonse and edward note val noise; EMA decay ≈ 0.9995 should give ~3-5% surf_p drop nearly free.
  - Cosine_epochs=80 over 21 epochs means LR mult was still ~0.93 at end — schedule fine, but compute cap binding.

### 2026-04-27 — resmlp-h384-b6-bf16-skipval (iter 2)
- **Hypothesis:** Iter 1 was undertrained (only 8 epochs in 30 min). Reduce model to h=384 b=6, enable bf16 autocast, validate every other epoch, increase LR slightly. Should yield ~3-4× more gradient steps.
- **Change:** Reduce hidden=384 / n_blocks=6 (~7.3M params), lr=1.5e-3, `torch.autocast(bfloat16)` in train + val, `val_every=2`, `cosine_epochs=30` to anneal cosine LR by epoch 30. Free GPU mem before auto-predict subprocess to fix iter-1 OOM.
- **Result:** Best epoch 30, mean val surf_p MAE = **68.58** (single_in_dist=67.26, geom_rc=92.63, geom_cruise=42.68, re_rand=71.74). val/loss=2.45. 50 epochs in 28.3 min, ~38s/val-epoch + 30s/no-val-epoch. 12.4GB peak VRAM. Run: `frieren/resmlp-h384-b6-bf16-skipval` (`bfh13z32`). Auto-predict succeeded this time. Commit `8e0978c`.
- **Verdict:** Kept — major improvement (1.81×), and would have ranked between alphonse (50.83) and askeladd (77.66) on apr27-4 leaderboard. Edward leads at 43.75. Still well off the front.
- **Notes / failure modes:**
  - **LR collapse at epoch 30** — `cosine_epochs=30` made the LR multiplier hit 0 exactly there, so epochs 32-50 had LR=0 and weights frozen. Validation outputs were *bit-identical* across epochs 32, 34, ..., 50 (proving the freeze). Wasted ~12 min of compute. Fix in iter 3: `cosine_epochs=80` so LR never bottoms.
  - **geom_rc still the worst track at 92.63**, almost 2× geom_cruise (42.68) — point-wise ResMLP can't model unseen camber geometry well. Need global communication (slice attention or kNN aggregation).
  - val curve was monotonically improving through epoch 30 — model was still learning when LR died. Strongly suggests both more epochs and more capacity will help.

### 2026-04-27 — resmlp-h512-b8-sw20-fourier (iter 1)
- **Hypothesis:** Replace baseline Transolver with proven frieren mar27 recipe — ResMLP with hidden=512, 8 blocks, lr=1e-3, surf_weight=20 (bumped from 10 since leaderboard ranks by surface pressure MAE), Fourier features on 2D position for high-freq capacity. Subsample volumes to 60k/sample to fit more iters in 30 min.
- **Change:** Rewrote `train.py` & `predict.py`. New `ResMLP` model (~17M params, hidden=512, n_blocks=8, expansion=4, n_freqs=32, fourier_sigma=4). Added subsample helper, warmup+cosine LR, grad clip 1.0, checkpoint by mean surface-pressure MAE. Wrapped main loop in `if __name__ == "__main__":` so predict.py can `from train import ResMLP` without firing the CLI parser.
- **Result:** Best epoch 6, mean val surf_p MAE = **123.87** (single_in_dist=146.14, geom_rc=144.04, geom_cruise=90.87, re_rand=114.44). val/loss=5.52. 8 epochs total in 30.7 min, ~231s/epoch. 45GB peak VRAM. Run: `frieren/resmlp-h512-b8-sw20-fourier` (`ieozo3vf`).
- **Verdict:** Kept — first submission to apr27-4 leaderboard; commit `98b9503`. But **way behind** the apr27-bis baseline (thorfinn @ 45.94) and apr27-4 alphonse @ 51.66. Frieren's mar27 baseline got 33–55 MAE *with chained training*; from-scratch in 30 min stalls at ~125.
- **Notes / next ideas:**
  - **Auto-predict failed (OOM)** — train.py's subprocess to predict.py held both copies of the 17M model on a 96GB GPU; the train process alone retained 90GB at termination because Python hadn't released the buffers. Need to `del model; torch.cuda.empty_cache(); gc.collect()` before subprocess, or just run predict separately. (For now ran predict separately, fine.)
  - **Convergence is the bottleneck.** 8 epochs is too few for a 17M model from scratch. Options to try next: (a) bf16/fp16 mixed precision for ~2× speedup, (b) smaller model (hidden=256, 6 blocks ≈ 4M params) to do 25+ epochs, (c) skip-validation-every-other-epoch, (d) larger batch + smaller subsample.
  - **Pure ResMLP is point-wise** — every node is processed in isolation. The geometry context is encoded in dsdf/saf input features, but global flow features (e.g. wake interactions) can't be modelled. Adding a small set-attention block or grid-pool token could help, especially for tandem-foil splits.
  - val MAE ordering: geom_cruise (90) < re_rand (114) < geom_rc (144) ≈ in_dist (146). So in-distribution is currently *worse* than cruise OOD — model is severely undertrained.
