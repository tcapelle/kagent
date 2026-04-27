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

### 2026-04-27 — iter11: rc_single boost 5x, lr=3e-6, ema=0.9995 (kept)
- **Hypothesis:** Iter10 confirmed the boost lever is monotone; one more notch (5x) plus ultra-slow EMA finishing should still push.
- **Change:** train.py — `rc_single_boost` 4→5, lr 5e-6→3e-6, ema_decay 0.999→0.9995.
- **Result:** Best epoch 8 with **val avg_surf_p=45.37** (vs iter10 45.56 → -0.19). Smooth descent (45.55, 45.54, 45.51, 45.48, 45.44, 45.41, 45.39, 45.37). val_single_in_dist 2.6779→2.6513 — biggest single-split delta of any iteration so far. Wall time 25.7 min. WandB run cdrsns9k.
- **Verdict:** Kept. Curve still sloping.
- **Notes:**
  - Per-iter improvement is now ~0.2 val. Worth one more boost notch (iter12) before pivoting; beyond that the sampler is essentially showing only single-foil samples and we'd start regressing on tandem.

### 2026-04-27 — iter10: stronger rc_single boost (4x), lr=5e-6 (kept)
- **Hypothesis:** Iter9's 2.5x boost moved single_in_dist (45.40→44.14 test). Going to 4x should claw back more of the gap to nezuko (32.29).
- **Change:** train.py — `rc_single_boost` 2.5→4, lr 6e-6→5e-6. Also lowered predict.py default `batch_size` 4→1 to stop the auto-predict from OOMing on cruise meshes.
- **Result:** Best epoch 8 with **val avg_surf_p=45.56** (vs iter9 45.93 → -0.37). Trajectory: 45.90, 45.87, 45.79, 45.74, 45.70, 45.64, 45.59, 45.56. val_single_in_dist: 2.7090→2.6779. Wall time 27.1 min (faster again because more single-foil samples). WandB run uz9red8i.
- **Verdict:** Kept. Boost-then-polish loop is paying off.
- **Notes:**
  - Just as the leaderboard updated I jumped to **#1 at 39.52** test (iter9 commit `6fb87ef`), edging askeladd's 39.55 by 0.03. iter10 scored TBD.
  - Per-split wins on iter9 vs iter8: single_in_dist 45.40→44.14 (the deliberate target, ✓), rc 55.47→55.25, re_rand 36.58→36.47, cruise unchanged at 22.21. The boost lever is clean: it moves the targeted split without trashing the others.
  - Auto-predict OOM'd again with batch_size=4 on cruise (had to manually re-run with bs=1). Defaulting predict.py to bs=1 going forward.

### 2026-04-27 — iter9: boost racecar_single 2.5x in sampler (kept)
- **Hypothesis:** From the leaderboard split breakdown, my model is far behind on `test_single_in_dist` (45.4 vs nezuko's 32.3). The balanced sampler in `data.py` already gives equal weight to each domain, but the model has been over-fit to tandem regimes through the warm-start chain. Up-weighting racecar_single 2.5x in the sampler (on top of the existing 1/group_size weights) should claw back some of that gap.
- **Change:** train.py — added `rc_single_boost=2.5` cfg, multiplies sample_weights for the 599 racecar_single training indices read from `meta.json` after `load_data()`. Bumped lr 4e-6→6e-6 (slightly more energy to actually move on the new sampler distribution), ema_decay 0.9995→0.999 (faster adaptation).
- **Result:** Best epoch 8 with **val avg_surf_p=45.93** (vs iter8 46.36 → -0.43). Trajectory: 46.26, 46.19, 46.12, 46.08, 45.99, 45.95, 45.93, 45.93. Per-split losses confirm the boost helps: `val_single_in_dist` went 2.7293→2.7090 in normalized loss; cruise/re_rand stayed flat. Wall time 29.3 min (faster epochs because RC-single meshes are smaller). WandB run ywetcbgu. Predictions at commit `6fb87ef`.
- **Verdict:** Kept. Re-balancing the sampler is a more targeted lever than blind LR-tuning.
- **Notes:**
  - Leaderboard at this iteration: askeladd 39.55 (#1), nezuko 39.79 (#2), thorfinn 39.91 (#3 with iter8). My iter9 hasn't been scored yet — projected test ≈39.5 if val→test gap holds.
  - The val/test gap is split-dependent. My iter7 single_in_dist test=45.56 with val_single_in_dist surface_loss=2.74. Iter9 val_single_in_dist=2.71, so the test improvement should be small (<1pt) on that split. Need a bigger structural change to claw back the full gap to nezuko's 32.

### 2026-04-27 — iter8: continue iter7, lr=4e-6, ema_decay=0.9995 (kept)
- **Hypothesis:** With a full-EMA pipeline and another LR halving, the late-training oscillations get further smoothed and we squeeze out another fractional gain. Treat this as the "final polish" iteration.
- **Change:** train.py — lr 8e-6→4e-6, ema_decay 0.999→0.9995 (slower decay = even smoother averaging). Warm-start from iter7 ckpt.
- **Result:** Best epoch 8 with **val avg_surf_p=46.36** (vs iter7 46.55 → -0.19). Trajectory perfectly monotonic with no oscillation: 46.56, 46.55, 46.53, 46.48, 46.44, 46.41, 46.38, 46.36. WandB run vhp3g8od. Predictions copied to commit `b0beb9d`.
- **Verdict:** Kept. Diminishing returns: improvement-per-epoch is now ~0.03 val. Beyond this we are essentially polishing residual variance.
- **Notes:**
  - **Leaderboard pivot:** between iter7 and iter8 nezuko overtook me at 39.79 test (vs my 39.95). Their split breakdown is very different from mine — they crush `single_in_dist` (32.29 vs my 45.56) and `geom_camber_rc` is closer, but I crush `re_rand` (36.64 vs their 47.69). Their model is good at the easy single-foil split that I'm worst at.
  - This suggests I am over-fit to tandem regimes (because my warm-start chain came from a tandem-heavy loss balance), and there's room to claw back ~10 points on single_in_dist by improving on that domain specifically. Likely strategies for iter9: (a) up-weight single-foil samples in the WeightedRandomSampler, (b) explicit per-domain loss tracking, (c) inspect what nezuko's predictions look like to understand the gap.

### 2026-04-27 — iter7: EMA + ultra-low LR (kept)
- **Hypothesis:** Stochastic weight averaging via an EMA of the live weights tends to give a free 0.2-0.5 point improvement; combined with an ultra-low LR (8e-6) on top of iter6, the EMA averages over the late-training oscillations the live model has been wobbling through.
- **Change:** train.py — added `ema_decay=0.999` cfg, maintained an EMA shadow updated every step, validate (and save) using EMA weights. Set lr=8e-6. Warm-start from `checkpoints/best.pt` (iter6 ckpt).
- **Result:** Best epoch 8 with **val avg_surf_p=46.55** (vs iter6 46.84 → -0.29). Steady monotonic descent with no oscillation: ep1=46.81, ep2=46.75, ep3=46.74, ep4=46.70, ep5=46.65, ep6=46.61, ep7=46.59, ep8=46.55. Wall time 32.2 min. WandB run 70qaist2.
- **Verdict:** Kept. EMA is doing exactly what it says on the tin: silky-smooth descent and fractional gain.
- **Notes:**
  - **Auto-submit gotcha:** I committed the iter6 journal *after* iter7 training started. When the auto-predict subprocess ran `git rev-parse --short HEAD` at the end, it picked up the journal commit hash (`f00f383`), not the iter7 code commit. Predictions were saved under the wrong commit dir. Worked around by copying the predictions dir to the new ckpt-commit hash (`4e33c37`). Lesson: don't commit anything during a running training; or have predict.py accept an explicit commit override.
  - The descent is now ~0.05 per epoch on val. Diminishing returns are real; further fine-tuning at this LR will struggle to shave another 0.5 point. To push significantly lower we likely need a structurally different model (bigger, or trained on a different objective entirely).

### 2026-04-27 — iter6: continue iter5 ckpt at lr=1e-5 (kept)
- **Hypothesis:** Iter5's curve was still descending. Drop LR another factor of two (to 1e-5) and continue from `checkpoints/best.pt` (iter5 ckpt at 47.13). Same loss as iter5.
- **Change:** train.py — `lr` 2e-5→1e-5; everything else identical to iter5.
- **Result:** Best epoch 8 with **val avg_surf_p=46.84** (vs iter5 47.13 → -0.29). Trajectory: ep1=47.64, ep2=47.99, ep3=47.72, ep4=47.00, ep5=47.07, ep6=46.88, ep7=46.86, ep8=46.84. Wall time 32.4 min. WandB run vz708x3g. Predictions at `/mnt/new-pvc/predictions/apr27-4/thorfinn/a9065d5/`.
- **Verdict:** Kept. Marginal but real continued descent.
- **Notes:**
  - Auto-predict OOM'd on the cruise split at batch_size=4 (cruise meshes are ~208K nodes); re-ran predict.py manually with `--batch_size 1`. Should consider lowering predict.py default batch to 1 to avoid this in future.
  - **As of this iteration, thorfinn is leading the apr27-4 leaderboard at 40.68 test avg_surf_p (commit 09570d5 = iter5)**, with edward at 43.24 and nezuko at 42.58.
  - Tested weight averaging across iter2/iter4/iter5 ckpts — all averaging hurts because the checkpoints are highly correlated (each warm-started from the previous). Tested prediction ensemble of iter4+iter5+iter6 — also hurts vs iter6 alone (47.16 vs 47.13). Diversity is too low; ensembling isn't going to help unless we get a structurally different model.

### 2026-04-27 — iter5: gentle p-channel weight + lr=2e-5 (kept)
- **Hypothesis:** Iter3 had the right idea (up-weight pressure to match the leaderboard's pressure-only ranking) but p_weight=5 was too aggressive. A gentler p_weight=1.5 plus an even lower LR (2e-5) on top of iter4 should nudge the model toward pressure without destabilizing.
- **Change:** train.py — re-introduced `surf_p_weight=1.5` (per-channel weights `[1, 1, 1.5]` on the surface huber); lr 3e-5→2e-5; warm-start from `checkpoints/best.pt` (iter4 ckpt at 48.02).
- **Result:** Best epoch 8 with **val avg_surf_p=47.13** (vs iter4 48.02 → -0.89). Trajectory: ep1=49.42, ep2=49.10, ep3=48.60, ep4=47.94, ep5=47.92, ep6=47.31, ep7=47.28, ep8=47.13. Wall time 32.1 min. WandB run hr2zpk4f. Predictions at `/mnt/new-pvc/predictions/apr27-4/thorfinn/09570d5/`.
- **Verdict:** Kept. Gentle pressure-channel reweighting works, validating the iter3 hypothesis with a saner magnitude.
- **Notes:**
  - Even at p_weight=1.5 the warm-start drifts in epochs 1-3 (49.42 → 48.60) before crossing back below the 48.02 baseline at epoch 4. The early "re-converge" tax keeps showing up.
  - The current leaderboard leader (edward) is at 43.73 test. Prior thorfinn val→test gap was ~7 (49.78 val → 42.90 test), so 47.13 val could plausibly score test ~40 — competitive.
  - Diminishing returns: each iteration shaves ~0.5-1 point of val avg_surf_p. Likely need a structural change (larger model / weight averaging / TTA) to break much further.

### 2026-04-27 — iter4: continue iter2 ckpt at lr=3e-5 (kept)
- **Hypothesis:** Iter2's curve was still descending at epoch 8. Continue training from the iter2 checkpoint at a 3x lower LR (3e-5 vs 1e-4), no other loss changes, would shave more avg_surf_p before saturating.
- **Change:** train.py — `init_checkpoint = "checkpoints/best.pt"` (the iter2 ckpt), lr 1e-4→3e-5; also persist warm-start as initial ckpt at start of training so auto-predict always has a valid file even if no epoch beats the warm-start.
- **Result:** Best epoch 8 with **val avg_surf_p=48.02** (vs iter2 48.61 → -0.59). Trajectory: ep1=49.83, ep2=49.77, ep3=49.48, ep4=50.43 (wobble), ep5=49.02, ep6=48.51, ep7=48.20, ep8=48.02. Wall time 32.1 min. WandB run 3inh5kj2. Predictions at `/mnt/new-pvc/predictions/apr27-4/thorfinn/664daac/`.
- **Verdict:** Kept. Marginal but real improvement; checkpoint promoted to `checkpoints/best.pt`.
- **Notes:** Even at lr=3e-5 the warm-start still drifts in early epochs (49.83 ep1 > 48.61 baseline) before recovering. The cosine decay is what makes the late epochs compound; the early epochs are mostly wasted "re-converging." For iter5, consider an even gentler approach: warmup (very low LR) then peak.

### 2026-04-27 — iter3: per-channel surf weight on pressure (aborted)
- **Hypothesis:** Leaderboard ranks by surface pressure MAE only. Up-weighting the pressure channel 5x in the surface loss should trade velocity accuracy for pressure accuracy and lower avg_surf_p.
- **Change:** train.py — added `surf_p_weight=5.0` cfg, multiplied huber_err by `[1, 1, 5]` per-channel weights in surface loss; warm-start from iter2 ckpt at lr=5e-5.
- **Result:** ep1=50.85, ep2=52.68 (regressed!), ep3=50.88. Killed at epoch 3 — diverging from iter2's 48.61, no signs of crossover.
- **Verdict:** Discarded. p_weight=5 is too aggressive: the warm-started weights were already a balanced optimum, and 5x reweight pushed the model far enough to require many epochs to recover, which we don't have.
- **Notes:** A gentler p_weight (1.5–2.0) might still be net positive but the budget didn't allow another bake-off.

### 2026-04-27 — iter2: warm-start from sae8usmw + Huber surf loss (kept)
- **Hypothesis:** Warm-start from the prior best thorfinn checkpoint (`apr27/model-sae8usmw`, 49.78 val avg_surf_p) and fine-tune with lower LR (1e-4), Huber surface loss, T_max=8 cosine schedule. Goal: drive val avg_surf_p below 49.78 within the 30-min budget.
- **Change:** train.py — added `init_checkpoint` cfg field that defaults to sae8usmw, switched surface loss to SmoothL1 (Huber, beta=1), surf_weight=15, lr=1e-4, epochs=8, T_max=8, ckpt selected by avg_surf_p (not combined val/loss), grad clip 1.0, `__main__` guard so predict.py can import. predict.py — load Transolver via `from train import Transolver` and read config.yaml from checkpoint dir.
- **Result:** Best epoch 8 with **val avg_surf_p=48.61** (vs warm-start baseline 49.78 → -1.17). Trajectory: ep1=61.43, ep2=59.24, ep3=56.54, ep4=54.49, ep5=52.47, ep6=50.94, ep7=49.13, ep8=48.61. Total wall time 32.1 min. VRAM peak 74GB. WandB run: m7c1v7qx. Predictions auto-submitted to `/mnt/new-pvc/predictions/apr27-4/thorfinn/269b348/`.
- **Verdict:** Kept. Beat the prior best on val. Test leaderboard score TBD.
- **Notes:**
  - Discovered most of the apr27 thorfinn checkpoints in the PVC are *bad* mid-training snapshots. Built `/tmp/eval_ckpt.py` to evaluate them: `model-sae8usmw` and `model-fxwfocit` are tied at 49.78; `model-eue2l8uf` is 372 (corrupted/early). Always re-evaluate warm-start candidates before trusting them.
  - Switching MSE→Huber on surface caused a regression in epoch 1 (61.43 vs warm-start 49.78); the model needs ~7 epochs to re-converge below the warm-start. If the run had been timeout-killed earlier we'd have shipped a worse model than the one we warm-started from. Lesson: evaluate warm-start at epoch 0 and gate ckpt-saving on beating that.
  - Improvement is monotonic and mostly comes from `val_geom_camber_rc` (2.69→1.66 split loss) and `val_single_in_dist` (3.90→2.66). Cruise barely changes (0.41→0.34).
  - Next: try harder loss alignment (pure L1 or per-channel surf weighting that emphasizes p), and potentially go bigger or use TTA.

### 2026-04-27 — iter1: Transolver-192/6 + SmoothL1 surf loss (aborted)
- **Hypothesis:** Restoring the proven thorfinn-apr27 architecture (n_hidden=192, n_layers=6, n_head=6, slice=64) plus SmoothL1 (Huber) on surface (better aligned with leaderboard MAE) and surf_weight=15 would beat the prior best of 42.90 avg_surf_p.
- **Change:** train.py — switched to Huber loss for surface component, surf_weight 10→15, added grad clip 1.0, ckpt selected by avg_surf_p, added `__main__` guard. predict.py — load Transolver via import + read config.yaml from checkpoint dir.
- **Result:** Killed at epoch 3/50 after ~12 min. Trajectory: ep1=206, ep2=189, ep3=173 avg_surf_p. ~243s/epoch ⇒ only ~7 epochs in 30 min budget. Cosine T_max=50 means LR barely decays — model trained at near-constant 5e-4 throughout. Far from converging to ~42 baseline. VRAM peak 74GB.
- **Verdict:** Discarded. Cold-start from scratch in 7 epochs cannot match a prior thorfinn checkpoint that already exists at 42.90 avg_surf_p. Wrong starting point; need warm-start.
- **Notes:** Confirmed prior thorfinn checkpoint at `/mnt/new-pvc/kagent/apr27/thorfinn/checkpoints/model-eue2l8uf/checkpoint.pt` loads cleanly into the same architecture (1.71M params). Next: warm-start iter2 from that checkpoint with reduced LR and faster cosine decay.
