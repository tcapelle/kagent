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

### 2026-04-28 — iter19: fullmesh fine-tune at lr=5e-6
- **Hypothesis:** Iter18 only had one good epoch — lr too high for the fully-converged checkpoint. Drop lr to 5e-6 and run shorter cycle.
- **Change:** lr 1e-5→5e-6, epochs 14→10. Initial = iter18 (model-lhl3kzpw).
- **Result:** Trained 9 epochs in 32.6 min. Best epoch 4, val/loss=1.4593 (vs iter18 1.4767, vs iter17 1.5368). Val mae_surf_p avg=41.64. Auto-submit predict.py OOM'd; ran manually after process exit. Predictions at commit c6b824f. Test pending.
- **Verdict:** kept (best val/loss yet, expect rank 4 hold or improvement).
- **Notes:** Test-time fragmentation may differ from training, need to wait for scoring.

### 2026-04-28 — iter18: full mesh + batch=2 chain
- **Hypothesis:** Eliminate train/test mesh-size mismatch by training on full meshes (no subsample).
- **Change:** train_max_volume 80k→0, batch 4→2, lr 1.5e-5→1e-5, epochs 16→14. Initial = iter17 (model-zvhh3d5w).
- **Result:** Trained 9 epochs in 32.7 min (220s/epoch). Best at epoch 1, val/loss=1.4767. Val mae_surf_p avg=42.53. Predictions auto-submitted to commit e2d1721 (not yet scored).
- **Verdict:** TBD — best val/loss yet, but only one good epoch (the training drifted afterwards). Score pending.
- **Notes:** Lr=1e-5 may still be too aggressive after a long warm-start chain. Iter19: lower lr further or stay at iter17/16 setup.

### 2026-04-28 — iter17: chain at vol=80k
- **Hypothesis:** Continue increasing train subsample for tighter train/test match.
- **Change:** train_max_volume 50k→80k, lr 2e-5→1.5e-5, epochs 22→16. Initial = iter16 (model-3fsnww54).
- **Result:** Trained 16 epochs in 30.5 min (114s/epoch). Best epoch 11, val/loss=1.5368 (slightly worse than iter16's 1.5162). Val mae_surf_p avg=43.79. **Test 40.57 (commit ff7547b) — rank 4 (best yet)**.
- **Verdict:** kept (modest test improvement 40.89→40.57). Diminishing return — val/loss got worse but test improved slightly.
- **Notes:** Iter18 plan: try vol=100k or full mesh; or pivot to a stronger lever.

### 2026-04-28 — iter16: bigger subsample (50k) for train/test alignment — **RANK 4!**
- **Hypothesis:** Train/test distribution mismatch (training on 25k vol points but testing on 100k+) is leaving accuracy on the table. Doubling the train subsample to 50k should reduce that gap.
- **Change:** train_max_volume 25000→50000. lr 3e-5→2e-5. epochs 40→22 (compute budget).
- **Result:** Trained 22 epochs in 28.8 min (79s/epoch — 1.6× slower as expected). VRAM 18.7GB. Best epoch 10, val/loss=1.5162. Val mae_surf_p: single=36.5, geom_rc=59.6, geom_cruise=31.7, re_rand=47.2 (avg=43.75). **Test 40.89 — rank 4 (passed edward 41.11)**. Commit e49f4af.
- **Verdict:** kept (clear win — 1.6 points off test, jumped one rank).
- **Notes:** Bigger subsample wins. Try iter17 with full mesh (no subsample) at small batch + warm-start; or iter17 with 50k vol but more epochs.

### 2026-04-28 — iter15: longer cycle (40 epochs at lr=3e-5)
- **Hypothesis:** Stretch the cosine cycle to 40 epochs to extract more from each chain step.
- **Change:** epochs 30→40. Initial = iter14 (model-6t65htqq), lr=3e-5.
- **Result:** Trained 38/40 epochs in 30.2 min. Best epoch 13, val/loss=1.5494 (vs iter14 1.6053). Val mae_surf_p avg=45.75. **Test 42.49 (commit 4d83214) — rank 5 best yet for me.**
- **Verdict:** kept (small improvement: 42.64 → 42.49 on test).
- **Notes:** Training has flatlined after epoch 13; longer schedule didn't help much. Chain is in deep diminishing returns. Need a step-change approach: bigger model from scratch with KD, or test-time augmentation.

### 2026-04-28 — iter14: chain warm-start from iter13 (plateauing)
- **Hypothesis:** Continue chain at lr=3e-5.
- **Change:** lr 5e-5→3e-5. Initial = iter13 (model-04j8mo2u).
- **Result:** Best epoch 13, val/loss=1.6053 (vs iter13 1.6064 — within noise). Val mae_surf_p avg=45.95 (vs 46.66). Predictions auto-submitted to commit dab91be (not yet scored at last refresh).
- **Verdict:** kept (marginal improvement). The chain has plateaued near val/loss=1.6.
- **Notes:** Each cycle is now barely moving the needle. Need a different lever: bigger model from scratch, augmentation, or knowledge distillation. Consider iter15 with longer cycle (epochs=40) before pivoting.

### 2026-04-27 — iter13: chain warm-start from iter12 (rank 5!)
- **Hypothesis:** Continue chain at lr=5e-5.
- **Change:** lr 7e-5→5e-5. Initial = iter12 (model-2qpz3rzk).
- **Result:** Best epoch 23, val/loss=1.61. Val mae_surf_p: single=43.0, geom_rc=61.1, geom_cruise=33.6, re_rand=49.0 (avg=46.7). **Test leaderboard 42.64 — RANK 5 (was rank 6 at 59.94)**. Best on geom_rc split among all agents (52.90). W&B run 04j8mo2u. Commit c5216b8.
- **Verdict:** kept (huge win — passed alphonse, ~3 from top-4 rank).
- **Notes:** Warm-start chain converging slowly. Each cycle drops val ~3-4%. Plan iter14 chain at lr=3e-5.

### 2026-04-27 — iter12: chain warm-start from iter11
- **Hypothesis:** Continue the warm-start chain at lr=7e-5 (down from iter11 lr=1e-4) for finer convergence.
- **Change:** train.py — lr 1e-4→7e-5. Initial checkpoint = iter11 (model-qfi1mn6t).
- **Result:** Trained 30 epochs in 23.9 min. Best epoch 13, val/loss=1.75. Val mae_surf_p: single=45.35, geom_rc=65.20, geom_cruise=35.28, re_rand=52.98 (avg=49.70, vs iter11 51.84). W&B run 2qpz3rzk. Commit 8c687e1.
- **Verdict:** kept (best yet). Below alphonse-test (49.19) on val — expect rank 4-5 on test.
- **Notes:** Diminishing returns: iter5→iter10 -16%, iter10→iter11 -5.3%, iter11→iter12 -4.1%. Still going. Plan iter13 chain at lr=5e-5.

### 2026-04-27 — iter11: chain warm-start from iter10
- **Hypothesis:** Warm-start cycles work — repeat the recipe with iter10 as starting point and even lower lr=1e-4 to keep refining.
- **Change:** train.py — lr 1.5e-4→1e-4. Initial checkpoint = iter10 (model-hqd0w255). Same epochs=30 cosine schedule.
- **Result:** Trained 30 epochs in 23.9 min. Best epoch 19 (val/loss=1.89), then later epochs hovered around 2.0. Val mae_surf_p: single=49.1, geom_rc=68.2, geom_cruise=35.7, re_rand=54.4 (avg=51.8, vs iter10 54.7). W&B run qfi1mn6t. Predictions at commit 597d79e.
- **Verdict:** kept. Best val yet by a wide margin.
- **Notes:** Warm-start chains (iter5 → iter10 → iter11) consistently push val/loss down 15-25% per cycle. Each cycle uses a fresh cosine schedule starting at lower lr. Plan iter12: chain again from iter11 with lr=7e-5.

### 2026-04-27 — iter10: warm-start from iter5, low-LR fine-tune
- **Hypothesis:** Iter5's training loss was still falling at the 30-min cutoff. Loading its weights and continuing for another 30 min with cosine schedule restarted at lr=1.5e-4 should let optimization push past iter5's plateau.
- **Change:** train.py — added `--init_ckpt` to load weights at start, lr=5e-4→1.5e-4, epochs=38→30. Reverted iter8's slice/mlp tweaks back to iter5 model_config (must match the checkpoint).
- **Result:** Trained 30 epochs in 23.9 min. Best epoch 30 (final), val/loss=2.41 (vs iter5 2.90, -17%). Val mae_surf_p: single=51.9, geom_rc=71.8, geom_cruise=37.7, re_rand=57.2 (avg=54.7, vs iter5 65.0). W&B run hqd0w255. Predictions auto-submitted to commit 839aa0d.
- **Verdict:** kept (best yet — should beat iter5 on leaderboard).
- **Notes:** The very first epoch after warm-start gave an LR-shock (val regressed to 2.88 then bounced back), but cosine decay through 30 epochs ended at a clean optimum. Warm-start cycles are a strong lever — could iterate (warm-start iter10 → iter11). Still falling at epoch 30, so another cycle should help.

### 2026-04-27 — iter9: ensemble iter5 + iter8 (commit 2516cbc)
- **Hypothesis:** Iter8 (val 67.4) is closer to iter5 (val 65.0) than iter3 was, so an iter5+iter8 ensemble should work better than the iter3+iter5 attempt.
- **Change:** Reused predict_ensemble.py with iter5 (xqti3sr8) + iter8 (zcy58guy) checkpoints. Submitted at commit 2516cbc (created via trivial journal commit since the ensemble overwrote the iter8 commit dir).
- **Result:** Awaiting leaderboard scoring. Iter5 alone: 59.94. Iter8 alone: 62.43. Naive mean: 61.19.
- **Verdict:** TBD on score; iter10 has likely surpassed both anyway.
- **Notes:** Ensemble is still a free additional submission to fall back on.

### 2026-04-27 — iter8: complementary arch (slice=64, mlp_ratio=4) for ensemble
- **Hypothesis:** Train a high-quality model with slightly different inductive bias (smaller slice_num, larger mlp) for ensembling with iter5.
- **Change:** train.py model_config — slice_num 96→64, mlp_ratio 2→4. Same epochs/lr as iter5.
- **Result:** Trained 38 epochs in 25.8 min (41s/epoch — faster than iter5 due to smaller slice). Best epoch 32, val/loss=3.14. Val mae_surf_p avg=67.4. **Test 62.43 (commit 8bb8cae)**. W&B run zcy58guy.
- **Verdict:** kept as ensemble component; weaker than iter5 standalone.
- **Notes:** Didn't beat iter5 on its own; useful only for ensembling.

### 2026-04-27 — iter7: ensemble iter3 + iter5 predictions (regression)
- **Hypothesis:** Averaging two trained checkpoints in physical space should reduce error vs either alone (free ensemble win).
- **Change:** Added `predict_ensemble.py`. Averaged iter3 (model-2ypzrvpq, val 65.0) and iter5 (model-xqti3sr8, val 65.0… wait, iter5 val 65.0, iter3 val 77.4) predictions, applied no-slip post-hoc.
- **Result:** Test avg_surf_p = 61.14 (vs iter5 alone 59.94). Per split: single=56.61, geom_rc=73.81, geom_cruise=41.59, re_rand=72.56 (commit 95b634c on PVC).
- **Verdict:** **discarded** as winning submission — iter5 (b5078b3, 59.94) stays best. The weaker iter3 dragged the average up.
- **Notes:** Ensemble *did* beat naive mean-of-MAEs (64.76), confirming error decorrelation, but the quality gap between iter3 and iter5 was too big. Need ensemble of similarly-strong models. Plan iter8: rerun iter5-style training with slightly different hyperparams to get a complementary high-quality checkpoint, then re-ensemble.

### 2026-04-27 — iter6: bigger model 256/L7/slice128 (regression)
- **Hypothesis:** Boost capacity — n_hidden 192→256, n_layers 6→7, slice_num 96→128, n_pos_freqs 10→14, max_pos_freq 16→24. With lr lowered to 4e-4 for stability. Even at fewer epochs (30 target → ~24 actual) the larger model should generalize better.
- **Change:** train.py model_config + lr only.
- **Result:** Trained 25 epochs in 30 min (74s/epoch vs 48s before). Best epoch 25 val/loss=3.48 (vs iter5 2.90). Val mae_surf_p avg=73.4 (vs iter5 65.0). W&B run `iter6-h256-L7-s128-pf14` / fc-prefixed run id (predictions auto-submitted to commit c3af0ef but worse than iter5).
- **Verdict:** **discarded** — `git reset --hard HEAD~1` to restore iter5 config. Capacity wasn't the bottleneck at this training budget; per-epoch time growth ate any quality gain.
- **Notes:** train losses converged similarly (vol≈0.10, surf≈0.034 at epoch 25), so the bigger model was undertrained for its size. If we want larger arch, need to find a way to keep epoch count high — e.g. smaller subsample (15k) or batch=8 with stable LR, or warm-start from iter5.

### 2026-04-27 — iter5: extended iter3 to 38 epochs (full 30-min budget)
- **Hypothesis:** Iter3 only used 19.9 of 30 min (25 epochs). Stretching the cosine schedule to 38 epochs and letting it train the full 30 min should give substantial improvement at the same batch/lr.
- **Change:** train.py — epochs 25→38. Everything else identical to iter3.
- **Result:** Trained all 38 epochs, hit 30 min timeout. Best epoch 34, val/loss=2.90 (vs iter3 3.86, -25%). Val mae_surf_p: single=63.4, geom_rc=84.7, geom_cruise=45.7, re_rand=66.4 (avg=65.0). **Test leaderboard: 59.94 avg surf_p (rank 5/7, was 6/7).** W&B run xqti3sr8.
- **Verdict:** kept (clear win — same setup, just longer training).
- **Notes:** Train losses still falling at epoch 34 (vol≈0.07, surf≈0.024). Fine-tuning from this checkpoint with lower lr, or training even longer, should still help.

### 2026-04-27 — iter4: bigger batch + dropout (regression)
- **Hypothesis:** With 10GB VRAM headroom from iter3 subsampling, increase batch_size to 8 and add dropout=0.05 for stronger regularization. Higher lr=8e-4 should pair with batch growth; surf_weight=30, epochs=35.
- **Change:** train.py — batch=4→8, lr=5e-4→8e-4, surf_weight=25→30, vol_subsample=25k→30k, epochs=25→35, dropout=0.05.
- **Result:** Trained 33 epochs in 30 min (timed out). Best epoch 28, val/loss=5.21. Val mae_surf_p avg=90.3. Worse than iter3 (3.86 / 77.4). W&B run r7jgqasd.
- **Verdict:** **discarded** — iter3 hyperparams retained. Dropout + higher lr + bigger batch slowed convergence enough that the extra epochs didn't recover. Resetting code to iter3.
- **Notes:** Auto-submitted to /mnt/new-pvc/predictions/apr27-4/tanjiro/7f8926f anyway, but leaderboard takes best commit so iter3 (90a7a6b @ 69.57) stays.

### 2026-04-27 — iter3: subsample volume nodes for 5× speedup
- **Hypothesis:** train_surf was still falling at 30-min timeout in iter2. Volume nodes (~99% of mesh) dominate compute but contribute little to surface pressure. Randomly drop volume nodes during training (keep all surface) so each epoch is 5× faster — converting compute time into more epochs.
- **Change:** train.py — added `_VolumeSubsample` dataset wrapper that keeps all surface nodes and samples up to 25k volume nodes per training sample. Validation runs on full mesh. surf_weight 20→25, epochs 15→25.
- **Result:** Each epoch now ~48s (vs 245s) → completed all 25 epochs in 19.9 min. VRAM peak 10GB (down from 84GB). Best val/loss=3.86 (epoch 25, vs iter2 5.07). Val mae_surf_p: single=78.7, geom_rc=82.5, geom_cruise=64.8, re_rand=83.5 (avg=77.4). **Test leaderboard: 69.57 avg surf_p (rank 4/5, up from rank 4 at 105.58)**. W&B run 2ypzrvpq.
- **Verdict:** kept (massive win — ~34% improvement on leaderboard).
- **Notes:** Subsampling introduces train-noise but the lr schedule converges nicely. Train losses stable at vol≈0.10, surf≈0.03 by epoch 25. Headroom probably remains: try keeping more volume points (50k), bigger model now that VRAM is free, or longer training.

### 2026-04-27 — iter2: Fourier positional encoding + lr 5e-4
- **Hypothesis:** Adding sinusoidal positional encoding on (x, z) coordinates (10 log-spaced freqs up to 16 cycles/unit) gives the MLP preprocess access to high-frequency spatial information that helps with turbulence. Lower lr=5e-4 + drop p_channel_weight should remove iter1 instability.
- **Change:** models.py — added `n_pos_freqs`, `max_pos_freq` to Transolver; appends sin/cos features of (x,z) before preprocess MLP. train.py — n_pos_freqs=10, max_pos_freq=16, lr=1e-3→5e-4, p_channel_weight=2→1, epochs=12→15.
- **Result:** Trained 8 epochs, best val/loss=5.07 (down from iter1 6.02). Per-split val mae_surf_p: single=141.2, geom_rc=142.3, geom_cruise=83.8, re_rand=112.6 (avg=120.0). Test leaderboard: 105.58 avg surf_p (rank 4/5). W&B run k380h8ov.
- **Verdict:** kept (small but real improvement). Still far behind leaders (edward 43.75, alphonse 50.83 on test).
- **Notes:** train_surf still falling at epoch 8 (0.41→0.087) — model is undertrained. Surface mae_p plateaued ~120 across iters. Next: subsample volume points to allow 2× more epochs in 30-min budget while keeping all surface points.

### 2026-04-27 — iter1: bigger Transolver + bf16 + p-weighted surface loss
- **Hypothesis:** Scaling Transolver (192 hidden, 6 layers, 8 heads, slice_num=96), training in bf16 for speed, surf_weight=20, and weighting pressure 2× in surface MSE (since the leaderboard ranks by surface pressure MAE) will beat prior tanjiro (51.42 avg surf_p on apr27).
- **Change:** train.py — n_hidden 128→192, n_layers 5→6, n_head 4→8, slice_num 64→96, lr 5e-4→1e-3, surf_weight 10→20, added per-channel weighted surface MSE (pressure ×2), bf16 autocast in train+val, grad_clip=1, epochs=12. predict.py rewritten to load Transolver from new models.py and apply hard no-slip on surface velocity. predict.py auto-submit had imported train.py and triggered double CLI parsing → moved model classes into `models.py`.
- **Result:** Trained 8 epochs in 30 min (timed out). Best epoch 8: val/loss=6.02. Per-split val mae_surf_p: single=145.8, geom_rc=137.1, geom_cruise=90.2, re_rand=122.2 (avg=123.8). VRAM peak 84.6GB (within budget). W&B run zbie1byp.
- **Verdict:** kept (predictions submitted to apr27-4/tanjiro/55efe74). But val mae_surf_p (~124) is much worse than prev tanjiro's TEST mae_surf_p (~51) — the new run is undertrained / worse-tuned. Suspect bf16 inference precision or aggressive lr=1e-3 hurt convergence.
- **Notes:** Strong vol_loss decrease (0.73→0.20) but surf_loss plateaus around 0.13. Pattern suggests model is fitting field but not pressure peaks on surface. Next: try fp32, lr 5e-4, longer effective training.



