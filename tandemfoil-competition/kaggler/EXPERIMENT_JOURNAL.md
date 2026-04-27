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

### 2026-04-27 — iter13: EMA chain at lr=5e-7
- **Hypothesis:** Continue EMA chain at lower LR — small but stable refinements.
- **Change:** No code change. CLI `--warm_start models/model-13kcrwn3 --lr 5e-7 --ema_decay 0.99 --camber_noise 0.05`. Run id `p2atdl76`.
- **Result:** Best val/avg_surf_p=46.59 at epoch 5/8 (-0.16 from iter12). Plateauing around 46.6.
- **Verdict:** Kept (ckpt commit `b957b98`). Top-3 leaders: nezuko 35.20, thorfinn 35.20 (tied!) tanjiro 38.45 — they have something architectural I don't. Need fundamental change. Going for Fourier features (iter14).

### 2026-04-27 — iter12: EMA + camber_noise chain (BREAKTHROUGH)
- **Hypothesis:** Tanjiro uses `ema_decay`, nezuko has `fourier_*` flags. Add EMA (exponential moving average of weights) — known trick for plateaued models, costs little to implement.
- **Change:** train.py: `ema_decay` flag, EMA shadow state updated each opt step, EMA weights used for validation and saved as the `best.pt`. Code commit `e4556eb`.
- **Result:** Best val/avg_surf_p=46.75 at epoch 7 (-0.16 from iter10). EMA gave a small but real boost — first improvement in 3 iters of chaining.
- **Verdict:** Kept. Will continue chaining with EMA.

### 2026-04-27 — iter11: AoA noise (regression)
- **Hypothesis:** Add AoA noise alongside camber_noise to make the model robust to AoA perturbations and continue squeezing out gains.
- **Change:** train.py: `aoa_noise` flag added to subsample collate. CLI `--aoa_noise 0.02 --camber_noise 0.05`. Run id `4y23f5f6`. Code commit `c3f8792`.
- **Result:** Best val/avg_surf_p=47.00 at epoch 2, then drifted up to 47.4 by epoch 8. Slightly worse than iter10's 46.91.
- **Verdict:** Discarded checkpoint (iter10 still best). aoa_noise=0.02 is too aggressive — AoA std is only 0.07-0.09 raw, so noise=0.02 is ~25% of std, which destabilizes physics. Code change kept (flag is useful).
- **Notes:** Tested SWA averaging of iter5b/6/7/8/9/10 weights — all worse than iter10 alone. Models too correlated (same chain lineage). Need different angle.

### 2026-04-27 — iter10: continue chain at lr=1e-6
- **Hypothesis:** iter9 still descending at timeout. Continue chaining at same LR to push past 47.
- **Change:** No code change. CLI `--warm_start models/model-4cx1uha0/checkpoint.pt --lr 1e-6 --camber_noise 0.05 --epochs 8 --batch_size 2 --train_subsample 0`. Run id `w0gqy59p`.
- **Result:** Best val/avg_surf_p=46.91 at epoch 6/8 (-0.34 abs from iter9). Plateau forming.
- **Verdict:** Kept (ckpt commit `7ae71a8`, predictions at `fern/a8e19c5`).
- **Notes:** Diminishing returns from chaining. Need new angle: EMA, Fourier features, larger model, or TTA in predict.

### 2026-04-27 — iter9: chain camber_noise at lr=1e-6
- **Hypothesis:** iter8 was still descending at the timeout (best at epoch 7 of 8). Continue chaining at lower LR with camber noise to squeeze more.
- **Change:** No code change. CLI `--warm_start models/model-viigighk/checkpoint.pt --lr 1e-6 --epochs 8 --camber_noise 0.05 --train_subsample 0 --batch_size 2`. Run id `4cx1uha0`.
- **Result:** Best val/avg_surf_p=47.25 at epoch 8 (-0.46 abs). Improved every epoch. Train surf 0.40 → 0.38.
- **Verdict:** Kept (ckpt commit `c722642`). Will continue chaining (iter10).
- **Notes:** Leaderboard updated: thorfinn dropped to 35.72 (val~55, ratio 0.65). My val/test ratio is 0.88. Suspicious gap suggests thorfinn may have TTA / ensembling / different predict.py — worth investigating once chain plateaus.

### 2026-04-27 — iter8: camber_noise + full-mesh — second breakthrough
- **Hypothesis:** iter7 plateaued at 49.33 (val). Single-foil split (val_single_in_dist) was actually my weakest at 53.65 — plateau is overfit. Combine the camber_noise augmentation (originally tried in iter5, which was killed early) with the full-mesh fine-tune to add regularization without giving up signal.
- **Change:** No code change — flag already exists from `395a063`. CLI: `--warm_start models/model-q5t9h880/checkpoint.pt --lr 2e-6 --epochs 8 --batch_size 2 --train_subsample 0 --camber_noise 0.05`. Run id `viigighk`.
- **Result:** Best val/avg_surf_p=**47.71** at epoch 7/8 (-1.62 abs from iter7). Steady improvement every epoch — augmentation gave the model fresh signal. Train loss stayed ~similar (small noise, not destabilizing).
- **Verdict:** Kept (ckpt commit `61a9205`).
- **Notes:** camber_noise=0.05 σ in raw NACA-M units (σ≈0.5 in M digit) was a sweet spot — small enough not to destroy semantics, large enough to interpolate adjacent camber values. Will continue chaining with augmentation at lower LR (iter9).

### 2026-04-27 — iter7: chain at lr=2e-7
- **Hypothesis:** Continue chaining low-LR fine-tune from iter6's 49.61 — diminishing returns expected but cheap.
- **Change:** No code change; CLI `--warm_start models/model-rz1akevm/checkpoint.pt --lr 2e-7 --epochs 8`. Run id `q5t9h880`.
- **Result:** Best val/avg_surf_p=49.33 at epoch 6/8 (-0.28 abs). Per-split: cruise=30.36, rc=64.97, re_rand=49.30, single=53.65.
- **Verdict:** Kept (ckpt commit `53a2155`). The chain is plateauing; need new angle.
- **Notes:** predict.py auto-run again OOM'd at split 3 — workaround: kill training procs, run predict manually. Single-foil split is now my weakest (53.65) — thorfinn beats me 40 vs 50 there. Need to focus next iter on regularization or augmentation.

### 2026-04-27 — iter6: chain at lr=5e-7
- **Hypothesis:** iter5b's val/avg_surf_p=50.53 was still trending down at the timeout (epoch 9). Chain another fine-tune from that ckpt at a smaller LR to squeeze more out before plateauing. Frieren's `iter4-warm-bs2-nosub-lr2e6-chain3` shows chaining works.
- **Change:** No code change — just CLI `--warm_start models/model-hahrr3i7/checkpoint.pt --lr 5e-7 --epochs 8 --batch_size 2 --train_subsample 0`. Run id `rz1akevm`.
- **Result:** Best val/avg_surf_p=**49.61** at epoch 5/8. Per-split: cruise=30.34, rc=64.96, re_rand=49.30, single=53.86. Marginal -0.9 absolute. **Pushed fern to RANK #1 on leaderboard at 45.01** (iter5b commit `cc186e5` actually scored — iter6 not yet rescored). predict.py OOM'd from training subprocess; ran manually.
- **Verdict:** Kept (ckpt commit `df602dd`). Plateau is real — most of the gain came in iter5b; further chains return diminishing improvements.
- **Notes:** Subsequent epochs at lr=5e-7 oscillated 49.61-50.13 (basically noise). Need fundamentally different angle for further gains: lower LR / camber augmentation / different loss / TTA.

### 2026-04-27 — iter5b: full-mesh fine-tune (frieren-style) — BREAKTHROUGH
- **Hypothesis:** Frieren leaderboard scoring 46.87 jumped from val/avg_surf_p~78 to ~53 by fine-tuning with **full mesh** (`train_subsample=0`), `batch_size=2`, and very low LR (2e-6). Subsampling 40k pts during training was leaving a lot of supervision on the table — the model wasn't seeing the wake/far-field nodes during fine-tune.
- **Change:** train.py: `make_subsample_collate` now treats `n_keep<=0` as "no subsampling". Killed iter5 (camber noise) early to free GPU. Run `--warm_start iter4 --slice_num 128 --train_subsample 0 --batch_size 2 --lr 2e-6 --epochs 10`. Code commit `cc186e5`. Run id `hahrr3i7`.
- **Result:** Best val/avg_surf_p=**50.53** at epoch 9/9 (still improving when timeout hit). Per-split: cruise=30.6, rc=65.55 (was 121.75!), re_rand=49.6, single=56.3 (slight regression vs iter4's 53.4). Train each epoch ~3.5 min, peak VRAM 42 GB. -23 absolute val/avg_surf_p in ~30 min!
- **Verdict:** Kept (ckpt commit `b7ff534`, predictions in `fern/cc186e5`).
- **Notes:** The huge gain came from `geom_camber_rc` (-56) — the model now sees full mesh during fine-tune so it learns the OOD camber regimes via wake propagation, not just the local geometry features. The camber-noise augmentation experiment (iter5) was a less-good idea given how well full-mesh works. Next: chain another low-LR fine-tune (iter6).

### 2026-04-27 — iter5: warm-start iter4 + camber-noise augmentation (KILLED)
- **Hypothesis:** `val_geom_camber_rc` surf_p MAE stuck at ~122 — test OOD camber (M=7,8 unseen for tandem). Add Gaussian noise on foil1 camber/position/thickness inputs to interpolate.
- **Change:** train.py: added `camber_noise` flag in subsample collate. Run `--warm_start iter4 --lr 2e-5 --camber_noise 0.1`. Code commit `395a063` (kept — flag is still useful).
- **Result:** Killed at epoch 1 (val=74.49 — slightly worse than iter4's 73.21). Replaced with iter5b (full-mesh) after seeing frieren's leaderboard jump.
- **Verdict:** Discard the run (no checkpoint kept). The flag in code is preserved; may revisit combined with full-mesh.
- **Notes:** Premature kill — could have let it converge. But frieren's full-mesh approach was clearly higher EV.

### 2026-04-27 — iter4: warm-start fine-tune of slice_num=128 cold-start
- **Hypothesis:** iter3's slice_num=128 cold-start undertrained at 30 min (val 90.50). Warm-starting the slice_num=128 model with iter2's loss recipe should let the bigger arch beat slice_num=64 + warm-start (iter2's 79.12).
- **Change:** No code change; CLI `--slice_num 128 --warm_start models/model-65tgozcw/checkpoint.pt --lr 5e-5 --epochs 50`. Run id `kjb26vxt`.
- **Result:** Best val/avg_surf_p=**73.21** at epoch 28/30. Per-split: cruise=41.6, rc=121.75, re_rand=76.1, single=53.4 — main gain on `single_in_dist` (-17 from iter2). geom_camber_rc still bottleneck at ~122.
- **Verdict:** Kept (predictions in `fern/5e79ba8`, ckpt commit `2f0eba4`). Best fern submission so far.
- **Notes:** Ensembling tried (iter1+iter2+iter4, etc.) — all worse than iter4 alone on val. Cross-arch averaging drags iter4 down on in-distribution splits even though it slightly helps geom_rc. iter4's plateau suggests we should attack OOD camber directly (→ iter5).

### 2026-04-27 — iter3: cold-start slice_num=128 with thorfinn loss
- **Hypothesis:** Thorfinn's leader checkpoint uses slice_num=128 (vs my 64). Maybe more slice tokens help capture fine-grained physics. Cold-start with slice_num=128 + balanced thorfinn loss to set up a stronger base for warm-start fine-tune.
- **Change:** No code change; just CLI `--slice_num 128 --epochs 200`. Run id `65tgozcw`. Epoch time 61s (vs 42s for slice_num=64). VRAM 14.0 GB (vs 9.7).
- **Result:** Best val/avg_surf_p=90.50 at epoch 27/30. Worse than iter1 (84.71) and iter2 (79.12). Per-split: cruise=52.2, rc=129.1, re_rand=86.9, single=93.8 — single_in_dist worse than iter1.
- **Verdict:** Discard for production checkpoint. Best.pt remains iter2's 79.12. But the slice_num=128 ckpt is useful as a warm-start source for iter4.
- **Notes:** 30-min cap forced only 30 epochs (vs 43 for slice_num=64), and the bigger model needs more time. Checkpoint preserved at `models/model-65tgozcw/checkpoint.pt` for warm-starting.

### 2026-04-27 — iter2: warm-start fine-tune with thorfinn 4-weight loss
- **Hypothesis:** Thorfinn (45.94) jumped from 458 → 65 by warm-starting from a converged checkpoint and fine-tuning with separate region/channel loss weights heavily favoring surface pressure (`surf_p=6, surf_uv=1, vol_p=0.5, vol_uv=0.5`). Apply the same trick to iter1's 84.71 ckpt.
- **Change:** train.py: replaced `surf_weight + ch_weights[p_weight]` with 4 explicit weights. Added `_l1_uv_p` helper. Run `--warm_start checkpoints/best.pt --lr 5e-5 --epochs 50`. Cosine over 50 epochs (no warmup since warm-starting). Run id `hsytzk3a`.
- **Result:** Best val/avg_surf_p=79.12 at epoch 3 (down from 84.71). Plateaued thereafter, oscillating 82–84. Hit 30-min cap at epoch 43. Train surf loss kept decreasing (0.45 → 0.27) but val surf_p stayed flat → overfitting.
- **Verdict:** Kept (`b17c849` + ckpt commit `9faf840`). Modest improvement (~5 surf_p). Predictions submitted to `fern/b17c849`.
- **Notes:** Quickly converges within 3 epochs then overfits, even with the surf-p heavy loss. Suggests model capacity / regularization is the bottleneck — the 1.71M-param Transolver with slice_num=64 has saturated. Next directions: bigger slice_num=128, dropout, or augmentation. Also worth trying lower LR (1e-5) for more stable fine-tune.

### 2026-04-27 — iter1: thorfinn-style 192/6/6 Transolver baseline
- **Hypothesis:** Reproduce thorfinn's recipe (Transolver 192h/6L/6H, L1 with channel weights on p, surface re-weighting, 40k-pt subsample, bf16+warmup+cosine) to establish a solid baseline before any custom work. Previous fern submission was 131.69; thorfinn at 45.94.
- **Change:** train.py: L1 loss with `ch_weights=[1,1,p_weight=3]`, `surf_weight=10`, batch_size=4, lr=5e-4 → cosine, warmup=3 epochs, 40k subsample (all surface + random volume), 30-min cap. Model 192/6/6, slice_num=64. Run id `go59tm9o`.
- **Result:** Best `val/avg_surf_p=84.71` at epoch 36/43. Per-split surf_p: single=80.5, geom_rc=123.6 (worst — OOD camber), geom_cruise=51.1 (best), re_rand=83.6. Train surf L1 0.51 → 0.11; vol 0.65 → 0.18. VRAM peak 9.7 GB.
- **Verdict:** Kept (`6e73d15`). Big improvement over previous fern submission (131.69 → ~85). Predictions submitted to `predictions/fern/6e73d15`.
- **Notes:** geom_camber_rc is the bottleneck — likely the "out-of-distribution camber" generalization gap. Next: try thorfinn's warm-start fine-tune trick — heavy `surf_p_weight=6` with low LR for last few epochs. Slice_num=128 would also be worth trying but requires retraining the model from scratch.

