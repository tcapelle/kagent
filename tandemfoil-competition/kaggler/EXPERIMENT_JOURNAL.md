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

### 2026-04-27 — iter8: chain link 3 — warm iter3 lr=2e-6 (commit f5695bd)
- **Hypothesis:** Following frieren's iter111 recipe: continue chain at lr=2e-6 for marginal val gain + ensemble averaging.
- **Change:** `--warm_start /tmp/iter3_best.pt --lr 2e-6 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same 192x6.
- **Result:** val/loss=**1.5073** at epoch 9 (25.1 min). Per-split val: single=2.39, rc=1.91, cruise=0.42, re_rand=1.31. ~0.6% gain over iter3 — chain saturating as frieren predicted.
- **Verdict:** kept. Predictions saved at `f5695bd` (after manual `predict.py` re-run because the train.py auto-call OOM'd: training process held 88GB and the `subprocess` spawn only had 5GB to work with).
- **Notes:** OOM bug: `train.py` auto-runs `predict.py` after training without freeing model VRAM first. Should fix: `del model; torch.cuda.empty_cache()` before subprocess call. Workaround for now: re-run predict.py manually after killing the train process.

### 2026-04-27 — iter6: warm iter5 256x8 bs=2 no-subsample (commit 7f2faa7)
- **Hypothesis:** Apply frieren's bs=2 + no-subsample breakthrough to the bigger 256×8 model. Larger capacity + full mesh + low LR should push past 192×6 ceiling.
- **Change:** `--warm_start /tmp/iter5_best.pt --n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. 50.7GB peak.
- **Result:** val/loss=**1.8014** at epoch 7 (30.0 min, hit MAX_TIMEOUT). Per-split val: single=2.91, rc=2.29, cruise=0.52, re_rand=1.50. Better than iter5 alone (2.31) but **worse than iter3 (1.52)** — iter5 was undertrained so iter6 inherits that gap.
- **Verdict:** kept for ensemble diversity. Different architecture should add value when blended with iter3/iter4.
- **Notes:** 256×8 at bs=2 no-subsample takes 257s/epoch (vs 152s for 192×6). Only 7 of 10 epochs ran. Could do iter9 to chain at lower LR after a longer pre-training pass. Next: iter7 = 2-way ensemble iter3+iter6.

### 2026-04-27 — iter5: 256x8 slice=64 fresh-train (commit 5b181c5)
- **Hypothesis:** Bigger Transolver (n_hidden=256, n_layers=8, n_head=8) for ensemble diversity. Frieren's best apr27 ckpt (model-9f4m2qmm) used the same shape.
- **Change:** `--n_hidden 256 --n_layers 8 --n_head 8 --slice_num 64 --epochs 25 --batch_size 4 --train_subsample 60000 --lr 5e-4 --warmup_epochs 3`. Fresh init.
- **Result:** **val/loss=2.3131 at epoch 17** (30.3 min, 26.6GB) — hit MAX_TIMEOUT before epoch 25. Per-split val: single=3.61, rc=2.93, cruise=0.82, re_rand=1.90. Still descending but undertrained.
- **Verdict:** kept as warm-start base for iter6 (apply bs=2 no-subsample breakthrough).
- **Notes:** Larger arch (3.94M params) is slower (107s/ep at bs=4 sub=60K vs 63s for 192x6). Predictions at commit `5b181c5` (HEAD when predict ran, not the iter5 placeholder). Next: iter6 = warm iter5 bs=2 no-subsample lr=2e-5 10ep — bigger capacity model after breakthrough recipe.

### 2026-04-27 — iter4: 2-way ensemble iter2 0.4 + iter3 0.6 (commit a00c6ea)
- **Hypothesis:** iter2 (val 1.532) and iter3 (val 1.516) are sequentially-trained chain links — adding their predictions with iter3 weighted higher (since it's stronger) should reduce variance like SWA. Free win, no training.
- **Change:** `python ensemble.py --sources 381bc71 e352f58 --weights 0.4 0.6`. Predictions written to `apr27/tanjiro/a00c6ea/`.
- **Result:** TBD — will appear on leaderboard next refresh. Expected: marginal improvement over iter3 alone.
- **Verdict:** kept. Free improvement to bank before continuing.
- **Notes:** Frieren's apr23 history shows weighted ensemble of chain iterations beats single best by ~0.1-0.2 score points consistently.

### 2026-04-27 — iter3: chain link 2 — warm iter2 lr=5e-6 (commit e352f58)
- **Hypothesis:** Following frieren's iter101 recipe: continue chain at 4× lower LR (5e-6) for further fine-tuning. Should give 1-2% val improvement and add ensemble diversity to iter2.
- **Change:** `--warm_start /tmp/iter2_best.pt --lr 5e-6 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same architecture.
- **Result:** val/loss=**1.5157** at epoch 9 (25.1 min, 29.1GB). Per-split val: single=2.42, rc=1.91, cruise=0.43, re_rand=1.31. Marginal improvement over iter2 (1.532); chain saturating.
- **Verdict:** kept. Marginal gain but valuable for ensemble averaging.
- **Notes:** Predictions written to commit `e352f58` (head moved due to ensemble.py commit). Frieren's experience: each chain link gives 0.01-0.05 val gain; bigger wins come from ensembling multiple chain links + architectural diversity (slice=128). Next: iter4 = ensemble iter2+iter3 (free win), then iter5 = slice=128 diversity model.

### 2026-04-27 — iter2: warm-start iter1 + bs=2 no-subsample (BREAKTHROUGH recipe, commit 381bc71)
- **Hypothesis:** Frieren's apr23 iter93 showed bs=2 + train_subsample=0 (full mesh) is ~30% better than bs=8 sub=40K. Apply to iter1 warm-start at lr=2e-5 cosine over 10 epochs (no warmup since model is pre-trained).
- **Change:** `--warm_start /tmp/iter1_best.pt --lr 2e-5 --epochs 10 --warmup_epochs 0 --batch_size 2 --train_subsample 0`. Same 192x6 architecture.
- **Result:** val/loss=**1.5324** at epoch 10 (25.1 min, 29.1GB). Per-split val: single=2.41, rc=1.96, cruise=0.44, re_rand=1.32. **23% improvement over iter1's 1.997**, all splits improved.
- **Verdict:** kept. Confirmed frieren's recipe works. Predictions saved at commit 381bc71.
- **Notes:** Still well above frieren's iter93 val 1.0158 — main reason is my iter1 warm-start (val 1.997) is much weaker than their iter79 (val 1.40 after 4 chain links). To close the gap I should: (a) chain more links at lower LR, (b) longer pre-training before bs=2 step, (c) eventually slice=128 diversity. Next: iter3 = warm iter2 lr=5e-6 10ep (chain link 2).

 + bf16 + p_weight=3 + warmup + bs=4 sub60K (commit 9a14753)
- **Hypothesis:** Reproduce frieren's apr23 recipe: 192x6 Transolver, L1 loss for outlier robustness, bf16, p_weight=3 surface-pressure boost, warmup+cosine, bs=4 with subsample to 60K volume nodes. Pre-train for warm-start chain.
- **Change:** Refactored `train.py`/`predict.py` and added `model.py` with Transolver. New flags: `loss_type`, `p_weight`, `warmup_epochs`, `train_subsample`, `warm_start`, bf16 autocast, grad_clip=1.0.
- **Result:** val/loss=1.9973 at epoch 29 (30.4 min, 15.3GB peak). Per-split val: single=2.54, rc=2.87, cruise=0.70, re_rand=1.87. Cosine still descending at the end → likely undertrained for this config.
- **Verdict:** kept as warm-start base for iter2. Score TBD but expected similar to last apr27 iter1 (~57 surf_p).
- **Notes:** A bit worse than last session's iter1 (val 1.685) — random init variance and `warmup_epochs=3` (last time was different). Real win comes next: bs=2 + no-subsample warm-start (frieren's iter93 went 1.4→1.0 → score 35).
