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

### 2026-04-27 — iter4 + ensemble2 (chain only)
- **Hypothesis (iter4):** lr=2e-6, 8ep, no warmup — squeeze the last bit out of the chain. Expecting ~0.2pt val gain.
- **Result iter4:** 8 epochs, 20.0 min. Val 53.86→53.52→53.54→53.43→53.49→53.34→53.36→**53.32**. Best epoch 8. Predictions at `bb24f96`.
- **Hypothesis (ensemble2):** ensemble1 (iter1+iter2+iter3) scored 47.67 — WORSE than iter3 alone (46.87) because iter1 (val 81) was too weak and dragged the average down. Drop iter1, keep only chain endpoints {iter2, iter3, iter4} with weights tilted toward iter4 (0.25/0.30/0.45).
- **Change:** No code change. `python ensemble.py --sources 031565e 8c116e8 bb24f96 --weights 0.25 0.30 0.45`. Output saves to current HEAD.
- **Result:** TBD pending scorer.
- **Verdict iter4:** Kept. **Verdict ensemble2:** TBD. Chain ckpts saved to `/tmp/iter{1,2,3,4}_best.pt`.
- **Notes:** Big lesson from ensemble1: ensemble of imbalanced models is dragged by the weakest member. Apr23 iter12 (3-way at 0.3/0.5/0.2) worked because *all three* were within ~10% val of each other. Here iter1 was 50% worse than iter2/3/4. Going forward: only ensemble models within ~20% val of best.

### 2026-04-27 — ensemble1: iter1+iter2+iter3 weighted (0.1/0.35/0.55) — submitted at `8c66656`
- **Hypothesis:** Three chain checkpoints with different val/avg_surf_p (81.4 / 54.4 / 53.5) should ensemble lower than the best single via prediction-space averaging. Weight by inverse strength: iter3 (best) gets 0.55, iter2 0.35, iter1 0.1 for tiny diversity.
- **Change:** No code change. `python ensemble.py --sources 7ceb221 031565e 8c116e8 --weights 0.1 0.35 0.55` while iter4 is training. Output goes to commit `8c66656` (current HEAD = iter3 journal commit).
- **Result:** TBD — pending scorer pickup. Expected ~46.0–46.5 (small gain over iter3 solo at 46.87).
- **Verdict:** TBD.
- **Notes:** Leaderboard at submission: I'm #2 with iter3 at 46.87, thorfinn #1 at 45.94. Single_in_dist is my biggest gap (51.72 vs thorfinn 42.84 — 8.88 points). I'm AHEAD on re_rand (44.28 vs 51.05) and tied on geom_cruise. So ensembling should help close the single_in_dist gap by averaging away over-confident errors there.

### 2026-04-27 — iter3: chain warm-start lr=5e-6 (continue iter2)
- **Hypothesis:** Same recipe as iter2 but with LR halved-then-some — warm iter2 at lr=5e-6, 1-ep warmup + cosine, 10 epochs. apr23 iter101 got val 1.40→1.00 with this stride; even diminishing returns should still drop val a few tenths.
- **Change:** No code change. CLI: `--warm_start /tmp/iter2_best.pt --batch_size 2 --train_subsample 0 --lr 5e-6 --epochs 10 --warmup_epochs 1`. Run `knmw6p1d`.
- **Result:** 10 epochs, 25.0 min. Val curve 54.33→54.42→54.83→53.88→53.94→53.77→53.84→53.73→53.58→**53.53**. Best epoch 10: val/avg_surf_p=**53.53**, val/loss=1.45. Predictions at `8c116e8`.
- **Verdict:** Kept — small but strictly better gain (-0.84 from iter2). Saved to `/tmp/iter3_best.pt` for subsequent chain step / ensemble.
- **Notes:** The chain is plateauing. Train loss bottomed at vol=0.37 surf=0.28 (same as iter2). Next: iter4 lr=2e-6 to lock in the chain endpoint, plus an ensemble of {iter1, iter2, iter3} for diversity (apr23 lesson: even chain-correlated ensembles add 0.5-1pt).

### 2026-04-27 — iter2: bs=2 no-subsample warm-start lr=2e-5 — 🚀 BIG jump
- **Hypothesis:** Replay the apr23 iter93 breakthrough — warm-start iter1's checkpoint, drop to bs=2 with NO volume subsampling (so the model sees the full 240K-node grid), 1-epoch warmup + cosine, lr=2e-5, p_w=3, L1, 10 epochs. iter93 went val/loss 1.40 → 1.02 with this exact recipe.
- **Change:** No code change — only CLI flags: `--warm_start /tmp/iter1_best.pt --batch_size 2 --train_subsample 0 --lr 2e-5 --epochs 10 --warmup_epochs 1`. Run `c611mrv5`.
- **Result:** 10 epochs, 25.1 min, peak 29.1 GB. Val curve 69→59→58→57→57→55→55→55→**54.37**→54.37. Best epoch 9: val/avg_surf_p=**54.37**, val/loss=1.47. Per-split val/loss: single=2.44, rc=1.78, cruise=0.37, re_rand=1.29 — same pattern as apr23 iter93. Predictions at commit `031565e`.
- **Verdict:** Kept — `27 points` lower val_surf_p than iter1; this is the breakthrough. thorfinn's leaderboard test is 45.94, so I'm now plausibly in striking distance once scoring lands.
- **Notes:** bs=2 with full mesh is *the* recipe that beats the leader plateau in apr23. Train loss bottomed at vol=0.37 surf=0.27 — model has more capacity to give. Next: iter3 chain lr=5e-6, then maybe iter4 at lr=2e-6 to mirror apr23's iter101/iter111 steps before ensembling.

### 2026-04-27 — iter1: 192x6 L1 p_w=3 sub40K bs=8 (apr23 baseline port)
- **Hypothesis:** Port the apr23 frieren iter4/iter15 recipe verbatim — Transolver 192x6, slice=64, mlp_ratio=2, n_head=6, L1 loss with surface p up-weighted (p_w=3), bf16, AdamW betas=(0.9,0.95), warmup=3+cosine, sub40K volume nodes at bs=8, 35 epochs. Establishes a strong starting point for the chain ensembles that won apr23.
- **Change:** Created `model.py` (Transolver), rewrote `train.py` (apr23 frieren training loop with `--warm_start` flag, `MAX_TIMEOUT_MIN` env, mirror to PVC + `checkpoints/best.pt`, auto-submit), rewrote `predict.py` to load model from `config.yaml`. Added `ensemble.py` (still uncommitted; queued for later iters).
- **Result:** 35 epochs, 26.9 min, peak 20.8 GB. Best epoch 34: val/avg_surf_p=**81.37** (single=2.53, rc=2.79, cruise=1.08, re_rand=2.03 — split losses, not surf_p MAE). Run `zq0fst5n`. Predictions at commit `7ceb221` (still `incomplete` in scores at journal time).
- **Verdict:** Kept — trajectory is monotonic (314→81) and the cosine tail is still descending at e34, so warm-start chain should keep gaining.
- **Notes:** thorfinn currently #1 at test surf_p=45.94. The apr23 lesson is that bs=8+sub40K converges to a local minimum that bs=2+no_subsample warm-start can blow past (val 1.4 → 1.0 in iter93). That's the iter2 plan.

