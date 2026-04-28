# Autonomous Agents on the TandemFoil CFD Surrogate Competition

## Setup

Eight Claude Opus 4.7 agents competed autonomously on our internal CFD surrogate
benchmark built on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn)
dataset: given 24-dim per-node features of a 2D overset mesh around one or two
airfoils (position, signed arc-length, shape descriptors, Reynolds number,
angle-of-attack, NACA profile, gap/stagger), predict the full
`(Ux, Uy, p)` field at every mesh node. Each agent received a single GPU pod,
a 30-minute-per-training-run budget, its own git branch, and the standard
instruction loop: *check the leaderboard, formulate a hypothesis, modify
`train.py`, train, score, commit, repeat.* The cohort ran for **~13 hours**
(2026-04-23 14:53 UTC → 2026-04-24 04:00 UTC) without human intervention
before the scheduled stop.

| Resource | Location |
|---|---|
| Dataset | [`TandemFoilSet`](https://openreview.net/forum?id=4Z0P4Nbosn) — 2,699 samples, 4 val/test splits testing geometry and Re generalisation |
| Primary metric | avg surface pressure MAE across 4 test splits (lower is better) |
| Agent branches | `origin/apr23/kaggler/<name>` |
| Leaderboard branch | `origin/apr23-leaderboard` |
| W&B project | [`wandb-applied-ai-team/kagent-tandemfoil`](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil) |
| Baseline (Transolver, `n_hidden=128, n_layers=5`) | published to each agent's `README.md` |

## Experimental harness

The experiment runs on a shared Kubernetes cluster with one GPU pod per
agent and a separate, smaller organiser pod. Agents and organiser communicate
strictly through three channels: a persistent shared volume for data,
predictions and logs; the git remote for code; and a W&B project for training
telemetry. No network path exists between agents, and each agent sees only its
own competition-facing working directory — the organiser area holding ground
truth and scoring code is invisible from inside a kaggler pod.

Each agent pod boots into an autonomy loop: pull the head of its branch, start
Claude Code with a short role prompt pointing at the competition instructions,
and let the model drive. The model reads the leaderboard, its own experiment
journal and its source files, modifies the training script, commits, trains
under the 30-minute wall-clock budget, writes predictions to the shared volume,
and loops. When context fills, a lightweight session-resume mechanism restarts
the loop from the same branch; the experiment journal (which every agent is
required to keep and commit separately from code changes) is the durable
cross-restart memory.

The organiser pod polls every 60 seconds, scores any new prediction files
against held-out ground truth, updates a single markdown leaderboard, and
pushes that leaderboard to a dedicated branch. Scoring is the only privileged
operation in the system.

Operationally, the apr23 run consumed one launcher invocation
(`n_kagglers=8 --organizer`) and one scheduled kill at 04:00 UTC, no manual
edits on any agent branch, **330 commits** across the eight agents, and
**223 leaderboard updates** from the organiser — roughly one scored
submission every 2.4 minutes for 13 hours.

## Leaderboard at the 12-hour mark (2026-04-24 03:03 UTC)

Verbatim from `origin/apr23-leaderboard` at commit `2a0fba24` — the
leaderboard update closest to launch + 12h. Reported here for like-for-like
comparison with the four 12h-target apr27 runs.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | frieren  | `0d4a21a` | **34.58** | 39.05 | 48.00 | 18.28 | 32.99 |
| 2 | fern     | `bdc8350` | 42.62 | 46.80 | 55.53 | 26.65 | 41.50 |
| 3 | edward   | `47149c7` | 44.00 | 48.00 | 59.19 | 26.44 | 42.35 |
| 4 | askeladd | `e8e0149` | 49.33 | 52.96 | 62.63 | 32.38 | 49.35 |
| 5 | alphonse | `8846478` | 56.21 | 57.41 | 70.36 | 40.80 | 56.29 |
| 6 | thorfinn | `71362e7` | 62.57 | 45.81 | 77.35 | 40.44 | 86.70 |
| 7 | tanjiro  | ensemble  | 68.70 | 54.78 | 83.55 | 45.84 | 90.62 |
| 8 | nezuko   | `adce894` | 85.84 | 96.56 | 96.43 | 63.43 | 86.94 |

The 12h ordering is identical to the 13h final; the extra hour gained
frieren 0.17 pts (34.58 → 34.41) on the same iter153 ensemble lineage and
small refinements at ranks 4 and 7. Every defining unlock of this run
(full-mesh `bs=2`, the rediscovered subsampling-trap diagnosis) was
already on the board by 12h.

## Final leaderboard (apr23 run, 13h actual)

Ranked by avg surface pressure MAE across the four hidden test splits.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | frieren  | `c57076b` | **34.41** | 38.73 | 47.92 | 18.21 | 32.78 |
| 2 | fern     | `bdc8350` | 42.62 | 46.80 | 55.53 | 26.65 | 41.50 |
| 3 | edward   | `47149c7` | 44.00 | 48.00 | 59.19 | 26.44 | 42.35 |
| 4 | askeladd | `.164885` | 48.89 | 52.09 | 62.46 | 32.02 | 49.01 |
| 5 | alphonse | `8846478` | 56.21 | 57.41 | 70.36 | 40.80 | 56.29 |
| 6 | thorfinn | `71362e7` | 62.57 | 45.81 | 77.35 | 40.44 | 86.70 |
| 7 | tanjiro  | ensemble  | 68.25 | 55.52 | 84.19 | 45.00 | 88.27 |
| 8 | nezuko   | `adce894` | 85.84 | 96.56 | 96.43 | 63.43 | 86.94 |

*Final snapshot: 2026-04-24 03:59 UTC.*

The baseline Transolver scores ~120 on this metric; **seven of eight agents
finish below 70**, **four below 50**, and the winner at **34.41** is a 3.5×
improvement over the baseline. The top four are tightly clustered (range
14.5 points), then there's a step to thorfinn and tanjiro, and nezuko alone
in the high 80s.

Frieren's lead over fern is **8.21 points** — widening rather than narrowing
in the final hour. Frieren is the best on every test split.

### Per-agent commits

| Agent | Commits | Final rank | Δ vs 5h snapshot |
|---|---:|---:|---|
| frieren  |  98 | 1 | +0 (held the top after overtaking askeladd at ~4h45m) |
| fern     |  47 | 2 | **+2** (up from rank 4) |
| edward   |  28 | 3 | **+4** (up from rank 7) |
| askeladd |  33 | 4 | −3 (down from rank 2) |
| alphonse |  41 | 5 | **+1** (up from rank 6) |
| thorfinn |  20 | 6 | **−3** (down from rank 3) |
| tanjiro  |  24 | 7 | −2 (down from rank 5) |
| nezuko   |  39 | 8 | +0 |
| **total** | **330** | | |

The late-game movement is striking: three of the top four (frieren, fern,
edward) all climbed during the overnight window, while thorfinn and tanjiro
fell despite committing actively. The single cause is the same for every
mover.

## Evolution of the competition

Four phases are visible in the journals and the leaderboard-branch history:

1. **Quick baselines (14:55 – 16:30 UTC).** All eight agents write a complete
   Transolver-based submission within ~90 minutes. Thorfinn, fern and
   alphonse are first to land, at test surf_p 88–135. Everyone independently
   discovers bf16 autocast + point subsampling (keep all surface nodes +
   random volume to 30–40K) as the dominant throughput lever, unlocking
   3–4× more epochs inside the 30-minute cap.

2. **Recipe convergence (16:30 – 18:00 UTC).** Agents converge on the same
   core configuration — `n_hidden=192, n_layers=6, slice_num=64, mlp_ratio=4,
   lr≈5-7e-4, warmup+cosine-by-steps, surf_weight≈20, surf_p_weight≈2` — by
   reading each other's commits from the leaderboard and copying what
   works. Nezuko, tanjiro and edward all reference thorfinn's config
   explicitly in their journal entries. Scores drop into the 75–90 range
   for most of the cohort.

3. **Warm-start fine-tune chains (18:00 – 23:00 UTC).** The top three agents
   chain multiple 30-minute runs — each warm-starting from the previous best
   checkpoint at roughly half the previous peak LR. Askeladd runs a six-link
   chain v1 → v6 at descending LRs `[…, 2e-4, 5e-5, 2e-5, 1e-5, 3e-5]`,
   taking #1 with test 56.07. Frieren replies with iter17 → iter19 →
   iter21 and overtakes at 55.32. Test scores across the cohort land in
   the 55–90 band, and the leaders' gap shrinks to <1 point. The agents
   visibly converge on the *same* shape of solution and start running out of
   low-LR head-room.

4. **The full-mesh unlock (23:00 UTC – 04:00 UTC).** Three agents
   independently discover, within a few hours of each other, that the
   subsampling they all relied on for speed was *causing* a train/eval
   distribution gap that the warm-start chain could not close. Turning off
   subsampling entirely, dropping to `batch_size=2`, and warm-starting a
   subsample-trained checkpoint into this regime produces step-changes of
   15–25% in a single 30-minute run. Frieren rides this from 52 → 35.27 and
   runs away with the lead; edward catches up from 67 → 44; alphonse from
   4.23 → 2.93 (val/l2); fern (who independently found a different
   regulariser — dropout — but then also discovered the full-mesh idea)
   rides it to 42.62. Thorfinn and tanjiro never make this jump and stall in
   the high 60s.

## The defining aha moment: the subsampling trap

The central finding of the run — rediscovered independently by three of the
top four agents — is that **training on a random subsample of mesh nodes
introduces a train/eval distribution mismatch for physics-attention models
that pool over the node set**. The slice-weight tensors inside Transolver's
`PhysicsAttention` are computed across the full node population; training
them on 40K points per sample and evaluating them on 240K points per sample
silently caps the model's re_rand generalisation. Subsampling looked like a
throughput win and behaved like a latent bug.

### Frieren: the breakthrough

Frieren's iter93 is the single largest single-iteration gain of the run:

> **iter93: bs=2 + no-subsample + warm-start = 🚀🚀 BREAKTHROUGH 🥇 #1 at 35.27.**
> Askeladd uses `batch_size=2` and no volume subsampling. My `batch_size=8` +
> subsample-to-40K-nodes was cheap but information-lossy. Try bs=2 no-subsample
> to match their setup…
>
> **Scored 35.27 avg_surf_p — #1 by 12.66 pts over edward (47.93), 14.68 over
> askeladd (49.95).** Per-split test: single=40.26, rc=48.87, cruise=18.50,
> re_rand=**33.43 (was 73 before). Re-generalization FIXED.**
>
> Subsampling was the root cause of my re_rand weakness — dropping 60% of
> volume nodes left the model unable to learn Re-dependent field structure.
> bs=2 gives 4x more gradient updates per epoch, and with no subsampling the
> model sees the full 240K-node grid. Askeladd's edge was entirely this
> config difference.

Frieren then ran 60+ more commits polishing this: ensembles of chained
low-LR fine-tunes at `slice=64` and `slice=128`, landing on a 6-way weighted
ensemble `iter153` at test 34.41 — and *widened* the lead to >8 points in
the final hour.

### Alphonse: identical diagnosis, different wording

Alphonse reached the same conclusion independently, a few hours after
frieren, driven by a direct hypothesis about slice-weight distribution
shift:

> **v11-fullmesh-breakthrough (MAJOR WIN).** Training has been using
> `train_max_points=80_000` random subsample while val/predict use the full
> mesh (up to ~240K nodes). That's a distribution mismatch — the
> PhysicsAttention slice weights are computed on a different density of
> nodes between train and val. Warm-start from v10 with `train_max_points=0`
> (no subsample, full mesh) and `batch_size=2` should close this gap.
>
> **22% over v10 in one run (4.23→3.30), 52.6% over v2 baseline.** Biggest jump
> since v3. Split-level gains are dramatic on the OOD tracks: val_geom_camber_rc
> 2.98→1.70, val_re_rand 2.70→1.25, val_geom_camber_cruise 1.57→0.58.
> val_single_in_dist barely moved — so the subsampling was mostly hurting OOD
> generalisation.
>
> HUGE learning — I should have tried full-mesh training much earlier. The
> hypothesis that subsampling "just acts like a regulariser" was wrong in this
> domain; it creates a real train/eval distribution gap for attention models
> that pool over the node set.

Alphonse then found a secondary insight — that dropping LR "too aggressively"
in the full-mesh regime looked like convergence but wasn't:

> **v14-fullmesh-lr1e5-b (LR restart win).** v12/v13 used lr=5e-6/3e-6 and
> their gains were flattening. Maybe the LR was *too* low, not the
> model-was-converged. Try going *back up* to lr=1e-5 (same as v11) and
> see if it's still training… Gain 0.07 over v13 — bigger than the last two
> rounds combined.

### Edward: third independent rediscovery

Edward hit the same wall around v12 (plateauing at 67) and made the same
decision, inferred from reading askeladd's W&B runs rather than frieren's
journal:

> **v13 full-mesh fine-tune (lr=5e-6).** Leader askeladd trains without
> subsampling (full mesh). Sub40k was great for pre-training, but fine-tuning
> at full mesh preserves fine surface detail that's essential for pressure
> MAE… 67.57 → 56.94 (−15.7 %). Cruise, rc and re_rand all dropped ~25 %.
> Full mesh + long-tail fine-tune is the unlock. 9 epochs at full mesh beats
> 35 at sub40k in this regime.

### Who did not find it, and what happened to them

- **Thorfinn** kept pushing the fine-tune chain at ever-lower LR and
  documented the plateau explicitly: *"val trajectory — 85.63 → 80.38 →
  78.95 → 77.92 → 74.20 → 72.09 → 71.69 → 71.83. Clean logarithmic decay;
  we hit the architecture's capacity floor."* He never tested whether the
  floor was the subsampling. Dropped from rank 3 to rank 6.
- **Tanjiro** never tried full-mesh; found a different aha (removing
  `surf_p_weight` after reading nezuko's journal dropped val 91 → 78) but
  couldn't break below 68 on test.
- **Nezuko** went hard in the opposite direction — tuning `surf_weight`
  down from 10 → 5 → 3 → 1.5 ("let the volume loss dominate") — and
  achieved −21% on val, but stayed at rank 8 on test.

## Other aha moments

### Nezuko: the epoch/cosine mismatch, formulated as a rule

Earlier in the run, nezuko found the cheapest single-line win of the day
(the starter's `epochs=50` cosine schedule was 82% unused when the 30-min
cap hit; changing it to `epochs=10` dropped val surf_p MAE by 10.4%) and
then explicitly formulated the counter-principle that constrained every
subsequent nezuko experiment:

> Rule for this codebase: **compute-per-epoch is the binding constraint** —
> any change that slows each batch needs a matching `epochs` adjustment,
> and usually nets negative.

Four subsequent nezuko experiments — Eidetic attention, `slice_num=128`,
EMA, a surface-specialist MLP head — all regressed for exactly this reason.
The lesson was real but it kept her away from the one change (removing
subsampling) that would have helped, because that change violates her own
rule by making each epoch ~4× slower.

### Edward: the opposite-direction fix of the same mismatch

Edward inverted nezuko's observation and reached the same conclusion from
the other side:

> Leader thorfinn set `epochs=200` so in the 30-min budget the cosine barely
> decays (≈88% of peak at ep32). Keep v2's config verbatim and change only
> `epochs=60 → 200`… **Avg mae_surf_p = 90.46** (v3 97.70). Keeping lr high in
> the valid regime matters much more than aggressive annealing in this setup.

So one agent found a big win by making the cosine decay *faster* to match
the budget, another by making it decay *slower* to keep peak LR — both
worked, because both corrected the same mismatch between schedule and the
30-min cap.

### Askeladd: principled decorrelation via loss geometry

Askeladd's chain ultimately failed to crack the top two (finished at 48.89,
rank 4), but the *method* was the most principled in the run. After
correlated-lineage ensembles regressed, askeladd deliberately changed the
loss geometry to force decorrelation:

> v6 pressure-focused fine-tune. Need a model with genuinely different error
> characteristics. Bias v6 toward pressure on surface specifically:
> `surf_weight 10→20, p_weight 1→2` effectively makes pressure-surface loss
> 4× more important during finetune.

Askeladd's later chain pushed `p_weight` through 2 → 3 → 5 → 10 → 20 → 8,
each time warm-starting the previous step and halving LR, finding a genuine
sweet spot at `p=10` rather than the "higher is better" prior that tanjiro
and fern followed past the optimum.

### Fern: L1 loss + dropout as an orthogonal path to #2

Fern took the longest alternative route to #2: L1 loss on surface pressure
directly, then EMA, then — when a clear train/val gap appeared at val 52 —
added attention dropout:

> **iter15-dropout01.** Big train/val gap (train vol=0.32 vs val ~1) implies
> overfitting — add dropout=0.1 to Transolver attention… val/avg_mae_surf_p =
> 49.70 (from 52.20). **Big −2.5 jump.**

Fern was the only agent who got to the top-4 without the full-mesh move as
the decisive lever (her final config still uses subsampling) — the dropout
unlock plus aggressive L1 + `surf_p_weight≥30` produced comparable gains.
She *then* added the full-mesh move on top and landed at 42.62, rank 2.

### Universal finding: same-lineage ensembles regress

Every agent that tried ensembling on a single warm-start chain reported
regression, and several explicitly formulated the rule:

- **alphonse**: "v6 alone 4.322 · v5+v6 4.349 · v6+v7 4.459 — all worse than
  v6 alone. Members have to be (a) individually close in quality, or (b)
  strongly decorrelated."
- **askeladd**: "v1 + v2 → 89.21, worse than v2 alone (85.22). v1 is a
  strictly-worse ancestor of v2 (warm-start lineage)."
- **frieren**: the 4-way ensemble iter18 "underperformed at 65.92 vs iter17
  single's 59.14. Lesson: only ensemble strong models."

Frieren eventually *did* make ensembling pay, once she had architectural
diversity (`slice=64` vs `slice=128` chains at different LRs): her final
6-way weighted ensemble at commit `c57076b` (34.41) sits ~1 point below her
best single model.

### The no-slip red herring

The starter `README.md` describes no-slip (`velocity = 0` on airfoil
surface) as a "hard physical constraint models should exploit". Three
independent agents (askeladd, tanjiro, thorfinn) inspected the data early
and caught the same discrepancy — `is_surface=True` marks inlet/outlet/walls
too, not just the airfoil — and disabled the constraint before it caused
harm. Nobody in the cohort ended up enforcing it.

## A framework flaw surfaced by the agents: the scorer race

A real scoring bug surfaced midway through the run, not by an agent
deliberately exploiting it but by symmetric bad luck between two agents and
the organiser's polling loop. `predict.py` writes the four `test_*.pt` files
sequentially (~300–500 MB each); the organiser polls every 60 s and stamps
any commit dir where a file is missing as `"incomplete"` in `scores.json`.
The "already scored" check then treats `"incomplete"` as terminal and
**never re-scores** once the missing file lands.

Every kaggler has at least one `incomplete` entry. Thorfinn and frieren both
hit it repeatedly and worked around it by forcing a fresh commit hash:

> Iter 6's submission is still marked `incomplete` in `scores.json` even
> though all 4 files are present and readable — probably a scorer race
> condition; not going to chase it. (thorfinn)

The agents correctly chose not to fight the scoring bug and to submit fresh
commits instead. For us on the harness side this is the same class of finding
as tanjiro's `val.pt` exploit in the GRaM run: a self-play experiment
surfacing a real framework flaw before it reaches a downstream evaluation.
Fix is straightforward — re-score any `incomplete` entries whose files are
now all present.

## What the framework demonstrates

- **Rediscovery of the same solution by independent agents, with different
  reasoning paths.** The full-mesh unlock was found by frieren, alphonse and
  edward within a ~3-hour window, with three different chains of reasoning
  (inference from askeladd's bs=2, direct hypothesis about PhysicsAttention
  slice-weight distribution shift, observation of askeladd's W&B runs).
  That's evidence the harness gives agents enough observational latitude to
  diagnose framework-level issues without collusion.

- **Convergence to one recipe → fracture at a breakthrough → one agent runs
  away.** The first ~8 hours look like classic cohort convergence: all eight
  agents drift toward a near-identical config and the leaderboard compresses.
  The defining moment is a *single* breakthrough (full-mesh) rediscovered
  independently; whichever agent found it first and iterated on it the most
  wins. Frieren committed 98 times, 2-3× more than anyone else, almost all
  of them incremental polish on top of the iter93 breakthrough.

- **Failure analysis is consistent and principled.** Agents routinely quote
  specific per-split metrics where a change regressed and explain *why* in
  physical or optimisation terms (nezuko's compute-per-epoch rule; alphonse's
  slice-weight distribution argument; askeladd's decorrelation-via-loss-
  geometry; frieren's diagnosis of re_rand weakness as a subsampling
  artefact).

- **Framework flaws get surfaced.** The scorer-race bug is the apr23
  equivalent of tanjiro's val-as-test exploit in the GRaM run — caught in
  self-play rather than in downstream evaluation.

- **A human-legible research record.** All eight journals totalling ~1100
  lines, 223 organiser-side scored leaderboard updates, and 330 kaggler
  commits are rebuildable from the repo alone. Every claim in this document
  is traceable to `origin/apr23/kaggler/<name>:tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md`
  or `origin/apr23-leaderboard`. The "commit journal separately from code"
  rule means failed experiments (Huber loss, EMA under tight budgets, Eidetic
  attention, slice_num=128, input noise, correlated-lineage ensembling) are
  retained just as thoroughly as the successes that produced the final
  leaderboard.

## Acknowledgements

The winning submission (frieren's `iter153` at commit `c57076b`) is entirely
the product of the `frieren` agent's own iteration: 98 commits across the
warm-start chain, the full-mesh breakthrough (iter93), the slice_num=128
diversity model (iter33/47), and the final 6-way weighted ensemble. No
manual edits. The narrative in this document is reproducible from
`origin/apr23/kaggler/frieren` and the scored predictions on the shared
volume.
