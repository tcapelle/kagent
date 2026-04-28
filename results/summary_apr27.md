# Autonomous Agents on the TandemFoil CFD Surrogate Competition (apr27)

## Setup

Eight Claude Opus 4.7 agents competed autonomously on our internal CFD surrogate
benchmark built on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn)
dataset: given 24-dim per-node features of a 2D overset mesh around one or two
airfoils (position, signed arc-length, shape descriptors, Reynolds number,
angle-of-attack, NACA profile, gap/stagger), predict the full
`(Ux, Uy, p)` field at every mesh node. Each agent received a single GPU pod,
a 30-minute-per-training-run budget, its own git branch, and the standard
instruction loop: *check the leaderboard, formulate a hypothesis, modify
`train.py`, train, score, commit, repeat.* The cohort was scheduled to stop
at **12 hours**, but the timed kill (an `at` job on the operator laptop)
never fired — macOS `atd` was not loaded, so the queued kill silently
expired. The pods were killed manually at ~14 hours
(2026-04-27 15:11 UTC → 2026-04-28 05:34 UTC). Both the **12-hour-target
snapshot** and the final ~14-hour leaderboard are reported below; the
12-hour view is the fair like-for-like comparison across runs.

| Resource | Location |
|---|---|
| Dataset | [`TandemFoilSet`](https://openreview.net/forum?id=4Z0P4Nbosn) — 2,699 samples, 4 val/test splits testing geometry and Re generalisation |
| Primary metric | avg surface pressure MAE across 4 test splits (lower is better) |
| Agent branches | `origin/apr27/kaggler/<name>` |
| Leaderboard branch | `origin/apr27-leaderboard` |
| W&B project | [`wandb-applied-ai-team/kagent-tandemfoil2`](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil2) |
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

Operationally, the apr27 run consumed one launcher invocation
(`n_kagglers=8 --organizer`) and a manual kill at ~05:35 UTC (the scheduled
12h kill failed to fire — see Setup), no manual edits on any agent branch,
**422 commits** across the seven active agents (nezuko's apr27 branch is
empty — see below), and **192 leaderboard updates** from the organiser —
roughly one scored submission every 4.5 minutes for 14 hours.

*Leaderboard-evolution plot: not regenerated for this run — the apr27 pods
were torn down before commit-timestamp resolution could be done from the live
PVC, and the existing plot script depends on per-pod state.*

## Leaderboard at the 12-hour target (2026-04-28 03:12 UTC)

Verbatim from `origin/apr27-leaderboard` at commit `8a72673c` — the leaderboard
update closest to launch + 12h, our intended stop. This is the comparable
across-run snapshot.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | alphonse | `6f8f043` | **25.53** | 23.95 | 42.00 | 12.67 | 23.51 |
| 2 | askeladd | `6b2a0a8` | 32.07 | 29.93 | 45.02 | 19.69 | 33.65 |
| 3 | frieren  | `defe1da` | 32.49 | 30.83 | 48.95 | 17.93 | 32.27 |
| 4 | tanjiro  | `248dd0e` | 34.04 | 35.52 | 48.61 | 18.42 | 33.62 |
| 5 | thorfinn | `3f10f4b` | 35.80 | 38.31 | 50.35 | 20.53 | 34.02 |
| 6 | thorfinn-test | `00c70c0` | 42.90 | 49.17 | 58.61 | 24.52 | 39.31 |
| 7 | fern     | `bd2caf2` | 51.64 | 38.24 | 67.03 | 33.39 | 67.88 |
| 8 | nezuko   | `feca7e3` | 58.02 | 57.16 | 86.87 | 30.44 | 57.60 |
| 9 | edward   | `94495c7` | 84.08 | 48.34 | 122.40 | 45.67 | 119.91 |

Alphonse already led at the 12-hour mark, immediately after her predict.py
decoder bug fix (~04:00 UTC) had reordered the top half of the table.
The Cp + Huber recipe and the bug-fix scramble are described in detail
below; the run-defining work was done in the 12h window.

## Final leaderboard (apr27 run, ~14h actual)

Ranked by avg surface pressure MAE across the four hidden test splits.
Verbatim from `origin/apr27-leaderboard:leaderboard.md` at the manual-stop
snapshot (2026-04-28 05:34 UTC). The extra ~2h beyond the 12h target
gained alphonse 1.13 pts (25.53 → 24.40) on the run's existing recipe; no
ranks changed inside the top five.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | alphonse | `b3812ec` | **24.40** | 22.88 | 40.98 | 11.53 | 22.19 |
| 2 | tanjiro  | `02bf3ca` | 31.13 | 32.29 | 45.36 | 16.46 | 30.43 |
| 3 | frieren  | `776eade` | 31.43 | 29.21 | 48.23 | 16.90 | 31.38 |
| 4 | askeladd | `6b2a0a8` | 32.07 | 29.93 | 45.02 | 19.69 | 33.65 |
| 5 | thorfinn | `66b382d` | 35.22 | 37.41 | 49.98 | 19.94 | 33.54 |
| 6 | thorfinn-test | `00c70c0` | 42.90 | 49.17 | 58.61 | 24.52 | 39.31 |
| 7 | fern     | `93b9291` | 47.96 | 36.03 | 63.97 | 32.70 | 59.15 |
| 8 | nezuko   | `feca7e3` | 58.02 | 57.16 | 86.87 | 30.44 | 57.60 |
| 9 | edward   | `94495c7` | 84.08 | 48.34 | 122.40 | 45.67 | 119.91 |

The "thorfinn-test" entry is a stale carry-over baseline from a prior
calibration commit and is not an active apr27 agent — the competing thorfinn
agent is rank 5 at 35.22.

The baseline Transolver scores ~120 on this metric; **eight of the nine
ranked entries finish below 60**, **five below 36**, and the winner at
**24.40** is roughly a 5× improvement over the baseline. The top four
finish within 7.7 points of each other, then there is a clear step to
thorfinn (35.22), and edward — who tapped out before the breakthrough — is
the lone tail at 84.

Alphonse's lead over second place (tanjiro) is **6.73 points**, and the
margin grew rather than shrank in the final hour: when alphonse's predict.py
bug fix landed ~04:00 UTC, every score in the top half of the table shifted
by 5–10 points and reordered the ranking, with alphonse jumping from a
mid-pack 32–33 to the new floor of 24.40. Alphonse is the best on every test
split.

### Per-agent commits

| Agent | Commits | Final rank | Notes |
|---|---:|---:|---|
| alphonse | 140 | **1** | predict.py decoder bugfix at iter30 unlocked the entire Cp+Huber chain |
| tanjiro  |  96 | 2 | cycle (high-LR refresh ↔ bs=2 fine-tune) over 9 cycles |
| frieren  |  39 | 3 | warm-restart chain with single/tandem domain boosts |
| askeladd |  40 | 4 | discovered Cp normalisation + Huber-0.1 (the recipe alphonse later fixed) |
| thorfinn |  46 | 5 | full-mesh fine-tune, channel-weighted loss, surface-only training |
| fern     |  38 | 7 | NeRF-style Fourier features + Re-aware pressure normalisation |
| edward   |  23 | 9 | tapped out at 21:49 UTC after FiLM-on-Re experiments |
| nezuko   |   0 | 8 | branch never moved past `origin/main`; commit `feca7e3` referenced on the leaderboard is not present in any pushed apr27 ref |
| **total** | **422** | | |

Nezuko is the run's anomaly: she has a leaderboard entry (rank 8, 58.02) and
an empty `EXPERIMENT_JOURNAL.md` with zero commits ahead of `origin/main`.
Either her predictions were submitted via a since-orphaned commit (the SHA
`feca7e3` cited on the leaderboard is unreachable from any current ref) or
her pod scored a single attempt early and never iterated again. The
apr27-bis / apr27-4 / apr27-5 sibling branches show nezuko was active in
parallel sessions, but those are separate experiments outside this report.

## Evolution of the competition

Four phases are visible in the journals and the leaderboard-branch history:

1. **Quick baselines (15:11 – 16:30 UTC).** All active agents land a working
   Transolver-based submission within ~90 minutes. Frieren, thorfinn,
   alphonse and tanjiro converge fast on the apr23-derived recipe
   (`192×6 / slice=64`, L1 loss, bf16 autocast, surface-aware subsampling at
   40K volume nodes, surf_weight≈10–20). First scored entries land in the
   80–115 range.

2. **Recipe convergence + the bs=2 / no-subsample carry-over (16:30 – 19:30
   UTC).** Multiple agents already know about frieren's apr23 iter93 unlock
   (full-mesh + bs=2 warm-start chains) and apply it on day one rather than
   rediscovering it: tanjiro's iter2 quotes the recipe by name, fern adopts
   it implicitly, frieren iterates her own chain forward from the prior
   run's checkpoint. Test scores compress into the 38–50 range. Thorfinn
   takes rank 1 briefly at 41.19 with the chain-only recipe.

3. **The Cp normalisation + Huber-0.1 unlock (19:30 – 22:30 UTC).** Askeladd
   independently derives a structural change that nobody from apr23 had
   tried: per-sample Re² rescaling of pressure targets (a Cp-style
   nondimensionalisation) plus Huber loss with `delta=0.1` to align the
   loss shape with the L1 leaderboard metric. Iter6 (Cp alone) takes rank 1
   at 39.16; iter8 (Cp + Huber-0.1) takes rank 1 at 32.07 — a 7-point jump
   on top of 3 points — and holds rank 1 for several hours. Thorfinn,
   tanjiro and frieren respond by tightening their existing chains,
   improving by 1–3 points per iter but visibly running out of head-room
   below 33.

4. **The predict.py decoder breakthrough (22:30 UTC – 05:35 UTC).**
   Alphonse re-derives Cp + Huber-0.1 from askeladd's W&B traces (her
   journal calls it "post-mortem from competitor research") and chains it
   for five iterations, watching val/loss drop from 0.82 to 0.44 — a
   trajectory that should imply test scores in the low 20s. Every test
   submission instead scores ~200, an obviously-physical-units mismatch.
   At iter30 she discovers the cause: her `predict.py` was decoding the
   model's Cp-normalised output back to physical pressure with the *raw*
   pressure stats, not the Cp-normalised stats stored in `runtime.yaml`.
   A one-line fix and a re-run of every queued submission lands
   alphonse at **26.71**, then **24.40** with iter35 chain step, jumping
   from mid-pack to rank 1 by ~6 points. Fern in parallel finds her own
   structural unlock (Re-aware pressure normalisation at iter21) and
   improves 10.5% in a single iter, but never closes the gap to the top
   four because she started the run further back. The cohort closes with
   alphonse out alone at 24, three agents tightly clustered at 31–32, and
   the rest in the 35–58 band.

## The defining aha moment: the Cp normalisation + Huber-0.1 recipe

The central finding of the apr27 run is that **Cp-style Re² rescaling of the
pressure target plus a sharp Huber loss (delta=0.1) is a step-function
better recipe than the L1 chain that had won apr23**. It was discovered
twice, once correctly (askeladd) and once with a silent decoder bug
(alphonse), and the agent who eventually found and fixed the bug took the
top spot.

### Askeladd: the original derivation

Askeladd is the only agent in the run who reasoned the recipe from physics
rather than copying it from another agent's traces:

> Pressure scales as ρU² ∝ Re² in kinematic units. Test sets re_rand and
> geom_cruise involve out-of-distribution Re — globally normalizing
> pressure mixes vastly different scales (std 17 to 304 across regimes).
> Per-sample dividing y_p by exp(2·(log_re − log_re_ref)) decouples scale
> from shape: model predicts a roughly-O(1) Cp; physical p reconstructed
> at output via re_factor multiplication.
>
> **Test avg_surf_p=39.16 — RANK 1**, beating frieren (42.11) and thorfinn
> (42.90).

She then noticed that her own loss-shape was already nearly MSE on a
post-Cp pressure target, and went one step further:

> Iter 6's training surf loss is ~0.005 in normalized space — typical
> errors are well below huber_delta=1, so the loss is essentially MSE.
> Lowering huber_delta to 0.1 makes the loss almost-pure-L1 for typical
> errors, which directly matches the MAE test metric and reduces sensitivity
> to outlier nodes.
>
> **Test avg_surf_p=32.07 — RANK 1 by 6 points.** Per-split test:
> single=29.93, geom_rc=45.02, geom_cruise=19.69, re_rand=33.65 — beating
> every other agent on all splits.

### Alphonse: the breakthrough was hidden behind a decoder bug

Alphonse picked up the recipe by reverse-engineering askeladd's W&B runs,
implemented it independently, and then chained it harder than askeladd did
— but for hours her test scores looked catastrophic while her val/loss kept
dropping cleanly:

> **Hypothesis (post-mortem from competitor research):** askeladd jumped
> from 39.16 → 32.07 via two structural changes — (1) Cp-style Re²
> pressure normalization and (2) Huber loss with delta=0.1. Both are
> loss-shape changes; together a 7-pt jump on top of Cp norm's 11-pt jump.
> My L1 chain stalled at val 1.02 (iter20) → surf_p ~37 ceiling. Need
> to switch recipe.

Six iterations later, with val/loss at 0.44 (versus iter17's val 1.11 →
test 38.13), her test submissions were still scoring around 200. She
diagnosed it at iter30:

> **RANK 1 at 26.71 — fixed predict.py bug, regenerated everything.**
>
> **Bug found:** my `predict.py` was using the RAW `y_mean[2]/y_std[2]`
> from `stats.json` (raw pressure stats) when decoding model output,
> instead of the Cp-normalized stats `p_mean_cp/p_std_cp` saved in
> `runtime.yaml`. Model trained correctly to predict Cp-normalized values,
> but inference decoded those values back to physical pressure with the
> wrong scale, producing surf_p ~200 (vs 26-30 expected).
>
> Critical lesson — when using non-trivial output normalization, ALWAYS
> verify the inverse transform end-to-end on the test pipeline, not just
> the training metrics. The model was fine; the decoder was wrong.

The fix was a single conditional in `predict.py`. Once it landed, every
backed-up Cp+Huber prediction in the queue rescored simultaneously and
alphonse jumped from outside the top half straight to rank 1. iter35
(another chain step on top) took her to **24.40**, a clear 6-point gap
above tanjiro and frieren.

### Fern: parallel rediscovery of the Re-aware target rescaling

Fern, who had been stuck on a Fourier-feature plateau in the high 60s,
arrived at a closely related insight from a different angle:

> The dominant remaining error was on `val_re_rand` and `geom_camber_rc`.
> Pressure scales roughly with `Re^k` (k between 1 and 2). Dividing
> pressure targets by `(Re/Re_ref)^k` should remove this systematic
> Re-dependent variance, leaving the model to predict a Re-invariant
> pressure that we then multiply back at inference.
>
> **Best epoch 9 (EMA) mean=55.12 (-10.5% vs iter20 61.61). All four
> splits improved**, and the two hardest splits (geom_camber_rc and
> re_rand) improved the most, exactly as the hypothesis predicted.
> Biggest single-iteration improvement of the entire run.

Fern landed her version at iter21, less aggressively than askeladd's full
Cp + Huber package, and finished at 47.96 — outside the top of the
leaderboard but with a textbook rediscovery trajectory.

## Other aha moments

### Thorfinn: surface-only loss, late but real

Thorfinn spent most of the run on a long warm-restart chain at decreasing
LRs and hit the same plateau every other chain agent did. Late in the run,
having watched alphonse's surge, he finally formulated the surface-vs-volume
loss alignment correctly:

> **iter25 surface-only training (KEPT, real gain).** Leaderboard ranks
> only on surface pressure MAE; volume gradients are wasted capacity. Drop
> volume loss entirely (`vol_weight=0`) so the backbone allocates all
> gradient signal to surface nodes.
>
> Best epoch 11, val avg_surf_p=40.21 (Δ -0.65 vs iter22). Surface-only
> loss is the right alignment with the metric — should have done it
> earlier.

Thorfinn finished at 35.22, rank 5 — never made the Cp jump.

### Tanjiro: cycle-of-cycles as a chain optimiser

Tanjiro discovered, through nine consecutive iterations, that
*alternating* between two fine-tune regimes (a high-LR / batch=4 / sub=60K
"refresh" and a low-LR / bs=2 / no-subsample fine-tune) compounds where a
single chain saturates:

> **iter25: cycle-5 HIGH-LR refresh — NEW BEST 36.96.** Maybe the model
> is stuck in a local minimum. A higher LR (5e-5 vs prior 1e-5) cycle
> should "shake" the weights into a different/better basin.
>
> Confirms intuition: when chain saturates at low LR, a higher-LR refresh
> can escape the basin.

This carried tanjiro through nine cycle pairs, ending at iter34 with
val/loss 0.77 and test score 31.43, rank 2 — without ever adopting Cp
normalisation. The cycle pattern is essentially the apr23 frieren/askeladd
warm-restart chain at higher fidelity.

### Edward: tapped out before the breakthrough

Edward spent his 6.5 hours of activity on architectural changes — Fourier
features, FiLM-on-Re conditioning, snapshot ensembling — and never shifted
the loss recipe. His best run (iter8) reached val 83.49, test 84.08, rank
9. He stopped committing at 21:49 UTC, hours before the Cp + Huber + bug
discoveries that defined the run.

### Frieren: domain-boosted weighted sampling

Frieren extended her own apr23 warm-restart-chain methodology with an
explicit per-domain reweighting in the sampler — every two iters she
"warm-restarts" with a higher LR cycle and bumps `single_boost` and
`tandem_boost` to push the loss harder on the splits where she was lagging:

> **iter20: chain step + balanced boosts (6, 4) — MASSIVE WIN.**
> Per-test breakdown: single=30.62 (askeladd 29.93), geom_rc=49.49
> (askeladd 45.02), geom_cruise=19.03 (better than askeladd 19.69),
> re_rand=33.34 (better than askeladd 33.65). Beating askeladd on 2 of
> 4 splits!

She reached test 31.43 on the final scored commit (iter26), rank 3, on a
chain that never adopted Cp normalisation — the domain boost was her way
of compensating for the metric/loss mismatch.

### Universal finding: same-lineage ensembles regress (again)

As in apr23, every chain-only ensemble regressed:

- **frieren**: "iter11 prediction averaging across chain ckpts: val_avg_surf_p
  = **45.45** vs iter9's 44.73 — averaging worsens. Chain is too correlated
  for averaging to help in either weight or output space."
- **tanjiro**: "iter17: 3-way ensemble iter10/iter14/iter16 — **39.10**,
  slightly worse than iter16 alone (38.60). Confirms chain checkpoints have
  correlated errors."
- **thorfinn**: "SWA experiment: 4-way ckpt average → val=42.25, worse than
  iter17 alone. Chain trajectory is not on a flat basin."
- **askeladd**: chain ensembles rejected; the only ensemble that may help is
  iter 8 (Cp + Huber 0.1) plus iter 33 (same recipe + dropout regularisation)
  — different solutions, not different chain points. Outcome pending at
  scheduled stop.

## Framework observations

- **Carry-over of prior-run findings.** Apr23's full-mesh / bs=2 unlock was
  not rediscovered — multiple agents quote it explicitly from the previous
  run within the first hour. The competition-summary skill output and
  README baselines successfully transmit prior structural insights to the
  next cohort. The new structural insight in apr27 (Cp normalisation +
  Huber-0.1) is genuinely novel for this codebase and arrived from a
  physics-first analysis by askeladd.

- **Bug-as-breakthrough.** Alphonse's win was gated entirely on diagnosing
  her own decoder bug. Until iter30 her test submissions silently scored
  ~200 while her training metrics looked world-class — exactly the failure
  mode the prediction-pipeline check is supposed to catch. This is a
  realistic Kaggle failure: training infrastructure can be fine while the
  inference path destroys the result. Worth lifting "verify the inverse
  transform end-to-end" into the boilerplate `predict.py`.

- **A documented blank-branch case (nezuko).** Nezuko's apr27 branch is
  empty. She has a leaderboard entry, but its commit hash is unreachable
  from any pushed ref. Possible causes — pod startup failure that left
  her predicting from a stale `model.py`, or session-resume that got
  redirected onto an apr27-bis/4/5 sibling — are not recoverable from the
  git history alone. Apr23 had no equivalent failure. This is the cleanest
  framework gap surfaced this run; "every kaggler should produce at least
  one journal entry per hour" would have caught it.

- **Multiple parallel sessions.** The remote also contains
  `apr27-bis-leaderboard`, `apr27-4-leaderboard`, `apr27-5-leaderboard`
  branches with the same eight kagglers and disjoint commit histories.
  Those are separate, shorter sessions outside the scope of this report;
  noting them here so future archaeology can correlate.

- **A human-legible research record.** The seven non-empty journals total
  ~1100 lines, the organiser produced 192 scored leaderboard updates, and
  422 kaggler commits are rebuildable from the repo alone. Every claim in
  this document is traceable to
  `origin/apr27/kaggler/<name>:tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md`
  or `origin/apr27-leaderboard`.

## Acknowledgements

The winning submission (alphonse's `iter35` at commit `b3812ec`) is entirely
the product of the `alphonse` agent's own iteration: 140 commits including
the Cp-normalisation + Huber-0.1 implementation, the warm-restart cycle,
the predict.py decoder bug-fix, and the final regenerated test predictions.
No manual edits. The narrative in this document is reproducible from
`origin/apr27/kaggler/alphonse` and the scored predictions on the shared
volume.
