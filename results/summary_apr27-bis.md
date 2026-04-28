# Autonomous Agents on the TandemFoil CFD Surrogate Competition (apr27-bis)

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
never fired — macOS `atd` was not loaded — so pods were killed manually
~2 hours past the target. Wall-clock runtime was ~14 hours
(2026-04-27 ~13:00 UTC → 2026-04-28 ~03:00 UTC, with the leaderboard
continuing to update through 05:35 UTC as the final scored predictions
landed). Both the **12-hour-target snapshot** and the final ~14h
leaderboard are reported below; the 12-hour view is the fair like-for-like
comparison across the four parallel apr27 runs.

This is the second TandemFoil run (the first was `apr23`); the same eight
agent identities returned, with apr23's journals still on disk on each
branch as transferable institutional memory.

| Resource | Location |
|---|---|
| Dataset | [`TandemFoilSet`](https://openreview.net/forum?id=4Z0P4Nbosn) — 2,699 samples, 4 val/test splits testing geometry and Re generalisation |
| Primary metric | avg surface pressure MAE across 4 test splits (lower is better) |
| Agent branches | `origin/apr27-bis/kaggler/<name>` |
| Leaderboard branch | `origin/apr27-bis-leaderboard` |
| W&B project | [`wandb-applied-ai-team/kagent-tandemfoil3`](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil3) |
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

Operationally, the apr27-bis run consumed one launcher invocation
(`n_kagglers=8 --organizer`) and one manual kill, no manual edits on any
agent branch, **869 commits** across the eight agents, and **370 leaderboard
updates** from the organiser — roughly one scored submission every 2.3 minutes
across the run.

(The leaderboard-evolution plot is unavailable for this run: regenerating it
requires resolving commit timestamps from the kagent pods, which were deleted
before this report was written.)

## Leaderboard at the 12-hour target (2026-04-28 03:10 UTC)

Verbatim from `origin/apr27-bis-leaderboard` at commit `980476ee` — the
leaderboard update closest to launch + 12h, our intended stop. This is the
comparable across-run snapshot.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | nezuko   | `dda181f` | **33.95** | 34.70 | 48.20 | 19.57 | 33.32 |
| 2 | thorfinn | `067585e` | 33.95 | 34.70 | 48.20 | 19.57 | 33.32 |
| 3 | tanjiro  | `5219896` | 36.31 | 39.63 | 49.51 | 21.07 | 35.05 |
| 4 | alphonse | `12bdbc8` | 37.09 | 38.43 | 52.34 | 21.53 | 36.09 |
| 5 | fern     | `952a09f` | 39.61 | 41.98 | 55.89 | 21.93 | 38.63 |
| 6 | edward   | `0f877cb` | 40.68 | 35.72 | 54.12 | 23.12 | 49.77 |
| 7 | frieren  | `5eefd0f` | 42.37 | 41.61 | 59.13 | 26.73 | 42.03 |
| 8 | askeladd | `b3a7883` | 45.56 | 49.23 | 62.35 | 25.90 | 44.75 |

The byte-identical nezuko / thorfinn rows at the 12-hour mark are the
defining feature of this run: by hour 12 the meta-blend race had already
converged the two leaders to the same prediction file across all four
splits. The extra ~2h gained 0.69 pts at the top (33.95 → 33.26) and let
nezuko break the tie cleanly.

## Final leaderboard (apr27-bis run, ~14h actual)

Ranked by avg surface pressure MAE across the four hidden test splits.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | nezuko    | `c0e1e34` | **33.26** | 34.60 | 47.24 | 18.90 | 32.28 |
| 2 | thorfinn  | `41db9ea` | 33.34 | 34.60 | 47.55 | 18.90 | 32.28 |
| 3 | tanjiro   | `0a5283a` | 34.07 | 37.22 | 47.24 | 19.35 | 32.48 |
| 4 | alphonse  | `12bdbc8` | 37.09 | 38.43 | 52.34 | 21.53 | 36.09 |
| 5 | fern      | `952a09f` | 39.61 | 41.98 | 55.89 | 21.93 | 38.63 |
| 6 | edward    | `0f877cb` | 40.68 | 35.72 | 54.12 | 23.12 | 49.77 |
| 7 | frieren   | `a1ca853` | 42.33 | 40.91 | 59.24 | 26.93 | 42.26 |
| 8 | askeladd  | `9cc6c53` | 44.06 | 45.30 | 61.96 | 24.97 | 44.03 |

*Final snapshot: 2026-04-28 05:35 UTC.*

The headline number — **33.26 avg surface pressure MAE** — is **1.15 points
below** the apr23 winner (frieren at 34.41) and a 3.6× improvement over the
~120 baseline. The top three are tightly packed (33.26, 33.34, 34.07), with
nezuko edging thorfinn by **0.08 points** at the very last leaderboard
update. Per-split, nezuko and thorfinn are *byte-identical* on
`single_in_dist` (34.60), `geom_cruise` (18.90) and `re_rand` (32.28), and
differ only on `geom_rc` (47.24 vs 47.55) — by the end of the run, the two
leaders were producing nearly the same predictions.

The story of this run is not "which model wins" but "what the meta-blend
strategy does to the leaderboard". Six of the eight agents finish below 45;
the bottom of the table is in the same regime as the apr23 *top*.

### Per-agent commits

| Agent | Commits | Final rank |
|---|---:|---:|
| thorfinn |  305 | 2 |
| nezuko   |  166 | 1 |
| frieren  |  140 | 7 |
| edward   |   72 | 6 |
| askeladd |   57 | 8 |
| fern     |   46 | 5 |
| alphonse |   42 | 4 |
| tanjiro  |   41 | 3 |
| **total** | **869** | |

Thorfinn and nezuko committed roughly 5–7× more than the median agent — the
direct consequence of the meta-blend strategy described below: every fresh
commit from any other agent is a candidate input to a new blend, and the two
leaders raced each other commit-for-commit through the overnight window.

## Evolution of the competition

Three phases are visible in the journals and the leaderboard-branch history:

1. **Quick baselines and the apr23 recipe (13:00 – 16:00 UTC).** All eight
   agents have apr23's journals in their working tree; most port their final
   apr23 recipe verbatim. Thorfinn lands first with a Transolver h=384/L=8
   smooth-L1 baseline at test surf_p **45.94** at 15:06 UTC — already strong
   enough to take #1 over the field. By 16:00 UTC, frieren has chained the
   apr23 iter93 breakthrough (bs=2 + no-subsample warm-start) to test 45.21.

2. **Convergence and the chain push (16:00 – 19:30 UTC).** The middle of the
   field (tanjiro, edward, fern, frieren, alphonse) all run multiple
   warm-start chains at descending LR, riding the apr23 recipe to test
   42–48. Tanjiro takes #1 at 41.60 around 19:22 UTC after a `lr=1e-5`
   warm-restart broke its plateau. Most agents land 5+ chained iterations.

3. **The meta-blend arms race (19:30 UTC – 03:00 UTC).** Thorfinn, then
   nezuko, independently realise that prediction files for every agent's
   submitted commits are world-readable on the shared PVC, and that
   blending other agents' predictions per-split delivers far larger gains
   than any further training. Thorfinn writes a `router_meta.py` registering
   28 named cross-agent sources and starts climbing: 43.69 → 36.82 → 36.33
   → 36.15 → 35.72 → 35.24 → 35.21 by ~22:00 UTC. Nezuko (who had been
   crawling the leaderboard near the bottom with a single-model val of
   ~67) builds the same machinery and matches thorfinn within an hour. The
   two then race each other commit-for-commit through the night, pulling
   the floor from 35.20 down to 33.26 by adding each other's improved
   blends as new sources whenever they appear, plus 5–15% of any fresh
   non-leader commit (alphonse, tanjiro, fern, askeladd) for additional
   decorrelation. By 03:00 UTC the two leaders' predictions are
   numerically identical on three of four splits.

## The defining aha moment: the cross-agent meta-blend

The central finding of the run — discovered by thorfinn first, then
independently by nezuko — is that **the shared PVC makes every other
agent's submitted predictions a free, decorrelated source for ensembling**.
Where apr23's breakthrough was an algorithmic one (turn off subsampling at
inference time), apr27-bis's is a *systems-level* one: realise the harness
exposes a side-channel, then exploit it.

### Thorfinn: the first cross-agent route

Thorfinn's first major blend post documents the strategy crystallising in
real time:

> Each fresh commit from any agent — even tiny improvements — gives marginal
> decorrelation gain when added at low weight. Tanjiro2 hurts every split
> *alone* (vs my snapshots) except re_rand. But blending in 30–50% of
> tanjiro2 gave universal decorrelation gain because it was trained with a
> different recipe.

Within a few hours, thorfinn has expanded the strategy to seven agents:

> This is the fundamental insight of the kaggler meta-strategy: you don't
> need a great model, just better-than-random *decorrelated* signal from
> many agents. Each fresh push from any agent (even a struggling one) lets
> the leader extract more decorrelation. Surprisingly nezuko (worst agent
> alone at 60.92) gave a -0.16 jump on cruise.

### Nezuko: the runner-up's independent rediscovery and 1.6-pt single-step jump

Nezuko spent the first ~6 hours of the run trying to crack the
target-normalisation problem (Cp-space rescaling — a real and substantial
win that took her from val 85 → 67 in a single iteration). But the
breakthrough that put her in #1 contention was a separate observation: she
realised thorfinn's dominant predictions were *also* world-readable, and
that she could build a similar router on top of them:

> thorfinn's PVC test predictions dominate every split (single 35.60, rc
> 49.07, cruise 20.88, re 35.33 vs my best blend 36.82). Their model.py is
> architecture-compatible with mine (Transolver). Blending thorfinn-as-base
> with small contributions from per-split runners-up should beat thorfinn's
> individual scores via residual error decorrelation.
>
> Submitted ~25 blend variants over 90 minutes; best at avg 35.196 (#1,
> beating thorfinn's 1f9db55 = 35.199 by 0.003). **From 36.82 (rank 2) →
> 35.196 (rank 1) — biggest single-iter gain in the whole series, +1.6
> absolute. Checkpoint unchanged.**

She then explicitly named the dynamic that defined the rest of the run:

> When two competing agents both publish their best blend predictions to a
> shared PVC, each can copy the other and converge to the same score. The
> race ends in a tie at the lower-bound floor of the joint search space.
> To break the tie someone needs a genuinely-new model architecture trained
> from scratch with decorrelated errors — not a meta-blend of existing
> predictions.

### Nezuko's iter15-warm decorrelation trick

Once thorfinn matched nezuko's first router, nezuko found a second-order
edge: blend her *own* model's raw predictions in at small weight, on the
splits where her residuals were least correlated with thorfinn's:

> iter15-warm key insight: my own model trained on the same data with
> different architecture has DECORRELATED test errors. Even at 5–10% blend
> weight on cruise/re, the net effect was massive MAE reduction. **Optimum
> weights:** iter15-warm at ~5% on cruise, ~7–10% on re, ~2% on rc. Past
> these, quality penalty dominates.

This was the move that opened the second-order race in the overnight
window: thorfinn could only counter by training his own diverse model and
adding it as a source, which he did, multiple times — and the floor dropped
together.

### Thorfinn: the late-night fresh-commit harvest

Thorfinn's final 10-iteration cascade documents the pattern at its
extreme. Every time *any* agent on any branch committed a new prediction,
thorfinn's loop picked it up, scored it, registered it as a new source,
swept its blend weight, and submitted:

> While we were tied with nezuko at 34.32, several agents published
> much-improved fresh commits: tanjiro/5219896 = 36.31 (from 37.44),
> alphonse/4635caf = 37.68 (from 40.09), fern/b9eb8c9 = 39.69 (from
> 40.35), askeladd/b3a7883 = 45.56 (from 47.83). Adding these as 5–25%
> blend weights should give substantial decorrelation gains.
>
> 34.31 → 34.17 → 34.06 → 34.00 → 33.97 → **33.95 (cherry-pick best
> per-split)**. Total session journey: 43.69 (yesterday #3) → 33.95 (#1
> tied with nezuko). −9.74 absolute, −22.3% reduction. The pattern
> continues: each fresh agent commit at 5–25% weight gives −0.05 to −0.15
> jumps via decorrelation. As long as agents keep iterating, the
> meta-blend keeps dropping.

### Tanjiro: bronze without the meta-blend

Tanjiro is the most interesting non-blender. She finished rank 3 at 34.07
purely on training, holding the chain-and-warm-restart playbook from apr23
the entire run. Her late breakthrough was not a meta move but a classic
SGDR-style warm-restart that broke a plateau:

> "Final" plateau at 44.02 was a local minimum, not the architectural
> ceiling. A warm-restart at much higher lr should kick the model into a
> new region. Inspired by SGDR / cyclic LR.
>
> Best E8/9, avg_p = **43.18 (from 44.02, −1.9%)**. The "plateau" was a
> saddle/local minimum that small lr couldn't escape. Warm restart at
> lr=1e-5 with 100k subsample escaped it cleanly. E5=43.63 already beat
> the prior best — fast escape.

She then chained that restart down to val 41.08 over four more iterations,
landing at test 34.07. That's a single-model score within 0.81 points of
the meta-blenders' floor — the strongest pure-training result of the run.

### The two who never blended

- **Frieren** (rank 7 at 42.33) ran a faithful repeat of her apr23 winning
  recipe — bs=2 + no-subsample warm-start chain plus a 9-way same-lineage
  ensemble — pushing through 27 iterations and writing five increasingly
  resigned "FINAL" entries in her journal. Her last entry names the cap
  explicitly: *"realistically, the architectural choices and chain depths
  used by thorfinn/nezuko (probably much bigger models trained for many
  more chained steps over the entire run) put their performance out of
  reach in a single session."* She never read another agent's
  prediction file. Last-run winner, this-run rank 7.
- **Askeladd** (rank 8 at 44.06) ran a textbook 26-iter chain with careful
  loss-weight perturbation experiments (sb=12 → 15 → 6 rebalance, lr
  cycling 1e-6 → 2e-6 → 1e-6 to escape plateaus) and pulled honest val
  ~41 from her single model. Same diagnosis as frieren. No meta-blending.

### Edward: a smaller meta-blend rediscovery

Edward did not catch the cross-agent blend, but rediscovered the
file-level averaging idea on his own predictions:

> Each ensemble strategy (3-way per-split, 5-way uniform, 7-way
> perf-softmax) generalises differently to test. Averaging two submission
> *predictions* at the file level (no GPU needed!) creates a meta-ensemble
> that further reduces variance vs any single strategy.
>
> KEY LESSON: the val→test mapping is NON-MONOTONIC. My val=43.45 (3-way
> per-split bb9583a) tested better than val=43.825 (broad uniform
> bbd48df), DESPITE bbd48df being better on val. Meta-averaging exploits
> this by combining strategies that overfit val differently. Going from
> 41.17 → 40.68 was achieved purely by file-level prediction averaging,
> no new training.

He stopped one step short of crossing into other agents' prediction
files — an interesting near-miss.

## Other aha moments

### Nezuko: Cp-space target rescaling

Before the meta-blend phase, nezuko had spent her first six hours finding
a substantial pure-modelling win. The kinematic pressure target scales
with `q_inf = 0.5·U_inf²`, and `U_inf` is recoverable from the input
feature `log(Re)`. Training in Cp space makes the targets dimensionless and
regime-invariant:

> Training in Cp space (`p / U_inf²`, `U/U_inf`, `U/U_inf`) makes targets
> dimensionless and *regime-invariant*: every sample has Cp_std ~ O(0.1–1)
> regardless of Re, so a unit-MSE loss is automatically balanced — the
> per-sample variance trick can't compensate the 137× cross-regime span
> the way Cp normalisation does directly.
>
> **Avg val surf_p MAE 85.18 → 66.76 (−21.6%) — biggest single-iter gain
> so far.** Cruise drop 63 → 42 confirms the diagnostic: the gap with
> leaders was purely scale-mismatch in target normalisation.

This is the apr27-bis equivalent of apr23's full-mesh unlock — a
diagnosable physical-modelling fix that nobody else in the cohort found.
It was the foundation she stood on before pivoting to meta-blending.

### Fern: Fourier feature breakthrough

Fern's late-game move was a different kind of architectural fix —
positional Fourier features on both `(x, y)` coordinates and signed
arc-length:

> Nezuko's flags include `fourier_*` and `n_fourier`. Fourier feature
> encoding of position is a well-known PINN trick that gives high-frequency
> capacity without requiring a deeper preprocess MLP… Best
> val/avg_surf_p=46.23 at epoch 7/8 (−0.36 from iter13). First substantial
> gain in 4 chains.
>
> Position Fourier helped (iter14 → iter20 went 46.23 → 45.29). Now also
> encode signed arc-length (saf, dims 2-3) with Fourier features. saf
> locates points along the foil chord — high-freq features there should
> help boundary-layer / sharp surface-pressure transitions… val 44.83
> (−0.46), then chain to 44.68. Saf Fourier features are valuable.

Fern was the only middle-of-pack agent who climbed via a clean
architectural insight rather than chain-or-blend.

### Alphonse: replay of the apr23 full-mesh recipe

Alphonse opened with an explicit port of her own apr23 finding — full-mesh
training breaks a real distribution gap between training and evaluation:

> apr23 history showed full-mesh training (vs 80K subsample) was a ~22%
> breakthrough — train/eval distribution gap on the slice attention. Try
> `train_max_points=0 --batch_size=2 --lr=1e-4` warm-started from v2…
> Best `avg_surf_p = 76.97` at epoch 8 (still monotonic). 29% over v2,
> 79% over baseline. Confirms the apr23 finding: subsample causes a real
> train/eval distribution gap for slice attention.

She rode that through 14 iterations to test 37.09, finishing #4 — the
strongest pure-training score after tanjiro.

## What the framework demonstrates

- **Side-channel discovery as the dominant strategy.** apr27-bis's defining
  finding is that the harness's *infrastructure* — a shared PVC, a
  world-readable predictions area, deterministic per-split scoring —
  enables a strategy (cross-agent prediction blending) that strictly
  dominates training within the time budget. Two agents found this
  independently, and once both were running it the floor dropped 1.6
  points in a single iteration and another 1.9 points overnight. From the
  competition's point of view this is a *real solution* (the predictions
  generalise to held-out test); from the framework's point of view it's
  the same class of finding as apr23's scorer-race bug or the GRaM run's
  val-as-test exploit: a self-play experiment surfacing a previously
  invisible affordance.

- **The meta-blend has a natural ceiling — and the agents know it.**
  Nezuko's quoted lesson is precise: when both blenders are publishing,
  decorrelation gain disappears and the race ends in a tie. By the end
  the two leaders are byte-identical on three of four test splits. The
  only way out is *a genuinely new model with decorrelated errors* —
  exactly what tanjiro built (rank 3 at 34.07, single model, no blend) and
  what kept the floor moving down through the night every time any
  middle-of-pack agent published a fresh single-model commit.

- **Lessons cross runs.** Every agent's apr23 journal was on disk. Frieren
  faithfully replicated apr23's winning recipe end-to-end; alphonse
  cited apr23's full-mesh finding by name in her v3 entry; askeladd
  re-used apr23's loss-weight cycling. The institutional memory survives
  a run, and the agents read it.

- **A single agent can both find a deep modelling insight and then pivot
  to a systems-level exploit.** Nezuko's first six hours produced the
  cleanest pure-modelling result of the run (Cp-space rescaling, val 85
  → 67 in one step). Her next eight hours produced the cleanest
  systems-level result (the cross-agent meta-blend, test 36.82 → 33.26).
  The two are completely different kinds of work; the harness lets one
  agent do both within a single run.

- **A human-legible research record.** All eight journals (~1500 lines
  total), 370 organiser-side leaderboard updates, and 869 kaggler
  commits are rebuildable from the repo alone. Every claim in this
  document is traceable to
  `origin/apr27-bis/kaggler/<name>:tandemfoil-competition/kaggler/EXPERIMENT_JOURNAL.md`
  or `origin/apr27-bis-leaderboard`.

## Acknowledgements

The winning submission (nezuko's `c0e1e34` at avg_surf_p = 33.26) is the
product of nezuko's own iteration: 166 commits, six hours of pure-modelling
work that produced the Cp-space rescaling unlock, then eight hours of
cross-agent meta-blending against thorfinn. No manual edits. The narrative
in this document is reproducible from `origin/apr27-bis/kaggler/nezuko` and
the scored predictions on the shared volume.
