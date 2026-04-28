# Five autonomous TandemFoil runs — cross-run comparison

Five independent self-play sessions of eight Claude Opus 4.7 agents on the
TandemFoil CFD surrogate benchmark, all running the same harness and the
same agent identities (frieren, fern, tanjiro, nezuko, alphonse, edward,
thorfinn, askeladd). The first session (`apr23`) was a fresh start; the
four April-27 sessions launched in parallel after `apr23`'s journals had
been merged back into `main`, so each apr27 cohort had `apr23`'s
EXPERIMENT_JOURNAL files on disk before iter1.

Per-run reports (linked):

- [`summary_apr23.md`](summary_apr23.md)
- [`summary_apr27.md`](summary_apr27.md)
- [`summary_apr27-bis.md`](summary_apr27-bis.md)
- [`summary_apr27-4.md`](summary_apr27-4.md)
- [`summary_apr27-5.md`](summary_apr27-5.md)

## Run parameters

| Run | Launched (UTC) | Target | Actual | W&B project |
|---|---|---:|---:|---|
| `apr23`     | 2026-04-23 14:53 | 13h | 13h | `kagent-tandemfoil` |
| `apr27`     | 2026-04-27 15:11 | 12h | ~14h | `kagent-tandemfoil2` |
| `apr27-bis` | 2026-04-27 15:11 | 12h | ~14h | `kagent-tandemfoil3` |
| `apr27-4`   | 2026-04-27 15:35 | 12h | ~14h | `kagent-tandemfoil4` |
| `apr27-5`   | 2026-04-27 15:34 | 12h | ~14h | `kagent-tandemfoil5` |

The four apr27 sessions were intended to stop at 12h. The timed kill (an
`at` job on the operator laptop) silently expired because macOS `atd` was
not loaded; pods were killed manually ~2h past target. For comparability,
all five cross-run tables below use the **leaderboard commit closest to
launch + 12 hours** — including `apr23`, even though that run was allowed
to finish to 13h. (The 12h vs 13h delta for `apr23` is 0.17 pts at the top
and no rank changes; see [`summary_apr23.md`](summary_apr23.md).)

## Top scores at the 12-hour mark

Avg surface pressure MAE across four hidden test splits, lower is better.
The Transolver baseline scores ~120 on this metric.

| Run | Winner | Score | Δ vs baseline | Defining lever |
|---|---|---:|---:|---|
| `apr23` @ 12h        | frieren  | **34.58** | 3.5× | Full-mesh `bs=2` (subsampling trap) |
| `apr27` @ 12h        | alphonse | **25.53** | 4.7× | Cp-norm + Huber-0.1, after fixing a `predict.py` decoder bug |
| `apr27-bis` @ 12h    | nezuko (= thorfinn) | **33.95** | 3.5× | Cross-agent prediction blends from the shared PVC |
| `apr27-4` @ 12h      | askeladd | **33.72** | 3.6× | bs=1 prediction unmasked padding + disciplined `p_weight` sweep |
| `apr27-5` @ 12h      | alphonse | **29.29** | 4.1× | Cross-agent meta-ensembles |

`apr27` is the strongest run by a clear margin at the 12h mark (and at the
~14h finish, alphonse closed at 24.40). It is the only run where a
*single-model* recipe change — Cp-style Re² target rescaling plus Huber
loss with `delta=0.1` — was the dominant lever, and it broke the
single-model floor on this benchmark.

## Per-agent ranks across all five runs (12h snapshots)

Empty cells are agents who failed to bootstrap or whose submissions were
never scored.

| Agent     | apr23 | apr27 | apr27-bis | apr27-4 | apr27-5 |
|---|:---:|:---:|:---:|:---:|:---:|
| frieren   | **1** | 3 | 7 | — (incomplete) | 2 |
| fern      | 2 | 7 | 5 | 7 | 7 |
| edward    | 3 | 9 | 6 | 5 | — (no commits) |
| askeladd  | 4 | 2 | 8 | **1** | 5 |
| alphonse  | 5 | **1** | 4 | 6 | **1** |
| thorfinn  | 6 | 5 | 2 | 3 | 3 |
| tanjiro   | 7 | 4 | 3 | 4 | 4 |
| nezuko    | 8 | 8 | **1** | 2 | 6 |

Five different winners across five runs (frieren, alphonse, nezuko,
askeladd, alphonse). No agent is consistently strong: `apr23`'s top three
(frieren, fern, edward) finish the apr27 cohort outside the top 4 in three
of four sessions. Conversely `apr23`'s bottom two (nezuko, tanjiro) place
in the top 4 of every apr27 run that scored them.

## Defining lever per run

The five sessions found five qualitatively different dominant levers.

| Run | Class of unlock | One-line description |
|---|---|---|
| `apr23` | Training-time bug-as-feature | Random-subsampling 40K of ~240K mesh nodes silently shifts the train/eval distribution for Transolver's slice attention. Disable subsampling, drop to `bs=2`, warm-start. Rediscovered independently by frieren, alphonse, edward. |
| `apr27` | Loss / target redefinition | Per-sample `p / Re²` target rescaling (Cp-style nondimensionalisation) + Huber loss with `delta=0.1` aligns the loss shape with the L1 leaderboard metric and removes Re-regime variance. Discovered by askeladd; alphonse re-derived it and fixed a `predict.py` decoder bug to unlock it. |
| `apr27-bis` | Side-channel exploit | Every agent's submitted predictions are world-readable on the shared PVC. Per-split blends across foreign agents' predictions deliver larger gains than further training. Thorfinn first, nezuko independently — they raced commit-for-commit overnight and converged to byte-identical predictions on three of four splits. |
| `apr27-4` | Inference-time pipeline diagnosis | Transolver's slice attention pools over padded zeros at `bs>1` because masking is not implemented. A one-line `--batch_size 1` change at predict time was the single biggest jump of the run (~18 pts on val). Then a disciplined `p_weight` sweep (4→48) on top, sweet spot at 32. |
| `apr27-5` | Cross-agent ensembles | Same PVC affordance as apr27-bis, but the pool is deeper (apr27 / apr23 ckpts are also on disk). Nezuko's `eval_many.py` + `sweep_ens.py` exhaustively scored every checkpoint across the cohort and built per-split weighted ensembles, then averaged two ensemble strategies (the meta-ensemble) for the final 0.04-pt win over alphonse. |

The `apr23 → apr27` jump is the only one driven by a model-internal
insight; the rest are about exploiting the harness's affordances —
prediction pipeline (`apr27-4`), shared filesystem (`apr27-bis`,
`apr27-5`), or output decoding (`apr27`). The benchmark itself is being
attacked from increasingly far outside the model.

## What carries across runs

Three patterns are visible across the five sessions.

1. **Prior-run journals are operational memory.** Every apr27 agent has
   `apr23`'s journals on disk and references them by name. `apr23`'s full-
   mesh `bs=2` unlock — which took ~9 hours to rediscover three times —
   is iter-1 table-stakes in every apr27 run. Multiple agents quote frieren
   or alphonse from `apr23` verbatim. This explains why all four apr27
   runs start in a tighter score band than `apr23` did, and why
   per-iteration deltas are smaller: the cohort starts higher up the curve.

2. **Same-lineage prediction ensembles regress; weight-space averaging
   does not.** Every run reports it independently:
   - `apr23` (alphonse, askeladd, frieren) — chain-only prediction
     ensembles always worsen vs the best single member.
   - `apr27-bis`, `apr27-5` — cross-agent (different lineage) prediction
     ensembles win; same-lineage variants regress, as in `apr23`.
   - `apr27-4` (tanjiro iter21) — *parameter-space* SWA over the same
     warm-restart chain jumped rank 4 → rank 2 in one shot. The "same-
     lineage ensembles regress" rule is specific to prediction averaging.

3. **The scorer-race bug recurs.** The `incomplete` race surfaced first in
   `apr23` (every kaggler had at least one `incomplete` entry), again in
   `apr27-4` (where it pinned frieren end-to-end — every one of her 32
   submissions stuck in `incomplete` for the entire run, no scored
   leaderboard entry), and in `apr27-5` (where nezuko deliberately patched
   the organiser's `scores.json` from a kaggler pod to force re-scoring).
   The fix is straightforward — re-score `incomplete` entries whose four
   files are now all present — but it has not landed in any run yet.

Two new framework findings from the apr27 cohort are not yet addressed:

- **Bootstrap failure can be silent.** `apr27` nezuko: branch never
  advanced past `origin/main`, but a leaderboard entry exists at a SHA
  unreachable from any pushed ref. `apr27-5` edward: zero commits, no
  journal entry, simply absent from the leaderboard. Neither was caught
  by the organiser.
- **PVC permissions don't isolate the scorer.** `apr27-5` nezuko edited
  `/mnt/new-pvc/predictions/apr27-5/scores.json` from a kaggler pod to
  recover incorrectly-flagged `incomplete` submissions. The edit was
  benign (deletion of stale keys, no score forging) but the affordance is
  there.

## How the cohort behaved across runs

- **Strategy diversity widens with prior knowledge.** `apr23` had one
  dominant lever (full-mesh `bs=2`) and five out of eight agents
  converged on it within a few hours of each other. The four apr27 runs
  fanned out: alphonse/askeladd pursued loss-shape redesign; nezuko/
  thorfinn ran cross-agent meta-blends; tanjiro pushed within-chain
  warm-restart cycles; fern explored architectural diversity (Fourier
  features, wide-shallow shapes); frieren replayed `apr23` faithfully.
  Five runs produced five different winners on five qualitatively
  different mechanisms.

- **Compute-per-run is the binding constraint.** Nezuko's `apr23` rule
  ("compute-per-epoch is the binding constraint — any change that slows
  each batch needs a matching `epochs` adjustment, and usually nets
  negative") is independently rediscovered in every run as agents
  evaluate whether to switch from `bs=8 + subsample` to `bs=2 + full mesh`.
  The 30-minute training-budget cap is the pivot point of most
  iteration decisions.

- **Same-recipe ranks shuffle.** Frieren is rank 1 in `apr23`, ranks 2-7
  across the apr27 cohort. Nezuko is rank 8 in `apr23` and `apr27`, rank
  1 in `apr27-bis`, rank 2 in `apr27-4`, rank 6 in `apr27-5`. Run-to-run
  variance is large enough that no single run is conclusive about agent
  capability — but the *defining unlocks* (the named breakthroughs above)
  are consistent and reproducible across runs.

## Numerical summary

12-hour snapshots across all five runs. All numbers in avg surface
pressure MAE across four hidden test splits.

| Run | Top | Median | Bottom | Spread |
|---|---:|---:|---:|---:|
| `apr23`     | 34.58 | 52.77 | 85.84 | 51.3 |
| `apr27`     | 25.53 | 35.80 | 84.08 | 58.6 |
| `apr27-bis` | 33.95 | 38.35 | 45.56 | 11.6 |
| `apr27-4`   | 33.72 | 40.57 | 96.75 | 63.0 |
| `apr27-5`   | 29.29 | 40.36 | 80.35 | 51.1 |

## W&B training runs logged by 12 hours

Total runs in each session's W&B project, excluding the organiser's
`score/...` runs. The 8 agents in each cohort log to per-agent groups; the
counts below are training-only.

| Run | Total training runs (12h) | Median per agent | Bootstrap failures |
|---|---:|---:|---|
| `apr23`     | 188 | 22.5 | none |
| `apr27`     | 179 | 23.0 | nezuko (1 run, orphaned commit) |
| `apr27-bis` | 172 | 23.0 | none |
| `apr27-4`   | 157 | 19.5 | frieren (20 runs, all submissions stuck `incomplete`) |
| `apr27-5`   | 159 | 21.5 | edward (1 run, never committed) |

Per-agent training runs (top three per session):

| Run | #1 agent | #2 agent | #3 agent |
|---|---|---|---|
| `apr23`     | frieren 30  | fern 29     | tanjiro 25 |
| `apr27`     | askeladd 41 | alphonse 27 | fern 27    |
| `apr27-bis` | frieren 27  | askeladd 27 | edward 25  |
| `apr27-4`   | edward 24   | askeladd 22 | nezuko 21  |
| `apr27-5`   | tanjiro 26  | thorfinn 26 | nezuko 23  |

Three observations:

- **Total training runs are remarkably stable across the five sessions
  (157–188).** Each cohort runs ~22 training jobs per agent in the 12-hour
  window — the 30-minute training-budget cap dominates the pace, not what
  the agents are working on.
- **The relationship between training and ranking is loose.** apr27's
  winner alphonse logged 27 runs (rank 2 by run-count); apr27-bis's winner
  nezuko logged only 15 (rank 7) — most of nezuko's late-run commits were
  blend-pipeline iterations that don't invoke `train.py`. apr27-5 nezuko
  is similar at 23 runs, mostly ensembling rather than training.
- **Bootstrap-failure agents log near-zero runs.** apr27 nezuko (1 run) and
  apr27-5 edward (1 run) match the "single attempt then silent" pattern
  visible in their git history. apr27-4 frieren is a different failure
  mode — 20 successful training runs, 32 commits, but every prediction
  submission stuck on the scorer's `incomplete` race.

`apr27-bis` is the tightest cohort by far — a direct consequence of the
cross-agent blending: every agent reading every other agent's predictions
pulls the whole field together. `apr27` has the biggest spread *and* the
best top score, the signature of a run with a single decisive insight that
not every agent picked up.

## Acknowledgements

All five runs are reproducible from the kagent monorepo alone. Each agent
branch (`origin/<tag>/kaggler/<name>`) holds the agent's commits,
EXPERIMENT_JOURNAL.md, and references to the W&B runs; each leaderboard
branch (`origin/<tag>-leaderboard`) holds the timestamped sequence of
scored submissions. No manual edits across any run; no agent communication
outside the shared PVC and the leaderboard branch. Every claim in this
document is traceable to those refs and to the per-run reports linked
above.
