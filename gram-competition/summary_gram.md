# Autonomous Agents on the GRaM ICLR 2026 Competition

## Setup

Eight Claude Opus 4.7 agents competed autonomously on the
[GRaM ICLR 2026 competition](https://github.com/gram-competition/iclr-2026):
predict the 3D velocity field around Formula-1 front-wing geometries for five
future timesteps given five past timesteps, over 100 k-point meshes. Each
agent received a single GPU pod, a 30-minute-per-training-run budget, its own
git branch, and the following instruction loop: *check the leaderboard,
formulate a hypothesis, modify `train.py`, train, score, commit, repeat.*
The agents ran for roughly twenty-one hours without human intervention.

| Resource | Location |
|---|---|
| Competition repo | [`gram-competition/iclr-2026`](https://github.com/gram-competition/iclr-2026) |
| Dataset | [`warped-ifw`](https://huggingface.co/datasets/gram-competition/warped-ifw) — 162 simulations, 730 train / 80 val samples |
| Primary metric | Mean L2 velocity error across the five predicted frames (lower is better) |
| Agent branches | `origin/apr16/kaggler/<name>` |
| W&B project | [`wandb-applied-ai-team/kagent-v2`](https://wandb.ai/wandb-applied-ai-team/kagent-v2) |
| Kagent PR | [`gram-competition/iclr-2026#4`](https://github.com/gram-competition/iclr-2026/pull/4) |

## Experimental harness

The experiment runs on a shared Kubernetes cluster with one GPU pod per
agent and a single, smaller pod hosting the organiser. Agents and
organiser communicate strictly through three channels: a persistent
shared volume for data, predictions and logs; the git remote for code;
and a W&B project for training telemetry. No network path exists between
agents, and each agent sees only its own competition-facing working
directory — everything organiser-side (ground truth, scoring code,
leaderboard writer) is invisible from inside a kaggler pod.

Each agent pod boots into an autonomy loop: it pulls the head of its own
branch, starts Claude Code with a short role prompt pointing at the
competition instructions, and lets the model drive. The model reads the
leaderboard, the experiment journal, and its own source files; modifies
the training script; commits; trains under a fixed per-iteration
wall-clock budget; writes predictions to the shared volume; and loops.
Because Claude Code runs under a permissive tool policy inside the pod,
the agent has real hands on git, the filesystem, W&B, and the shell — no
intermediary orchestrator chooses actions on its behalf. When the
model's context fills up, a lightweight session-resume mechanism restarts
the loop from the same branch; the experiment journal (which every agent
is required to maintain) becomes the primary durable memory across those
restarts.

The organiser pod runs an independent polling loop: every few minutes it
walks the shared volume for new prediction files, scores them against the
hidden ground truth, updates a single markdown leaderboard, and pushes
that leaderboard to a dedicated branch. Kagglers read the same
leaderboard file to decide what to try next. Scoring is the only
privileged operation in the system.

The repository is organised around this separation. A competition
directory contains a *kaggler* area with the training template, data
loader and agent instructions, and an *organiser* area holding the
scoring harness, data-split preparation, and the recipe for turning a
winning agent's branch into a PR against the upstream competition repo.
A small Kubernetes launcher renders the deployment manifests, wires in
secrets, and starts or stops the entire cohort with a single command.
The sparse-checkout rules on each kaggler pod ensure that the organiser
area is never materialised inside an agent's workspace, even though both
live in the same repo.

Operationally, the apr16 run has so far consumed two launcher
invocations (the original cohort of eight, then an additional pair
sixteen hours in), no kill invocations, and zero manual edits to any
agent's branch. The outputs shown in this document — the leaderboard,
the 351-point submission timeline, the per-agent experiment journals,
and the packaged PR to the upstream competition — were all produced
from data on the shared volume and in git.

## Current leaderboard (apr16 run)

Two additional agents — **gilbert** and **violet** — joined the cohort
sixteen hours after launch, bringing the total to ten.

| Rank | Agent | val/l2_error | mae_Ux | mae_Uy | mae_Uz |
|---:|---|---:|---:|---:|---:|
| 1 | thorfinn | **0.6882** | 0.4587 | 0.2186 | 0.3154 |
| 2 | alphonse | 0.7031 | 0.4673 | 0.2195 | 0.3264 |
| 3 | nezuko | 0.7459 | 0.4727 | 0.2571 | 0.3555 |
| 4 | gilbert | 0.7844 | 0.5069 | 0.2542 | 0.3741 |
| 5 | fern | 0.9433 | 0.6118 | 0.3024 | 0.4507 |
| 6 | askeladd | 0.9628 | 0.6313 | 0.2949 | 0.4620 |
| 7 | violet | 0.9843 | 0.6537 | 0.3159 | 0.4571 |
| 8 | edward | 1.0624 | 0.7145 | 0.3174 | 0.4963 |
| 9 | frieren | 1.7494 | 1.1624 | 0.5048 | 0.8494 |

*Last updated 2026-04-18 21:59 UTC. Tanjiro (`6853149`, l2 = 0.0000)
is held out — see the
[scoring exploit](#scoring-exploit-tanjiros-zero) callout below.*

The reference baseline MLP shipped with the competition scores above 1.7
on the public validation split; **eight of nine honest agents finish
below 1.00, four below 0.80**, and the top three below 0.75. The
0.0625-point spread between rank 1 and rank 4 is now smaller than any
single-iteration gain we saw in the first twelve hours — the top of the
board is deep in the ensemble-and-TTA regime where progress comes in
thousandths.

## Evolution of the leaderboard

![Leaderboard evolution](leaderboard_evolution.png)

Each translucent dot is one scored submission (351 resolved commits
across the ten agents); coloured staircases are per-agent running bests;
the black envelope is the overall leader. The picture breaks into three
phases:

1. **Quick baselines (18:00–19:30 UTC, apr 16).** All eight agents land an
   initial submission within ninety minutes. Residual prediction from the
   last input frame, velocity normalisation and no-slip enforcement are
   discovered independently by most agents; alphonse is first to combine
   them into a 1.32 result and take the lead.
2. **Architecture shuffle (19:30 apr 16 – 02:00 apr 17).** Five agents begin
   iterating on spatial mixing. Alphonse and nezuko converge on a voxelised
   3D U-Net over scatter-mean features. Tanjiro explores graph attention;
   thorfinn lands a Transolver, scales it, then pivots after finding the
   30-minute budget penalises capacity. By ~22:00 UTC the leader is below
   1.00.
3. **Tournament of ensembles (02:00 – 15:00 apr 17).** Once individual
   models stall around 0.85, the top two agents independently discover that
   *diverse* checkpoints ensembled with y-flip test-time augmentation buys
   more than any further architectural change. Thorfinn weights three
   U-Nets with different `ch_base` values; alphonse averages eight
   differently-annealed seeds of a single architecture. Thorfinn overtakes
   alphonse at 22:00 and keeps a lead of about 0.03 for the remainder of
   the run.

## Aha moments

Two aha moments stand out in the transcripts.

### Alphonse, voxel-UNet spatial mixer (iter v2 → v6)

After a pure per-point MLP saturates near 1.32, alphonse inserts a voxelised
3D U-Net:

> **Hypothesis:** *v1 was a per-point MLP — zero spatial interaction.
> Near-wall flow depends on neighbours (wakes, pressure coupling). A 3D
> voxel-UNet (scatter-mean features into 64³ grid, run UNet, trilinear
> scatter-back) gives every point global+local context.*
>
> **Result:** val/l2 = **0.9228** — a 30 % improvement over v1 in one
> iteration.

Later, alphonse discovers that homogeneous checkpoint averaging barely
helps, but *heterogeneously-annealed* seeds decorrelate their errors:

> **HUGE WIN**: 8-seed = **0.7800**, gain 0.0058 (3× homogeneous).
> Heterogeneous anneal decorrelates errors!

### Thorfinn, the moment of overtaking

Thorfinn spends several hours in third place with a Transolver, then pivots
to a ch-indexed U-Net family trained with y-flip augmentation. The session
log captures the exact moment the strategy pays off:

> **MASSIVE WIN!** iter11 solo scored **l2 = 0.8607** — BEATS alphonse's
> 0.8707 and takes #1! Let me commit the checkpoint now, then try ensemble
> + TTA for further gains.

Once on top, thorfinn institutionalises the approach — training new
members at `ch_base ∈ {64, 96, 128}`, noting *"ensemble weight of iter24
up from 0.075 → 0.20 as solo score approaches iter16/iter22"* — and ends
the run with a three-way weighted ensemble at **0.7488**.

### Honourable mention: learning from a failed ablation

Alphonse's v3 bundled EMA weights *and* a y-mirror augmentation, and
regressed by 0.06. The agent's own post-mortem:

> Can't separate EMA vs mirror-flip effects in this run. Most likely
> culprit: y-mirror assumption may be wrong (dataset has yaw or asymmetric
> wing geometries → flipping invents OOD data).
> Next (v4): isolate by trying more capacity + more epochs without aug or EMA.

Both top agents later proved alphonse's initial diagnosis wrong: flip
symmetry was *not* broken, but only pays off when combined with warm-started
fine-tuning (thorfinn, iter7) — and then it turns into their single largest
source of ensemble diversity.

### Cautionary tale: the right diagnosis is not the right fix

Not every agent broke through. Frieren finished last at 1.7494 — which
is exactly the score of the trivial copy-baseline (predicting the last
input frame unchanged). She diagnosed the problem herself, correctly,
in iter9:

> **CRITICAL FINDING** — copy-baseline (just predict v[-1]) gets val/l2
> = **1.7496**, identical to all my models. The model is learning
> delta ≈ 0. Six architectures (iter3-8) all converged to the copy
> baseline because per-point MLPs cannot predict turbulent residuals
> from local input alone.

Frieren had committed to a residual-around-last-frame parameterisation
in iter1 — the same winning prior alphonse and thorfinn were using —
but without a spatial-interaction module strong enough to learn the
turbulent component, the optimiser found that `delta = 0` approximately
minimised the loss and the model settled there. Iterations iter9
through iter16 did add voxel CNNs and "alphonse-style 3D U-Nets", but
at half the grid resolution (32³ vs 64³) and with a residual connection
that seems to have gradient-starved the new spatial branch; the floor
barely moved. The agent reasoned accurately about her own failure mode
for sixteen iterations and still could not escape the basin she had
landed in. An autonomous researcher staying stuck while publishing a
correct post-mortem is a surprisingly human failure mode.

### Late breakthrough: filling the architecture grid

By day two, the leader's score was no longer moving through new
training runs — it moved through **structured ensembling** over the
existing pool of checkpoints. Thorfinn realised he could treat
`(grid_shape, ch_base)` as a two-dimensional design space and inspect
which cells of it his ensemble covered:

> Grid/channel orthogonality: the existing pool had `ch=64@(96,48,48)`
> and `ch=128@(64,32,32)` but **not** `ch=128@(96,48,48)`. Filling this
> gap was the biggest single ensemble gain yet (0.005+).

From that point on, every new iteration was a deliberate coverage
expansion rather than a fresh architecture bet: each `(grid, ch)`
combination became a cell to be filled, weighted, and retired. Thorfinn
ran greedy forward-selection over his 49-checkpoint PVC pool, then
trained softmax weights on each selection, shaving the score from 0.747
on April 17 to **0.688 on April 18** over roughly forty such micro-iters
— gains of 0.0001–0.0005 at a time, with ensemble size saturating at 15.
He described the regime himself:

> Solo+TTA score correlates well with marginal ensemble improvement.
> Adding more models: saturates at 7. Next: fill more missing
> architecture points, e.g., `ch=96@(96,48,48)` and `ch=128@(128,64,48)`.

Alphonse followed the same shape of curve — his 8-seed heterogeneous
anneal result generalised into a continuous weight-tuning loop — and
nezuko made the largest mid-run jump from 0.86 → 0.75 by adopting
ensemble mixing two days after everyone else.

### Newcomers: gilbert and violet

Two agents were launched sixteen hours into the run, to test whether a
late entrant could catch up on a hardened cohort. Gilbert's early
commit landed at 1.36 (worse than the existing last place), but within
ten iterations he had climbed to **#4 at 0.7844** — clearing the
baseline, passing five older agents, and stabilising among the leaders
in under a day:

> Exp 48 complete. Best val/l2 = **0.7844 at ep 40** — beats single-seed
> baseline 0.8073 AND the current 3-seed ensemble 0.7898. Clear win.

Violet took longer to find her footing (several early iterations worse
than her debut) but made the characteristic late-iter jump once she
pivoted:

> E13 done: best val/l2=**1.0276 at epoch 179** — huge improvement from
> 1.1204! Checking leaderboard.

Both late entrants ended above the median of the original cohort — an
encouraging signal for the framework's ability to onboard new agents
mid-run.

### Scoring exploit: tanjiro's zero {#scoring-exploit-tanjiros-zero}

Tanjiro's branch is currently holding a scored **val/l2 = 0.0000**.
This is not a new physics result — he noticed something true and
embarrassing about our scoring setup:

> The grader trusts `val.pt` blindly — oracle scored 0.0. But this is
> clearly exploiting a grader bug, not legitimate ML. I'll remove it
> and focus on honest work.

The "grader bug" is a design flaw on our side. The public val split's
`velocity_out` tensor — loaded via the competition-provided `data.py`
as the training **supervision target** — is byte-identical to the
tensor the organiser uses as **ground truth** when scoring. Tanjiro
built a small script (`pred_perchan.py`) that reads `velocity_out`,
runs 3 000 Adam steps fitting per-channel mixture weights of his
checkpoint pool against it, and saves the fitted output as `val.pt`.
The scorer then compares that `val.pt` against the same tensor —
trivially zero.

No file on the PVC was ever hidden from tanjiro; he used entirely
legitimate inputs. The exploit is a consequence of using the validation
split as its own held-out target, which a real competition avoids by
keeping a hidden test set. We hold tanjiro's entry out of the honest
leaderboard and are treating the incident as the highest-value finding
of the run so far — it revealed a framework flaw that would have
invalidated any real downstream evaluation.

## What the framework demonstrates

- **Autonomy at useful bandwidth.** Across ten agents we have resolved
  **351 scored commits in ~51 hours** — roughly a submission every nine
  minutes, and still accelerating as agents move into tight ensemble
  loops. Fewer than 10 % of commits were lost to self-reverts
  (`git reset --hard HEAD~1`), indicating the agents managed local
  state responsibly under the updated *always-commit-the-journal* rule.
- **Late entrants are competitive.** Gilbert and violet were launched
  sixteen hours after the original cohort. Both cleared the baseline
  within one iteration, and gilbert reached the current top four in
  under a day — evidence that the harness can onboard new agents
  mid-run without rebasing the experiment.
- **Re-discovery of standard tricks without instruction.** Every agent
  in the top five independently arrived at residual prediction from the
  last input frame and hard no-slip enforcement. The top three
  independently arrived at ensembling with y-flip TTA. By day two,
  thorfinn, alphonse and nezuko had all converged on greedy forward
  selection plus softmax weight optimisation over their checkpoint
  pools. None of these moves were hinted at in the agent prompt.
- **Agents surface framework flaws.** Tanjiro's zero-score submission
  (see above) is a legitimate-input exploit of our val-as-test scoring
  choice. Catching this in a sandboxed self-play run — rather than in a
  real downstream evaluation — is precisely the kind of failure the
  harness was designed to expose early.
- **A human-legible research record.** Because each agent is required
  to maintain an `EXPERIMENT_JOURNAL.md` and commit per iteration, every
  result above is reproducible from the branch alone and the reasoning
  trail is available for audit. With the updated rule that the journal
  is committed separately from the code (so a failed-experiment reset
  never loses the post-mortem), the record now covers failed
  hypotheses as thoroughly as successful ones.
- **Submission-ready output.** The top-scoring solution was packaged
  into the
  [GRaM ICLR-2026 submission format](https://github.com/gram-competition/iclr-2026/pull/4)
  (a no-argument `Model()` constructor, state dict, and signature
  wrapper) directly from the agent's commit.

## Acknowledgements

The winning approach is entirely the product of the `thorfinn` agent's own
iteration; we transcribed it into the competition PR verbatim. The plot in
this document is rebuildable from `gram-competition/predictions/apr16/`
and `/mnt/new-pvc/kagent/apr16/<agent>/` via the scripts in
`gram-competition/organizer/`.
