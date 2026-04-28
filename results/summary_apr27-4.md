# Autonomous Agents on the TandemFoil CFD Surrogate Competition (apr27-4)

## Setup

Eight Claude Opus 4.7 agents competed autonomously on our internal CFD surrogate benchmark built on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset: given 24-dim per-node features of a 2D overset mesh around one or two airfoils (position, signed arc-length, shape descriptors, Reynolds number, angle-of-attack, NACA profile, gap/stagger), predict the full `(Ux, Uy, p)` field at every mesh node. Each agent received a single GPU pod, a 30-minute-per-training-run budget, its own git branch, and the standard instruction loop: *check the leaderboard, formulate a hypothesis, modify `train.py`, train, score, commit, repeat.* The cohort was scheduled to stop at **12 hours**, but the timed kill (an `at` job on the operator laptop) never fired — macOS `atd` was not loaded — so pods were killed manually ~2 hours past the target. Wall-clock runtime was ~14 hours (2026-04-27 13:12 UTC → 2026-04-28 ~05:32 UTC). Both the **12-hour-target snapshot** and the final ~14h leaderboard are reported below; the 12-hour view is the fair like-for-like comparison across the four parallel apr27 runs.

| Resource | Location |
|---|---|
| Dataset | [`TandemFoilSet`](https://openreview.net/forum?id=4Z0P4Nbosn) — 2,699 samples, 4 val/test splits testing geometry and Re generalisation |
| Primary metric | avg surface pressure MAE across 4 test splits (lower is better) |
| Agent branches | `origin/apr27-4/kaggler/<name>` |
| Leaderboard branch | `origin/apr27-4-leaderboard` |
| W&B project | [`wandb-applied-ai-team/kagent-tandemfoil4`](https://wandb.ai/wandb-applied-ai-team/kagent-tandemfoil4) |
| Baseline (Transolver, `n_hidden=128, n_layers=5`) | published to each agent's `README.md` |

## Experimental harness

The experiment runs on a shared Kubernetes cluster with one GPU pod per agent and a separate, smaller organiser pod. Agents and organiser communicate strictly through three channels: a persistent shared volume for data, predictions and logs; the git remote for code; and a W&B project for training telemetry. No network path exists between agents, and each agent sees only its own competition-facing working directory — the organiser area holding ground truth and scoring code is invisible from inside a kaggler pod.

Each agent pod boots into an autonomy loop: pull the head of its branch, start Claude Code with a short role prompt pointing at the competition instructions, and let the model drive. The model reads the leaderboard, its own experiment journal and its source files, modifies the training script, commits, trains under the 30-minute wall-clock budget, writes predictions to the shared volume, and loops. When context fills, a lightweight session-resume mechanism restarts the loop from the same branch; the experiment journal (which every agent is required to keep and commit separately from code changes) is the durable cross-restart memory.

The organiser pod polls every 60 seconds, scores any new prediction files against held-out ground truth, updates a single markdown leaderboard, and pushes that leaderboard to a dedicated branch. Scoring is the only privileged operation in the system.

Operationally, the apr27-4 run consumed one launcher invocation (`n_kagglers=8 --organizer`) and one manual kill on 2026-04-28, no manual edits on any agent branch, **364 commits** across the eight agents, and **207 leaderboard updates** from the organiser. First scored entry landed ~17:46 UTC (~4h30m in); first leaderboard update was 15:35 UTC.

## Leaderboard at the 12-hour target (2026-04-28 03:34 UTC)

Verbatim from `origin/apr27-4-leaderboard` at commit `6632d32c` — the
leaderboard update closest to launch + 12h, our intended stop. This is the
comparable across-run snapshot.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | askeladd | `6b42d98` | **33.72** | 34.81 | 49.42 | 18.33 | 32.31 |
| 2 | nezuko   | `90df277` | 37.60 | 30.53 | 53.45 | 21.36 | 45.05 |
| 3 | thorfinn | `c3dc5b2` | 37.68 | 39.04 | 54.17 | 21.46 | 36.07 |
| 4 | tanjiro  | `ff7547b` | 40.57 | 34.91 | 52.12 | 25.50 | 49.75 |
| 5 | edward   | `a7dde30` | 41.08 | 43.89 | 55.86 | 24.75 | 39.83 |
| 6 | alphonse | `0ace418` | 49.19 | 55.69 | 65.11 | 29.42 | 46.51 |
| 7 | fern     | `0621c0b` | 96.75 | 62.98 | 139.18 | 49.71 | 135.13 |

apr27-4 is unique among the four parallel runs: askeladd's winning
submission (`6b42d98` at 33.72) was already on the board at the 12-hour
mark and held unchanged through the extra ~2h. The cohort behind her
tightened by 1–2 pts in the final hours (notably nezuko 37.60 → 36.60
and thorfinn 37.68 → 37.39), but the lead was decided in the first 12h.
Frieren's submissions remained `incomplete` for the entire run on both
snapshots — see "Stuck on the scorer" below.

## Final leaderboard (apr27-4 run, ~14h actual)

Ranked by avg surface pressure MAE across the four hidden test splits.

| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | askeladd | `6b42d98` | **33.72** | 34.81 | 49.42 | 18.33 | 32.31 |
| 2 | nezuko   | `0f5d3f1` | 36.60 | 30.38 | 51.86 | 20.86 | 43.28 |
| 3 | tanjiro  | `feab9ab` | 37.10 | 36.16 | 49.08 | 23.95 | 39.22 |
| 4 | thorfinn | `aa470ac` | 37.39 | 38.42 | 54.04 | 21.17 | 35.92 |
| 5 | edward   | `b595f15` | 41.04 | 43.82 | 55.79 | 24.74 | 39.81 |
| 6 | alphonse | `0ace418` | 49.19 | 55.69 | 65.11 | 29.42 | 46.51 |
| 7 | fern     | `5c33571` | 90.80 | 60.25 | 127.50 | 50.83 | 124.63 |
| — | frieren  | (no scored entry) | — | — | — | — | — |

*Final snapshot: 2026-04-28 05:32 UTC.*

The leaderboard cohort is **seven agents, not eight**. Frieren produced 32 training-iteration commits and a working ResMLP→FiLM→GatedSliceAttn ensemble chain at val ≈ 41.0, but **every one of her submissions hung as `incomplete` in `scores.json` for the entire run** — the same scorer-race class of bug that surfaced in apr23, but here it pinned a single agent end-to-end. She documented it explicitly in her journal but never broke through. Her best projected test (extrapolating her val→test gap of ~5 points) would have placed her around rank 3.

Among the seven scored agents, askeladd at **33.72** is the run winner — a fractional improvement over apr23's frieren-iter153 winner (34.41) and the best single-agent result we have on this benchmark so far. The top four (askeladd, nezuko, tanjiro, thorfinn) are tightly clustered in a 3.7-point band; edward sits 3.5 behind them; alphonse and fern lag in the 49 / 91 zones. The baseline Transolver scores ~120 on this metric, so **six of seven scored agents finish below 50**, and the top four below 38.

### Per-agent commits

| Agent | Commits since main | Final rank | Notes |
|---|---:|---:|---|
| askeladd | 37 | 1 | held #1 from iter25 (~03:00 UTC 04-28) onward; ensemble chain on `surf_p_weight` sweep |
| nezuko   | 50 | 2 | jolt/polish-cycle warm-start chain; 25 iterations, no reset |
| tanjiro  | 59 | 3 | most commits in the cohort; SWA over iter16-20 was the single-iter rank-jump |
| thorfinn | 51 | 4 | 20 iterations, train+val-mixing breakthrough at iter17 |
| edward   | 64 | 5 | broad ensemble-optimiser approach (16 source ckpts, fast_optimize.py) |
| alphonse | 30 | 6 | converged early on a slice-mix ensemble (iter4+iter7+iter18) and stalled |
| fern     | 41 | 7 | single best 320×4 wide-shallow but never broke 90 on test |
| frieren  | 32 | — | submissions all `incomplete`; never appeared on the leaderboard |
| **total** | **364** | | |

The plot of leaderboard evolution over time is **unavailable for this run**: all apr27-4 pods were deleted before the report was generated, and the organiser-side `leaderboard_evolution` script needs live pods to resolve commit timestamps to wall-clock training time.

## Evolution of the competition

Three phases are visible in the journals and the leaderboard-branch history:

1. **Slow start, no leaderboard for ~4h (13:12 – 17:46 UTC).** The organiser-side leaderboard branch existed but was empty until 17:46 UTC, which is 4h30m into the run. By that time most agents had finished iter1 (cold-start at val 100–200) and iter2 (warm-start chain at val 50–80). Edward at 43.73 was first on the board, with alphonse at 50.83 right behind and a long tail (askeladd 67.7, tanjiro 69.6, fern 105). This "no-feedback" early window explains why the cohort fanned out further than apr23 did before they could see each other's work.

2. **Convergence on the apr23 recipe (18:00 – 23:00 UTC).** With the leaderboard live, every agent reads each other's commits and journals from prior runs (`apr23`, `apr27`, `apr27-bis` — all on disk in the PVC from the previous evening's runs). Frieren's `bs=2 + no-subsample + warm-start` apr23 breakthrough is rediscovered explicitly by **edward, alphonse, askeladd, fern** within a couple of iterations of each landing their first checkpoint — they cite frieren by name in their journals. Most of the cohort converges on the same Transolver-192/6/64 + `p_weight=3` + bs=2 + full-mesh chain by ~22:00 UTC. Test scores band into the 39–50 range. By 22:05 UTC the leaderboard reads thorfinn 39.52, askeladd 39.55, nezuko 39.79, edward 41.28 — a 0.27 gap between the top three.

3. **The askeladd unlock (23:00 UTC – 04:00 UTC).** Late in the run, askeladd makes two consecutive moves that jump her from 39.55 to 34.56 in a single round (iter18: `train_subsample 40k → 60k`, then iter19/iter21 push it to 80k for denser per-step gradients) and **immediately invents a `p_weight` sweep** on top: 4→12→16→24→32→48 across iter22–iter31, each warm-started from the previous step. Each step gives 0.1–0.5 single-model points but, crucially, also produces an ensembleable basin: her final iter32 ensemble (iter31×2 + iter29×2 + iter25 + iter21 + iter18) lands at **33.72**, beating the apr23 winner. By 03:32 UTC askeladd has a 3.88-point lead and *widens* it in the final hour. The other top agents (nezuko, tanjiro, thorfinn) each find their own 1–2-point overnight unlocks, but no one matches the pressure-weight sweep cleanly enough to close the gap.

## The defining aha moments

### Askeladd: bs=1 prediction unmasked padding (the single biggest jump)

Iter9 is askeladd's incidental discovery — she set out to ensemble three checkpoints and instead found that the *prediction* batch size, not the training batch size, was capping her score. The Transolver's slice attention pools over all batch nodes including pad-collated zeros, and the slice-norm gets corrupted at `bs=2` on padded meshes:

> while implementing the ensemble I evaluated each ckpt on val with `batch_size=1` and got **massively** lower scores than reported during training. iter8 dropped from val=70.62 (training, bs=4) to **val=52.16** (bs=1). Why: batched samples are pad-collated to the largest mesh in the batch, and the Transolver's slice attention doesn't mask padding — so the slice-token aggregation gets noise from zero-padded nodes…
>
> a one-line predict-time change for a 26% drop in val MAE without retraining. The biggest win in the run so far.
>
> Leaderboard competitors are almost certainly *also* affected by this — their submitted predictions used whatever batch_size their predict.py shipped, and many probably ran at bs=2 too. So the absolute jump for me may translate to a relative jump vs everyone else.

This is the defining "look at the harness, not the model" moment of the run: askeladd diagnosed a *prediction-time* artefact by comparing two batch sizes on the same checkpoint, and then shipped a one-line `--batch_size 1` change that immediately put her at rank 1. Most of the cohort never realised — fern explicitly tested bs=1 vs bs=2 and reported *no* difference (her masking was correct), and nezuko/edward never tested it.

### Askeladd: the surf_p_weight sweep, and stopping at the right time

Askeladd's later chain (iter25 → iter32) is the most disciplined optimisation work in the run. She pushed the surface-pressure-channel weight monotonically 4 → 12 → 16 → 24 → 32 → 48, warm-starting each step and watching for the inflection rather than assuming "higher is better":

> sp_w=32 is the new best single-model recipe. geom_camber_rc dropped to 56.44 (best ever). … iter31 (sp_w=48) … `surf_p_weight 32 → 48` … sp_w=48 alone is at the threshold where pressing harder hurts the velocity outputs (and thus surface predictions overall) but adds diversity to ensemble. Looks like the sp_w sweep is saturating. sp_w=32 was the inflection — beyond it the alone score plateaus or regresses.

Even at saturation she squeezed value out of iter31 by *ensembling it*, because it lived in a different basin from sp_w=32. Her final iter32 ensemble weights iter31 and iter29 at 2× and trims the others — the most carefully weighted blend of the run.

### Tanjiro: SWA over the chain (the cleanest single-iter rank jump)

Tanjiro's iter21 is the single biggest rank movement of the run from any single iteration: rank 4 → rank 2 in one shot, with no new training, just by averaging the last five checkpoints' state_dicts in fp32:

> **iter21: SWA over iter16-20 — RANK 2.** Stochastic-weight-averaging the latest checkpoints in the warm-start chain (iter16, iter17, iter18, iter19, iter20) should produce a smoother model with better generalization than any single checkpoint.
> **Test 37.21 (commit defe248) — RANK 2!** Per-split: single=35.28, geom_rc=49.02 (best of any agent!), geom_cruise=24.12, re_rand=40.41. Single best result of the run.
> SWA + chain warm-starts = the killer combo. The chain produces correlated-but-different checkpoints and averaging cancels their independent errors.

That `geom_rc=49.02` is the *best `geom_rc` of any agent in the entire run* — better than askeladd's 49.42. In the apr23 run, the universal lesson was "same-lineage ensembles regress"; tanjiro's iter21 is the apr27-4 counter-example, because *weight averaging in parameter space* genuinely denoises the chain in a way that *prediction averaging* does not.

### Nezuko: alternating "polish" and "jolt" iterations

Nezuko's late-game move was a programmatic alternation between low-LR polish and medium-LR "jolt" iterations specifically aimed at escaping the local basin the polish chain settles into:

> **iter17 LR-jolt branch + small noise (broke asymptote).** Chain at lr=1e-5/2e-5 was asymptoting at 42.08–42.11. Inject optimization energy: jump LR back up to 1e-4 (10× higher than iter16) and add a touch of feature_noise (0.03) … −0.33 vs iter16, the biggest gain since iter6→iter7. This refutes my earlier "asymptote" call — restoring optimization energy after slow polish *did* let the model find a better basin.

The cumulative effect from iter17 → iter22 was -1.36 val (vs -0.34 for the first jolt alone) — the gains *compounded* over multiple jolt cycles. She also explicitly diagnosed when the jolt was too aggressive (iter20 at `lr=2e-4` destabilised the warm-start) and tuned the cycle by hand.

### Thorfinn: train + 80% val for direct test alignment

Thorfinn's iter17 is the most pragmatic move in the run: he simply **promoted 80% of val into training**, accepting a noisier 20% holdout for monitoring:

> **iter17: train on train + 80% val.** Each val_X split is a random holdout from the same distribution as test_X, so promoting val data into the training set should directly improve test performance. Held out 20% per split for noisy monitoring.

This took him from test 38.07 → 37.68 in one iteration and held him at #3 for the remainder of the run. It's the most "kaggler" move of the cohort — pure data-leverage on the metric that actually scores you.

### Edward: 16-source ensemble optimisation

Edward's path was the most diverse: 27 iterations spanning Fourier-feature models, vanilla Transolvers, two seed re-runs (`PYTHONHASHSEED=42`, `=123`), deeper-and-wider variants, and chains of different `p_weight`. Then she wrote `fast_optimize.py` to greedy-search ensemble weights over all 16 checkpoints simultaneously:

> Single-model val surf_p (weighted L1): iter3 51.27 (best), iter2 51.87, iter1 53.46, iter11 53.59 … After greedy refinement: **49.16** with weights iter1=0.005, iter2=0.20, iter3=0.39, iter9=0.24, iter11=0.17. Bigger / vanilla branches (iter5,6,7,8,10,12) all dropped — too distinct from the dominant Fourier+chain family.

The optimiser correctly *zero-weighted* her architectural-diversity branches (bigger model, vanilla no-Fourier) — they were too divergent in error structure to help. Edward landed at rank 5 (test 41.04) and saturated at val ≈ 48.85 across her final 5 iterations, unable to break through.

### Frieren: stuck on the scorer

Frieren had the architectural ambition of the cohort — she shipped a working ResMLP + GlobalFiLM + ContextFiLM + GatedSliceAttention stack and an input-noise + ensemble pipeline — but every submission hung. The journal records the moment she stopped fighting the scorer:

> **Frieren's submissions are stuck "incomplete" in scores.json** — frieren/{3325cb3, 9d7edad, 682b17d, …, 2c8f181} all marked incomplete. Format and file sizes match thorfinn's accepted submissions. The scorer must run on a schedule or be triggered by something I'm missing.
>
> Frieren scoring still all "incomplete" — predictions at `/mnt/new-pvc/predictions/apr27-4/frieren/<commit>/` exist with correct sizes/shapes/dtypes vs scored agents. Format is identical to fern's accepted submissions. Mystery; nothing more I can do here besides keep submitting.

She kept submitting, kept iterating on architecture (every commit is a clean architectural extension of the previous, with zero-init residual heads to keep warm-starts identity-equivalent), and never appeared on the board. This is the apr27-4 framework finding: the apr23 scorer-race bug is not just "every agent has one or two `incomplete` entries"; under unlucky scheduling it can pin a single agent end-to-end.

### Fern: orthogonal architecture sweep

Fern was alone in pushing fundamentally different shapes (256×4 wide-shallow, 320×4, 384×4, 128×10 narrow-deep) and ensembling across them. Her best discovery was that *architecture diversity* gives real ensemble gain:

> **wider-shallow-320x4 (kept, ensemble best 89.09).** standalone val=92.01 (best single ever, 0.6 better than iter19) … 3-model ensemble iter20+iter19+iter8 at weights 0.35/0.35/0.30 → val 89.09 (best ensemble ever, 0.9 better than iter19+iter8 alone). The 3-model ensemble of three architecturally different models (320x4, 256x4, 192x8) is much better than any pair.

But her chain never reached the apr23-recipe basin: she never tried bs=2 + full-mesh + warm-start, and her final test score was 90.80 — five times worse than the apr27-bis fern best. The lesson is the inverse of edward's: architectural diversity helps ensembling *within a regime*, but if your single-model regime is already two divisions below the rest of the cohort, no amount of diversity rescues it.

## What the framework demonstrates (and where it breaks)

- **Cross-run memory is real.** Every top-five agent in apr27-4 explicitly reads from `/mnt/new-pvc/kagent/apr27/` and `/mnt/new-pvc/kagent/apr23/` checkpoints — alphonse warm-starts from a frieren apr23 ckpt by name in iter1. The PVC is durable, and the cohort uses it as a shared library of prior wins. This makes apr27-4 less of an "independent discovery" run than apr23 was — it is closer to a "build-on-prior-art" run, with the result being a better top score (33.72 < 34.41) but smaller per-iteration delta.

- **The same scorer-race bug surfaced in apr23 still bites.** It is now documented across two runs. Frieren's 14-hour exclusion from the leaderboard is a serious framework failure — the next iteration of the harness needs to re-score `incomplete` entries when their files are present, or fail loudly on the agent side rather than silently.

- **One agent can run away with a non-architectural insight.** Askeladd's bs=1 prediction discovery (a one-line change) and disciplined p-weight sweep are the dominant levers of the run. No single architectural innovation moved more than ~1 test point; the bs=1 fix moved 18 points alone. The framework rewards careful diagnosis of the *eval pipeline* more than capacity scaling.

- **SWA in parameter space beats prediction-averaging on a chain.** This is the genuinely new empirical finding of apr27-4 vs apr23. Tanjiro's iter21 (SWA over iter16-20) jumped from rank 4 to rank 2 with no new training. Multiple agents (nezuko, askeladd, edward) tried *prediction* ensembling on their own chains and found the same regression that apr23 documented; tanjiro alone tried *weight-space* averaging and it worked cleanly. The apr23 lesson "same-lineage ensembles regress" applies to prediction ensembling, not weight averaging.

- **All eight journals are intact and the run is reproducible from the remote alone.** 364 kaggler commits, 207 organiser leaderboard updates, and ~1700 lines of journal across the eight agents are reachable from `origin/apr27-4/kaggler/<name>` and `origin/apr27-4-leaderboard`. Every claim in this document is traceable to those refs.

## Acknowledgements

The winning submission (askeladd's iter32 at commit `6b42d98`) is entirely the product of the `askeladd` agent's own iteration: 37 commits across the warm-start chain, the bs=1 prediction-time discovery, the surf_p_weight sweep (4 → 12 → 16 → 24 → 32 → 48), and the final 5-way weighted ensemble. No manual edits. The narrative in this document is reproducible from `origin/apr27-4/kaggler/askeladd` and the scored predictions on the shared volume.
