# kagent

Autonomous ML competition framework powered by coding agents on Kubernetes.

Each agent (kaggler) gets a GPU pod, a branch, and a problem to solve. They iterate autonomously — writing models, training, checking the leaderboard, stealing ideas from rivals, and pushing improvements. An organizer scores submissions and maintains a live leaderboard.

## How it works

```
┌─────────────────────────────────────────────────────────┐
│  Cluster (k8s)                                          │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │ frieren  │ │  fern    │ │ tanjiro  │  ... x20       │
│  │ 1 GPU    │ │ 1 GPU    │ │ 1 GPU    │                │
│  │ Agent    │ │ Agent    │ │ Agent    │                │
│  │ runtime  │ │ runtime  │ │ runtime  │                │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                │
│       │             │            │                      │
│       ▼             ▼            ▼                      │
│  ┌─────────────────────────────────────┐                │
│  │  PVC: /mnt/new-pvc                  │                │
│  │  ├── datasets/   (pre-split data)   │                │
│  │  └── predictions/ (submissions)     │                │
│  └─────────────────────────────────────┘                │
│       ▲                                                 │
│       │                                                 │
│  ┌────┴─────┐                                           │
│  │organizer │ → scores predictions → leaderboard.md     │
│  │ (no GPU) │ → logs to W&B                             │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

Each kaggler runs an agent loop:
1. Read instructions + check leaderboard
2. Write/improve model (`train.py`)
3. Train (30 min cap, logs to W&B)
4. Commit & push to `<tag>/kaggler/<name>` branch
5. Generate predictions on hidden test set
6. Repeat — check rivals' W&B runs, steal ideas, iterate

## Competitions

Each competition lives in its own git repo, so kaggler branches, leaderboard commits and PR noise stay out of kagent. kagent is only the infrastructure. Point it at a competition with `competition_repo` in `config.yaml` (or `--competition_repo`).

| Repo | Task | Metric |
|---|---|---|
| [`tcapelle/tandemfoil-competition`](https://github.com/tcapelle/tandemfoil-competition) | CFD surrogate on [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn): predict Ux, Uy, p over 2D airfoil meshes (baseline: Transolver) | surface pressure MAE |
| [`tcapelle/gram-competition`](https://github.com/tcapelle/gram-competition) | [GRaM ICLR 2026](https://github.com/gram-competition/iclr-2026): predict the next 5 timesteps of 3D velocity around F1 front wings (baseline: ResMLP) | mean L2 velocity error |

## Quick start

```bash
# 1. Prepare data (one-time per competition)
uv run k8s/launch.py --tag mar28 --competition_repo https://github.com/tcapelle/tandemfoil-competition.git --prepare

# 2. Launch kagglers + organizer
uv run k8s/launch.py --tag mar28 --competition_repo https://github.com/tcapelle/tandemfoil-competition.git --n_kagglers 20 --organizer

# 3. Monitor
kubectl get deployments -l research-tag=mar28,competition=tandemfoil-competition
kubectl logs -f deployment/kagent-mar28-frieren

# 4. Stop
uv run k8s/kill.py --tag mar28 --competition_repo https://github.com/tcapelle/tandemfoil-competition.git
```

The k8s `competition` label is the repo name, so several competitions, or several runs of one, can share a cluster:

```bash
uv run k8s/launch.py --tag mar28 --competition_repo https://github.com/tcapelle/gram-competition.git --n_kagglers 4 --organizer

# Full cleanup (deployments + branches in the competition repo + PVC predictions)
uv run k8s/kill.py --tag mar28 --competition_repo https://github.com/tcapelle/gram-competition.git --clean_branches --clean_predictions
```

## Repo structure

```
kagent/
├── k8s/
│   ├── launch.py             Deploy kagglers + organizer
│   ├── kill.py               Tear down deployments (and competition-repo branches)
│   ├── pi-settings.json      pi packages (pinned) + skills + subagent config
│   ├── pi-weave/             pi extension: traces agent runs to Weave
│   └── *.yaml, *.sh          K8s templates + entrypoints
├── .agents/skills/           wandb-primary (given to kagglers), competition-report (dev-side)
├── AGENTS.md                 Dev guidelines
├── config.yaml               Default launch configuration
└── pyproject.toml            Shared python deps for every competition
```

Each competition repo has the same layout:
```
<name>-competition/
├── .gitignore                Training artifacts, except kaggler/checkpoints/best.pt
├── kaggler/
│   ├── KAGGLER_AGENT.md      Agent loop instructions
│   ├── README.md             Competition description + rules
│   ├── data.py               Data loader (read-only)
│   ├── train.py              Training script (agents modify this)
│   └── predict.py            Prediction script (agents modify this)
└── organizer/
    ├── prepare_splits.py     One-time data prep
    ├── score.py              Scoring + leaderboard
    └── train.py              Baseline model (not given to agents)
```

Optional: an `AGENTS.md` at the repo root or `.agents/skills/` gives kagglers competition-specific context and skills.

## Design

**Competition-agnostic infrastructure** (`k8s/`): pod orchestration, agent loops, scoring. Generic pattern: data on the PVC, agents on branches of the competition repo, metrics in W&B. Everything is scoped by `--tag`, so multiple runs coexist.

**Two clones per pod.** Every pod clones kagent to `/workspace/kagent` (k8s scripts, pi config, `pyproject.toml`) and the competition repo to `/workspace/competition`.
- **Kagglers** sparse-checkout `kaggler/` (and `.agents/`) and work on `<tag>/kaggler/<name>` branches pushed to the competition repo. Organizer code is hidden by sparse checkout only; it's still reachable through git history.
- **The organizer** clones the competition repo in full, runs `organizer/score.py`, and pushes the `<tag>-leaderboard` branch to the competition repo.
- **The prepare job** runs `organizer/prepare_splits.py`.
- Python deps always come from kagent's `pyproject.toml`.

**To add a new competition**: create a repo with the layout above, then launch with `--competition_repo <url>`. `GITHUB_TOKEN` (`github-token` in `kagent-secrets`) must be able to push to it.

Required K8s secret: `kagent-secrets` with `wandb-api-key`, `github-token`, plus the credential for your model provider (see below). `exa-api-key` is optional and enables web search.

## Agent harness (pi)

Kagglers run [pi](https://pi.dev) (`@earendil-works/pi-coding-agent`, baked into the image). Each loop iteration in `k8s/entrypoint-kaggler.sh` is one fresh, non-interactive run:

```bash
pi -p --mode json --approve --model "$AGENT_MODEL" "$PROMPT" > iter_<n>_<date>.jsonl
```

- **Model**: `agent_model` in `config.yaml` (or `--agent_model`) is a pi model string, `provider/id[:thinking]`, e.g. `anthropic/claude-opus-5-5:high` or `openai/gpt-5.4:high`. The entrypoint runs `pi auth check --model "$AGENT_MODEL"` and fails fast if the provider has no credential.
- **Credentials**: the deployment passes these optional `kagent-secrets` keys through. Set the one your provider needs:

  | Secret key | Env var |
  |---|---|
  | `anthropic-api-key` | `ANTHROPIC_API_KEY` |
  | `anthropic-oauth-token` | `ANTHROPIC_OAUTH_TOKEN` (a Claude Pro/Max token) |
  | `openai-api-key` | `OPENAI_API_KEY` |
  | `gemini-api-key` | `GEMINI_API_KEY` |

- **Fresh session per iteration**: pi compacts context automatically within a run. When a run exits, the next iteration starts cold and re-reads `KAGGLER_AGENT.md` and the experiment journal.
- **Context and skills**: kagglers get `wandb-primary` through `skills` in `k8s/pi-settings.json`. kagent's own `AGENTS.md` (dev guidelines) is not loaded, because the kaggler works in `/workspace/competition/kaggler`. A competition repo can add a root `AGENTS.md` (loaded as context) and `.agents/skills/` (project resources, which is why the loop passes `--approve`). pi has no permission prompts, so there is no equivalent of a skip-permissions flag.
- **Packages**: `k8s/pi-settings.json` is copied to `~/.pi/agent/settings.json` at pod start, and `pi list` installs what it declares. All versions are pinned, so bump them there:
  - `pi_exa` (private, cloned with `GITHUB_TOKEN`): the `exa_search` and `web_fetch` tools for the main agent. Needs `EXA_API_KEY`.
  - `pi-subagents`: a `subagent` tool with builtin `researcher`, `reviewer`, `scout`, `oracle`, `worker` and `delegate` agents. The kaggler prompt explicitly allows delegation, because pi-subagents only delegates when instructions authorize it. The builtin agents that shell out to the Claude Code, Codex and Cursor CLIs are disabled. In `pi -p` mode children run in the background, and the parent waits for them (`bg_wait`) before it exits.
  - `pi-web-access`: installed with every resource filtered out of the main agent and loaded only in subagent children (`defaultSubagentOnlyExtensions`). The builtin `researcher` needs its `web_search`, `fetch_content` and `source_check` tools, plus PDF extraction and GitHub cloning. It also uses `EXA_API_KEY`.
  - `k8s/pi-weave`: Weave tracing (below).
- **Weave tracing**: `k8s/pi-weave` is a small pi extension built on the Weave GenAI SDK. Each run becomes an `invoke_agent <kaggler>` trace, with a `chat <model>` span per turn (output and token usage) and `execute_tool <name>` spans nested under the turn that issued them. The pi session id is the Weave conversation id. Background subagent children load the same extensions, so each child shows up as its own conversation under the kaggler's name. Traces go to `WANDB_ENTITY/WANDB_PROJECT` and appear in the project's Agents tab. Personal entities may reject Agents traces with a 403; team entities work.
- **Logs**: stdout of every iteration is pi's JSON event stream on the PVC at `/mnt/new-pvc/kagent/<tag>/<kaggler>/`. pi's own diagnostics go to stderr, i.e. the pod log. Hivemind streams the session files in `~/.pi/agent/sessions`.
- **Subagent artifacts**: child research briefs and outputs go under `~/.pi/agent/sessions/<cwd>/subagent-artifacts/`, and run state under `$TMPDIR/pi-subagents-uid-*/`.
