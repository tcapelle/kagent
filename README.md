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

### CFD Surrogate (`cfd-competition/`)

Train neural network surrogates for computational fluid dynamics on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. Predict velocity and pressure fields over 2D airfoil meshes.

- **Task**: given 24-dim mesh features, predict 3 output channels (Ux, Uy, p)
- **Data**: 2,699 samples, 100k+ nodes per mesh, 4 val/test splits testing geometry and Reynolds number generalization
- **Metric**: surface pressure MAE (avg across 4 test splits)
- **Baseline**: Transolver (physics-aware attention)

See [`cfd-competition/`](cfd-competition/) for full details.

### GRaM 3D Airflow (`gram-competition/`)

Predict future 3D velocity fields around Formula 1-style front wings. Based on the [GRaM ICLR 2026 Workshop Competition](https://github.com/gram-competition/iclr-2026).

- **Task**: given 5 input timesteps of velocity field u(t, x), predict the next 5 timesteps
- **Data**: 810 samples from 162 F1 wing simulations, 100k 3D points per sample (~14 GB)
- **Metric**: mean L2 velocity error averaged over space and time
- **Baseline**: ResMLP (per-point, no spatial interaction)

See [`gram-competition/`](gram-competition/) for full details.

## Quick start

```bash
# 1. Prepare data (one-time per competition)
uv run k8s/launch.py --tag mar28 --competition cfd-competition --prepare

# 2. Launch kagglers + organizer
uv run k8s/launch.py --tag mar28 --competition cfd-competition --n_kagglers 20 --organizer

# 3. Monitor
kubectl get deployments -l research-tag=mar28,competition=cfd-competition
kubectl logs -f deployment/kagent-mar28-frieren

# 4. Stop
uv run k8s/kill.py --tag mar28
```

Run multiple competitions or multiple runs of the same competition in parallel:

```bash
# CFD competition with 20 agents
uv run k8s/launch.py --tag mar28 --competition cfd-competition --n_kagglers 20 --organizer

# GRaM competition with 4 agents (same cluster, same time)
uv run k8s/launch.py --tag mar28 --competition gram-competition --n_kagglers 4 --organizer

# Full cleanup (deployments + branches + PVC predictions)
uv run k8s/kill.py --tag mar28 --competition gram-competition --clean_branches --clean_predictions
```

## Repo structure

```
kagent/
├── cfd-competition/          CFD surrogate (2D airfoil meshes)
│   ├── kaggler/              What agents get
│   └── organizer/            Data prep, scoring, baseline
├── gram-competition/         GRaM 3D airflow (F1 front wings)
│   ├── kaggler/              What agents get
│   └── organizer/            Data prep, scoring, baseline
├── k8s/
│   ├── launch.py             Deploy kagglers + organizer
│   ├── kill.py               Tear down deployments
│   ├── pi-settings.json      pi packages (pinned) + subagent config
│   ├── pi-weave/             pi extension: traces agent runs to Weave
│   └── *.yaml, *.sh          K8s templates + entrypoints
├── .agents/skills/           pi skills (W&B, competition report)
├── AGENTS.md                 Dev guidelines (loaded by pi as context)
├── config.yaml               Default launch configuration
└── pyproject.toml
```

Each competition follows the same structure:
```
<name>-competition/
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

## Design

**Competition-agnostic infrastructure** (`k8s/`): pod orchestration, agent loops, scoring. Generic pattern — data on PVC, agents on branches, W&B for metrics. Everything scoped by `--tag` so multiple runs coexist.

**To add a new competition**: create `<name>-competition/` with `kaggler/{KAGGLER_AGENT.md, README.md, data.py, train.py, predict.py}` and `organizer/{prepare_splits.py, score.py}`, then launch with `--competition <name>-competition`.

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
- **Context and skills**: pi loads `AGENTS.md` from the repo root and discovers skills in `.agents/skills/` from the kaggler working directory. Skills under `.agents/` count as project resources, which is why the loop passes `--approve`. pi has no permission prompts, so there is no equivalent of a skip-permissions flag.
- **Packages**: `k8s/pi-settings.json` is copied to `~/.pi/agent/settings.json` at pod start, and `pi list` installs what it declares. All versions are pinned, so bump them there:
  - `pi_exa` (private, cloned with `GITHUB_TOKEN`): the `exa_search` and `web_fetch` tools for the main agent. Needs `EXA_API_KEY`.
  - `pi-subagents`: a `subagent` tool with builtin `researcher`, `reviewer`, `scout`, `oracle`, `worker` and `delegate` agents. The kaggler prompt explicitly allows delegation, because pi-subagents only delegates when instructions authorize it. The builtin agents that shell out to the Claude Code, Codex and Cursor CLIs are disabled. In `pi -p` mode children run in the background, and the parent waits for them (`bg_wait`) before it exits.
  - `pi-web-access`: installed with every resource filtered out of the main agent and loaded only in subagent children (`defaultSubagentOnlyExtensions`). The builtin `researcher` needs its `web_search`, `fetch_content` and `source_check` tools, plus PDF extraction and GitHub cloning. It also uses `EXA_API_KEY`.
  - `k8s/pi-weave`: Weave tracing (below).
- **Weave tracing**: `k8s/pi-weave` is a small pi extension built on the Weave GenAI SDK. Each run becomes an `invoke_agent <kaggler>` trace, with a `chat <model>` span per turn (output and token usage) and `execute_tool <name>` spans nested under the turn that issued them. The pi session id is the Weave conversation id. Background subagent children load the same extensions, so each child shows up as its own conversation under the kaggler's name. Traces go to `WANDB_ENTITY/WANDB_PROJECT` and appear in the project's Agents tab. Personal entities may reject Agents traces with a 403; team entities work.
- **Logs**: stdout of every iteration is pi's JSON event stream on the PVC at `/mnt/new-pvc/kagent/<tag>/<kaggler>/`. pi's own diagnostics go to stderr, i.e. the pod log. Hivemind streams the session files in `~/.pi/agent/sessions`.
- **Subagent artifacts**: child research briefs and outputs go under `~/.pi/agent/sessions/<cwd>/subagent-artifacts/`, and run state under `$TMPDIR/pi-subagents-uid-*/`.
