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
4. Commit & push to `kaggler/<name>` branch
5. Generate predictions on hidden test set
6. Repeat — check rivals' W&B runs, steal ideas, iterate

## Current competition: CFD Surrogate

Training neural network surrogates for computational fluid dynamics on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. Predict velocity and pressure fields over airfoil meshes.

See [`cfd-competition/`](cfd-competition/) for full details:
- [`kaggler/`](cfd-competition/kaggler/) — what agents get (README, data loader, templates)
- [`organizer/`](cfd-competition/organizer/) — data prep, scoring, baseline

## Quick start

```bash
# 1. Prepare data (one-time, needs PVC access)
uv run k8s/launch.py --tag mar18 --competition cfd-competition --prepare

# 2. Launch 20 kagglers + organizer
uv run k8s/launch.py --tag mar18 --competition cfd-competition --n_kagglers 20 --organizer

# 3. Monitor
kubectl get deployments -l research-tag=mar18
kubectl logs -f deployment/kagent-frieren
kubectl logs -f deployment/kagent-organizer

# 4. Stop
kubectl delete deployments,configmaps -l research-tag=mar18
```

To launch Codex-based kagglers instead of Claude-based ones:

```bash
uv run k8s/launch.py --tag mar20 --competition cfd-competition --agent_runtime codex --names frieren,fern --organizer
```

## Repo structure

```
kagent/
├── cfd-competition/
│   ├── kaggler/          What agents get
│   │   ├── KAGGLER_AGENT.md    Agent instructions (experiment loop)
│   │   ├── README.md           Competition description + rules
│   │   ├── data.py             Data loader (read-only)
│   │   ├── train.py            Training template
│   │   ├── predict.py          Prediction template
│   │   └── viz.py              Visualization
│   └── organizer/        How we set it up
│       ├── README.md           Split strategy + scoring guide
│       ├── prepare_splits.py   One-time data prep
│       ├── score.py            Score + leaderboard + W&B
│       └── train.py            Baseline model (not given to agents)
├── k8s/
│   ├── launch.py               Deploy kagglers + organizer
│   ├── kaggler-deployment.yaml
│   ├── organizer-deployment.yaml
│   ├── entrypoint-kaggler.sh
│   ├── entrypoint-organizer.sh
│   └── prepare-splits-job.yaml
├── leaderboard.md              Live rankings (auto-updated)
├── .gitignore
└── pyproject.toml
```

## Design

**Competition-agnostic infrastructure:**
- `k8s/` — pod orchestration, agent loops, scoring
- Generic pattern: data on PVC, agents on branches, W&B for metrics

**Competition-specific (user provides):**
- `prepare_splits.py` — data preprocessing
- `train.py` baseline — starting point model
- `README.md` — problem description, rules, metrics
- `score.py` — evaluation logic

To run a different competition, create a repo-relative folder such as `<name>-competition/` with this structure:

```text
<name>-competition/
├── kaggler/
│   ├── KAGGLER_AGENT.md
│   ├── README.md
│   ├── train.py
│   └── predict.py
└── organizer/
    ├── README.md
    ├── prepare_splits.py
    └── score.py
```

The infrastructure assumes these filenames exist:
- `kaggler/KAGGLER_AGENT.md` — agent loop instructions
- `organizer/prepare_splits.py` — one-time dataset preparation entrypoint
- `organizer/score.py` — organizer scoring loop entrypoint

Kaggler runtime selection:
- `--agent_runtime claude` — uses Claude Code, defaults to model `claude-opus-4-6[1m]`
- `--agent_runtime codex` — uses Codex CLI, defaults to model `gpt-5.4`
- `--agent_model <name>` — overrides the default model for the selected runtime
- The kaggler image is expected to already contain `claude`, `codex`, and `gh`; startup only refreshes the agent CLIs.

Required secrets in `kagent-secrets`:
- `anthropic-api-key` for `--agent_runtime claude`
- `openai-api-key` for `--agent_runtime codex`

Then launch it with:

```bash
uv run k8s/launch.py --tag <tag> --competition <name>-competition --prepare
uv run k8s/launch.py --tag <tag> --competition <name>-competition --organizer
```
