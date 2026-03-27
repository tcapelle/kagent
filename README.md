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
kubectl get deployments -l research-tag=mar18,competition=cfd-competition
kubectl logs -f deployment/kagent-mar18-frieren
kubectl logs -f deployment/kagent-mar18-organizer

# 4. Stop
kubectl delete deployments,configmaps -l research-tag=mar18,competition=cfd-competition
```

Multiple runs of the same competition can run in parallel with different tags:

```bash
# Run A: 20 kagglers
uv run k8s/launch.py --tag run-a --competition cfd-competition --n_kagglers 20 --organizer

# Run B: 4 kagglers, different config
uv run k8s/launch.py --tag run-b --competition cfd-competition --n_kagglers 4 --organizer

# Stop just run B
kubectl delete deployments,configmaps -l research-tag=run-b,competition=cfd-competition
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

Then launch with:

```bash
uv run k8s/launch.py --tag <tag> --competition <name>-competition --prepare
uv run k8s/launch.py --tag <tag> --competition <name>-competition --n_kagglers 20 --organizer
```

Required K8s secret: `kagent-secrets` with `anthropic-api-key`, `wandb-api-key`, `github-token`.
