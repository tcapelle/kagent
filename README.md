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
│   └── *.yaml, *.sh          K8s templates + entrypoints
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

Required K8s secret: `kagent-secrets` with `anthropic-api-key`, `wandb-api-key`, `github-token`.
