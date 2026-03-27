# ORCHESTRATOR.md — Living Context for kagent

**Last updated:** 2026-03-26

## Architecture Overview

**kagent** is an autonomous ML competition framework. Coding agents (Claude/Codex) compete on Kaggle-style problems in Kubernetes pods. Each agent gets a GPU, a git branch, and iterates autonomously: write models, train, predict, check leaderboard, improve.

### Directory Layout

```
kagent/
├── cfd-competition/              # Current competition (CFD Surrogate)
│   ├── kaggler/                  # What agents get (read-only contract for data.py)
│   │   ├── KAGGLER_AGENT.md      # Agent loop instructions
│   │   ├── README.md             # Competition rules & data format
│   │   ├── data.py               # Data loader — READ-ONLY, agents must not modify
│   │   ├── train.py              # Training template — agents fill in model
│   │   ├── predict.py            # Prediction template — agents fill in model loading
│   │   └── viz.py                # Visualization helpers
│   └── organizer/                # Organizer-only infrastructure
│       ├── prepare_splits.py     # One-time data preprocessing
│       ├── score.py              # Score predictions, update leaderboard
│       └── train.py              # Baseline Transolver model (reference)
├── k8s/                          # Kubernetes orchestration
│   ├── launch.py                 # Deploy kagglers + organizer
│   ├── kaggler-deployment.yaml   # Kaggler pod template (1 GPU, 96GB RAM)
│   ├── organizer-deployment.yaml # Organizer pod template (no GPU, 128GB RAM)
│   ├── prepare-splits-job.yaml   # Data prep K8s job
│   ├── entrypoint-kaggler.sh     # Kaggler pod startup
│   └── entrypoint-organizer.sh   # Organizer pod startup
├── leaderboard.md                # Auto-updated by organizer
├── pyproject.toml                # uv project, Python >=3.10
├── CLAUDE.md                     # Dev guidelines
└── ORCHESTRATOR.md               # This file
```

### Module Boundaries

| Module | Owner | Purpose | Touchable? |
|--------|-------|---------|------------|
| `kaggler/data.py` | Organizer | Data loader, SplitDataset, pad_collate | READ-ONLY |
| `kaggler/train.py` | Agents | Training loop template (agents fill in model) | Agents modify |
| `kaggler/predict.py` | Agents | Prediction template (agents fill in model loading) | Agents modify |
| `kaggler/viz.py` | Shared | Visualization helpers | Rarely touched |
| `organizer/prepare_splits.py` | Organizer | One-time data split & normalization | Rarely touched |
| `organizer/score.py` | Organizer | Scoring + leaderboard management | Infrastructure |
| `organizer/train.py` | Organizer | Baseline Transolver model | Reference only |
| `k8s/launch.py` | Infra | Pod deployment orchestration | Infrastructure |

### Key Contracts

**Model I/O:** Input `{"x": tensor[B, N, 24]}` → Output `{"preds": tensor[B, N, 3]}`

**Data paths (on K8s PVC):**
- Data: `/mnt/new-pvc/datasets/tandemfoil/splits/{train,val_*,test}/*.pt`
- Predictions: `/mnt/new-pvc/predictions/{agent}/{commit}/predictions.pt`
- Ground truth (hidden): `/mnt/new-pvc/datasets/tandemfoil/splits/.test_gt/`

**Validation splits:** val_in_dist, val_tandem_transfer, val_ood_cond, val_ood_re

**Primary metric:** overall surface pressure MAE (`mae_surf_p`, lower is better)

**Training constraints:** 30-min timeout, single GPU (96GB VRAM), surf_weight=10.0

### Conventions

- **Arg parsing:** `simple_parsing.parse(Config)` with `@dataclass Config`
- **Logging:** rich.Console for scripts, W&B for experiments
- **Types:** Python 3.12+ (`str | None`, `dict`, `list`)
- **Dependencies:** uv, run with `uv run script.py`
- **Error handling:** Fail-fast, minimal try/except
- **W&B:** entity=wandb-applied-ai-team, project=kagent-v1
- **Agent runtimes:** Claude (claude-opus-4-6[1m]) or Codex (gpt-5.4)
- **Agent names:** Anime characters (frieren, edward, tanjiro, etc.)

### Known Fragilities

1. **Hardcoded PVC paths** — `/mnt/new-pvc/...` assumed everywhere; not portable
2. **No model output validation** — agents can return wrong shapes silently
3. **Single organizer** — leaderboard git push has no locking
4. **weights_only=False** in score.py/prepare_splits.py — trusts data on PVC
5. **Git subprocess calls** in predict.py/score.py — assume repo & auth available

### Dependencies (pyproject.toml)

Core: torch, torch_geometric, wandb, simple-parsing, einops, timm, rich, matplotlib, tqdm, pyyaml, python-dotenv

### Current State

- 27 agents on leaderboard, top: edward (mae_surf_p=28.73), haku (29.45), tanjiro (31.62)
- Competition: CFD Surrogate (TandemFoilSet — predict velocity/pressure over airfoil meshes)
- Baseline model: Transolver (physics-aware attention)

---

## PhysicsNeMo Integration (branch: `nemo`)

### What is PhysicsNeMo

NVIDIA's open-source deep learning framework for physics AI models. Provides 24+ model architectures including MeshGraphNet, Transolver, DoMINO, FNO, and more. Models are standalone `nn.Module` subclasses usable independently of the framework's training utilities. Repo: https://github.com/NVIDIA/physicsnemo

### Version Constraint

- **physicsnemo 2.0.0** (latest, Mar 2026) requires **Python 3.11+**
- K8s container image (`ghcr.io/tcapelle/dev_box`) runs **Python 3.10** → installs **physicsnemo 1.3.0**
- The 1.3.0 Transolver API is compatible with our wrapper; no code changes needed
- To use 2.0 features (Transolver++, time conditioning, `plus=True`), the container image needs a Python upgrade

### Models Relevant to CFD Meshes

| Model | Import | Input Format | Best For |
|-------|--------|-------------|----------|
| **Transolver** | `physicsnemo.models.transolver.Transolver` | `fx[B,N,C_in]` + `embedding[B,N,C_emb]` → `[B,N,C_out]` | Unstructured meshes, physics-aware attention, our current choice |
| **MeshGraphNet** | `physicsnemo.models.meshgraphnet.MeshGraphNet` | `node_features, edge_features, graph` → `[N, C_out]` | Graph-based CFD, needs edge_index (not in current data pipeline) |
| **DoMINO** | `physicsnemo.models.domino.DoMINO` | Complex dict with geometry + mesh coords | Industrial CFD with explicit geometry, heavy input requirements |
| **FIGConvUNet** | `physicsnemo.models.figconvnet.FIGConvUNet` | Mesh-based conv | Convolutional approach to mesh CFD |

### NeMoTransolver Wrapper (our implementation)

```python
# cfd-competition/kaggler/train.py
class NeMoTransolver(nn.Module):
    # Wraps physicsnemo Transolver to match kaggler model contract:
    #   Input:  {"x": tensor[B, N, 24]}
    #   Output: {"preds": tensor[B, N, 3]}
    # Splits x into:
    #   embedding = x[:, :, :2]   (spatial coords)
    #   fx        = x[:, :, 2:]   (22 functional features)
```

**Constructor args passed to PhysicsNeMo Transolver:**
- `functional_dim=22`, `embedding_dim=2`, `out_dim=3`
- `unified_pos=False` (uses explicit spatial embedding, not distance-based)
- `structured_shape=None` (unstructured mesh mode)
- `use_te=False` (no Transformer Engine dependency)

**Current model config** (matches organizer baseline scale):
```python
model_config = dict(n_hidden=128, n_layers=5, n_head=4, slice_num=64, mlp_ratio=2)
# → 645,719 params, 18.7GB VRAM
```

### PhysicsNeMo Transolver API Details

```python
Transolver(
    functional_dim: int,           # non-spatial feature dims (22 for us)
    out_dim: int,                  # output dims (3: Ux, Uy, p)
    embedding_dim: int | None,     # spatial coord dims (2 for us), required if unified_pos=False
    n_layers: int = 4,
    n_hidden: int = 256,           # must be divisible by n_head
    dropout: float = 0.0,
    n_head: int = 8,
    act: str = "gelu",
    mlp_ratio: int = 4,
    slice_num: int = 32,           # physics-informed slices for attention
    unified_pos: bool = False,     # True = distance-based pos encoding
    structured_shape: tuple | None = None,  # None = irregular mesh
    use_te: bool = True,           # Transformer Engine (set False to avoid dep)
    time_input: bool = False,      # temporal conditioning
    plus: bool = False,            # Transolver++ variant (2.0 only)
)

# Forward: concatenates embedding+fx, projects to hidden, runs transformer blocks
def forward(fx, embedding=None, time=None) -> tensor
```

**Key internals:** `embedding` and `fx` are concatenated before the preprocess MLP. Physics attention decomposes the domain into `slice_num` learned slices, computes attention within slices, then projects back. This is the same mechanism as the organizer's hand-rolled Transolver, but with NVIDIA's optimized implementation.

### K8s Testing Results (2026-03-26)

- **Tested via**: `k8s/nemo-test-job.yaml` (batch Job, 1 GPU, 32GB RAM)
- **Install**: `uv pip install --system -e .` → physicsnemo 1.3.0 installs cleanly alongside existing torch
- **Debug run**: 3 epochs, 6 train samples, ~4 seconds total
- **VRAM**: 18.7GB peak (well within 96GB pod limit)
- **Loss trajectory**: decreasing (val/loss 18.9 → 16.7 → 16.1)
- **All 4 val splits**: producing correct metrics
- **W&B issue**: cluster `kagent-secrets:wandb-api-key` returns 401; used `WANDB_MODE=offline` to work around

### Future Directions with PhysicsNeMo

- **MeshGraphNet**: Would require adding edge construction (k-NN or Delaunay from positions) to the data pipeline — `data.py` is read-only so this would need a preprocessing step in train.py
- **Transolver++** (`plus=True`): Available in physicsnemo 2.0+, needs Python 3.11+ container
- **Larger configs**: n_hidden=256, n_layers=8 for more capacity (check VRAM)
- **Hyperparameter sweep**: slice_num (32/64/128), mlp_ratio (2/4), dropout, learning rate

---

## Decisions Log

- **2026-03-26**: Created `nemo` branch with PhysicsNeMo Transolver integration. NeMoTransolver wrapper splits 24-dim input into position (2) + features (22). Uses physicsnemo 1.3.0 (container is Python 3.10; physicsnemo 2.0 needs 3.11+). Tested on K8s GPU pod — works end-to-end, 18.7GB VRAM, loss decreases. W&B cluster secret may be stale (401 error).

## Subagent Guidelines

When spawning subagents, always include:
1. **Goal** — what to achieve
2. **Files owned** — what they can modify
3. **Files read-only** — what they must not touch
4. **Conventions** — arg parsing, typing, error handling, logging
5. **Verification** — how to confirm the work is correct
6. **Context** — relevant architecture from this file
