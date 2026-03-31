# GRaM Competition Results — gram-mar29 Run

## Setup

- **Competition**: 3D airflow velocity prediction around F1 front wings ([GRaM ICLR 2026](https://github.com/gram-competition/iclr-2026))
- **Task**: Given 5 timesteps of velocity field, predict the next 5 timesteps (100k 3D points per sample)
- **Agents**: 16 Claude Opus 4.6 agents running autonomously for ~12 hours
- **Training timeout**: 30 minutes per run
- **GPU**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
- **Data**: 730 train / 80 val samples, split by simulation ID (162 unique F1 wing geometries)
- **Metric**: Mean L2 velocity error (lower is better)
- **W&B project**: [wandb-applied-ai-team/kagent-gram](https://wandb.ai/wandb-applied-ai-team/kagent-gram)

## Where to find everything

| Resource | Location |
|----------|----------|
| Agent code (per agent) | `origin/gram-mar29/kaggler/<name>` git branches |
| Predictions | `/mnt/new-pvc/predictions/gram-mar29/<agent>/<commit>/val.pt` |
| Leaderboard | `/mnt/new-pvc/predictions/gram-mar29/leaderboard.md` |
| Leaderboard (git) | `origin/gram-mar29-leaderboard` branch |
| Scoring cache | `/mnt/new-pvc/predictions/gram-mar29/scores.json` |
| Competition code | `iclr-comp` branch |
| W&B runs | `wandb-applied-ai-team/kagent-gram` (tag: `gram-mar29`) |

### Agent branches

```
origin/gram-mar29/kaggler/violet      # 1st place (l2=0.884)
origin/gram-mar29/kaggler/gilbert     # 2nd place (l2=1.030)
origin/gram-mar29/kaggler/thorfinn    # 3rd place (l2=1.112)
origin/gram-mar29/kaggler/norman      # 4th place (l2=1.122)
origin/gram-mar29/kaggler/frieren     # 5th place (l2=1.153)
origin/gram-mar29/kaggler/haku        # 6th  (l2=1.176)
origin/gram-mar29/kaggler/askeladd    # 7th  (l2=1.195)
origin/gram-mar29/kaggler/senku       # 8th  (l2=1.197)
origin/gram-mar29/kaggler/chihiro     # 9th  (l2=1.252)
origin/gram-mar29/kaggler/emma        # 10th (l2=1.299)
origin/gram-mar29/kaggler/edward      # 11th (l2=1.313)
origin/gram-mar29/kaggler/tanjiro     # 12th (l2=1.328)
origin/gram-mar29/kaggler/nezuko      # 13th (l2=1.386)
origin/gram-mar29/kaggler/fern        # 14th (l2=1.505)
origin/gram-mar29/kaggler/kohaku      # 15th (l2=1.613)
origin/gram-mar29/kaggler/alphonse    # 16th (l2=1.637)
```

## Final Leaderboard

| Rank | Agent | Commit | l2_error | mae_Ux | mae_Uy | mae_Uz |
|------|-------|--------|---------|--------|--------|--------|
| 1 | violet | `2ea05b3` | 0.8846 | 0.5749 | 0.2826 | 0.4231 |
| 2 | gilbert | `d4749c4` | 1.0302 | 0.6824 | 0.3093 | 0.4926 |
| 3 | thorfinn | `1eb1149` | 1.1119 | 0.7368 | 0.3307 | 0.5344 |
| 4 | norman | `2f2673c` | 1.1223 | 0.7404 | 0.3441 | 0.5348 |
| 5 | frieren | `731a5c0` | 1.1535 | 0.7723 | 0.3343 | 0.5513 |
| 6 | haku | `ac1af4d` | 1.1758 | 0.7825 | 0.3496 | 0.5614 |
| 7 | askeladd | `5464a53` | 1.1952 | 0.8013 | 0.3463 | 0.5695 |
| 8 | senku | `414b028` | 1.1966 | 0.8008 | 0.3474 | 0.5709 |
| 9 | chihiro | `ce20885` | 1.2519 | 0.8355 | 0.3683 | 0.5974 |
| 10 | emma | `edc46aa` | 1.2991 | 0.8702 | 0.3723 | 0.6197 |
| 11 | edward | `b0e6abb` | 1.3126 | 0.8785 | 0.3796 | 0.6287 |
| 12 | tanjiro | `ac5d670` | 1.3284 | 0.8903 | 0.3757 | 0.6380 |
| 13 | nezuko | `3b8022d` | 1.3649 | 0.9219 | 0.3873 | 0.6477 |
| 14 | fern | `001dbf2` | 1.5054 | 0.9985 | 0.4424 | 0.7286 |
| 15 | kohaku | `40591ac` | 1.6034 | 1.0841 | 0.4535 | 0.7582 |
| 16 | alphonse | `a72c7d4` | 1.6244 | 1.1067 | 0.4555 | 0.7628 |

## Top 5 Solution Analysis

### 1st — violet (l2=0.884): Unmodified baseline

The winner submitted the **unmodified BaselineMLP** (256 hidden, 6 ResBlocks, ~1.6M params) with zero code changes. Trained on full 100k points, MSE loss, 50 epochs, lr=5e-4.

### 2nd — gilbert (l2=1.030): Unmodified baseline

Identical code to violet. Score difference comes from random seed / data ordering.

### 3rd — thorfinn (l2=1.112): EdgeConv + Fourier + temporal conv

The most architecturally complex solution (~11.8M params, 13 iterations):
- **EdgeConv k-NN** (k=16) for spatial message passing between neighboring points
- **Learnable multi-scale Fourier features** (3 frequency bands: low/med/high)
- **1D temporal convolution** over the 5 input timesteps
- **Two-phase loss**: MSE for first 40%, L2-norm for remaining 60%
- **No-slip enforcement**: hard zero at airfoil surface
- Subsampled to **15k/100k** points during training

### 4th — norman (l2=1.122): Unmodified baseline

Same baseline as violet/gilbert. Third run of the same approach.

### 5th — frieren (l2=1.153): Wide ResMLP + physics features

Largest model (~21M params, 7 iterations):
- **512 hidden, 10 ResBlocks, 4x FFN expansion**
- **Residual prediction** from last input timestep (zero-initialized output layer)
- **Physics features**: velocity trend, local turbulence intensity (std across time)
- **Velocity normalization** with dataset statistics
- **No-slip enforcement**
- Subsampled to **50k/100k** points

## Conclusions

### 1. Simplicity won

The unmodified baseline MLP placed 1st, 2nd, and 4th. Three identical runs scored 0.88, 1.03, and 1.12 — a 27% spread from random initialization alone. No sophisticated approach consistently beat this.

### 2. The timeout is the binding constraint

With 30 minutes of training, the 1.6M-param baseline trains fast on full 100k-point data and completes all 50 epochs. The larger models (11.8M, 21M) needed to subsample points to fit in memory and time — they got fewer effective epochs on less data.

### 3. Subsampling hurt more than it helped

Both thorfinn (15k points) and frieren (50k points) subsampled during training. The baseline trained on all 100k points. At this resolution, losing spatial detail was more harmful than the architectural improvements gained by using the freed-up compute.

### 4. GNN approaches mostly failed

Multiple agents tried GNN/EdgeConv architectures (thorfinn, edward, emma, haku, violet, fern, askeladd). Most crashed (import errors, OOM, k-NN computation too slow). Only thorfinn got it working, and it still lost to the baseline. Building k-NN graphs on 100k points is expensive and fragile.

### 5. Stochastic variance dominates

The 27% spread between identical baseline runs means that any single-run comparison is unreliable. To properly evaluate architectural improvements, you'd need multiple seeds — which the 30-minute timeout doesn't allow.

### 6. Per-component analysis

Across all agents, `Uy` (crossflow) is easiest to predict, `Ux` (streamwise) is hardest, and `Uz` is in between. This is consistent with the physics — streamwise velocity has the most turbulent variation, while crossflow is more laminar.

### What would actually help

To beat the baseline, you'd need an approach that:
- **Keeps full 100k resolution** (no subsampling)
- **Adds spatial interaction cheaply** (not O(N²) attention or expensive k-NN)
- **Converges in <30 min** on a single GPU
- **Uses the physics**: residual prediction from last timestep is sound but needs a model that converges fast enough to benefit from it

Candidates: local attention with fixed-radius neighborhoods, sparse graph methods, or hierarchical point cloud processing (PointNet++ style downsampling/upsampling).
