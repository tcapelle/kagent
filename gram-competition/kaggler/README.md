# GRaM Competition — 3D Airflow Velocity Prediction

## Problem

Predict future 3D velocity fields around Formula 1-style front wings. Given the velocity field for the first 5 time steps, predict the next 5 time steps.

Based on the [GRaM ICLR 2026 Workshop Competition](https://github.com/gram-competition/iclr-2026).

## Data

Pre-processed data on PVC at `/mnt/new-pvc/datasets/gram/splits/`:

```
splits/
├── train/*.pt              Each: {velocity_in: [5,N,3], velocity_out: [5,N,3], pos: [N,3], t: [10], idcs_airfoil: [M]}
├── val/*.pt                Same format (leaderboard ranks by val predictions)
└── stats.json              Normalization stats
```

No separate test split — the real competition holds out its own test set. We maximize training data and rank by validation performance.

### Per-sample fields

| Field | Shape | Description |
|-------|-------|-------------|
| `velocity_in` | `[5, 100000, 3]` | Input velocity field (first 5 timesteps) |
| `velocity_out` | `[5, 100000, 3]` | Target velocity field (next 5 timesteps) |
| `pos` | `[100000, 3]` | 3D spatial coordinates of mesh points |
| `t` | `[10]` | Time values for all 10 timesteps |
| `idcs_airfoil` | `[M]` | Indices into `pos` marking airfoil surface points (M varies per geometry) |

### Key physics

- **No-slip boundary condition**: velocity = (0,0,0) on the airfoil surface (`idcs_airfoil`). This is a strong physical prior.
- **Turbulence**: The hard part is predicting high-frequency turbulent components. The laminar flow is mostly captured by the input velocity field.
- **Geometry variation**: 181 different F1 front wing configurations (1-3 airfoils, varying positions and angles).

## Model contract

Your model must:
- **Input**: `velocity_in [B, 5, N, 3]`, `pos [B, N, 3]`, `t [B, 10]`, `idcs_airfoil list[tensor]`
- **Output**: `velocity_out [B, 5, N, 3]` — predicted future velocity field

N = 100,000 points per sample. This is large — you'll likely need to subsample, use efficient architectures, or both.

## Metrics

**Primary**: Mean L2 velocity error — `(pred - gt).norm(dim=3).mean(dim=(1,2))` averaged over samples.
**Also reported**: Per-component MAE (Ux, Uy, Uz).

## Memory

Each sample is ~24MB in float32 (100k points x 10 timesteps x 3 channels). With batch_size=1, a single forward pass needs significant VRAM. Plan accordingly:
- Subsample points (e.g., 10k-20k)
- Use mixed precision (AMP)
- Gradient checkpointing
- Small batch sizes with gradient accumulation

## Submission

Predictions are saved per split as `.pt` files:
```
/mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<commit>/val.pt
```

The `predict.py` template handles this — you just need to plug in your model.

## W&B Logging (required)

All runs log to W&B project `kagent-gram` with metrics:
- `train/loss` — MSE loss
- `val/l2_error` — mean L2 velocity error (primary metric)
- `val/mae_Ux`, `val/mae_Uy`, `val/mae_Uz` — per-component MAE

## Rules

- Training timeout: controlled by `MAX_TIMEOUT_MIN` env var (default 30 min)
- VRAM: 96GB. Don't OOM.
- `data.py` is read-only
- No new packages beyond `pyproject.toml`

## Files

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `data.py` | Data loader (`load_data`, `collate_fn`) | No |
| `train.py` | Training template — fill in your model | Yes |
| `predict.py` | Prediction template — adapt to your model | Yes |
| `README.md` | This file | Reference |
