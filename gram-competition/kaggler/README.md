# GRaM Competition — 3D Airflow Velocity Prediction

## Problem

Predict the **future 3D velocity field** around Formula 1-style front wings. Given the velocity field u(t, x) for the first half of a time window (5 time steps), predict the velocity field for the second half (5 time steps).

Based on the [GRaM ICLR 2026 Workshop Competition](https://github.com/gram-competition/iclr-2026). Source data: [gram-competition/warped-ifw](https://huggingface.co/datasets/gram-competition/warped-ifw) on HuggingFace.

### Physics background

The dataset contains 162 simulations of airflow around F1 front wings derived from the Imperial Front Wing (IFW) geometry. Each simulation uses a different geometric configuration (1, 2, or 3 airfoils at varying positions and pitch angles) with constant freestream velocity.

The aerodynamic dynamics decompose into:
- **Low-frequency laminar components** — easy to predict (the input velocity field is a strong prior)
- **High-frequency turbulent components** — the hard part; vortex shedding, wake turbulence, flow separation

**No-slip boundary condition**: velocity = (0, 0, 0) on the airfoil surface. This is a hard physical constraint that models should exploit.

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
| `velocity_in` | `[5, 100000, 3]` | Input velocity field (first 5 timesteps). Components: (Ux, Uy, Uz) in m/s |
| `velocity_out` | `[5, 100000, 3]` | Target velocity field (next 5 timesteps). Same components |
| `pos` | `[100000, 3]` | 3D spatial coordinates (x, y, z) of mesh points. Fixed per geometry |
| `t` | `[10]` | Time values for all 10 timesteps (5 input + 5 output) |
| `idcs_airfoil` | `[M]` | Indices into `pos` marking airfoil surface points. M varies per geometry (~8k-24k) |

### Dataset statistics

- **162 unique geometries** (F1 front wing configurations)
- **810 total samples** (5 time windows per simulation)
- **Split by simulation ID**: 146 train sims (730 samples) / 16 val sims (80 samples)
- **100,000 points per sample** — this is large, plan memory accordingly
- **Velocity stats** (train set): mean ≈ [35.6, 0.5, 1.9] m/s, std ≈ [20.1, 7.2, 9.5] m/s

### Normalization

Stats are in `stats.json`:
```python
stats = load_data()[2]  # {"vel_mean": [3], "vel_std": [3]}
```
Use these to normalize velocity fields if your model benefits from it. The competition metric is evaluated on **raw (unnormalized)** velocity, so denormalize predictions before submission.

## Model contract

Your model must accept these inputs and produce this output:

```python
def forward(self, velocity_in, pos, t, idcs_airfoil):
    # velocity_in: [B, 5, N, 3]  — past velocity field
    # pos:         [B, N, 3]     — spatial coordinates
    # t:           [B, 10]       — time values
    # idcs_airfoil: list[tensor] — surface point indices per sample
    # Returns:     [B, 5, N, 3]  — predicted future velocity field
```

**Note on the real competition**: their model signature is `model(t, pos, idcs_airfoil, velocity_in)` — different argument order. If you submit to the real competition, wrap your model accordingly.

## Metrics

### Primary: Mean L2 velocity error

```python
# Competition hint metric (exact code from their main.py):
metric = (velocity_out - ground_truth).norm(dim=3).mean(dim=(1, 2))
# dim=3: L2 norm over (Ux, Uy, Uz) per point per timestep
# dim=(1,2): average over timesteps and spatial points
# Result: one scalar per sample in the batch
```

**Lower is better.** The leaderboard ranks by this metric averaged over all val samples.

### Secondary: Per-component MAE

Also reported for debugging: `mae_Ux`, `mae_Uy`, `mae_Uz` — absolute error averaged over time and space per velocity component. Helps identify which component your model struggles with.

## Memory & Performance

Each sample is **~24 MB** in float32 (100k points × 10 timesteps × 3 channels). With batch_size=1, a single forward pass through a naive model already needs significant VRAM.

**Strategies to manage memory:**
- **Subsample points**: Random or farthest point sampling (FPS). 10k-20k points is a reasonable trade-off. Importance-weight near the airfoil surface.
- **Mixed precision (AMP)**: `torch.cuda.amp.autocast()` halves memory for most operations.
- **Gradient checkpointing**: Trade compute for memory with `torch.utils.checkpoint`.
- **Gradient accumulation**: Simulate larger batch sizes without holding multiple samples in memory.
- **Process time steps independently**: Instead of [B, 5, N, 3] as one tensor, loop over time steps.

The baseline MLP with 256 hidden and 6 blocks uses ~4 GB VRAM with batch_size=1 at full 100k resolution.

## Baseline model

The provided baseline is a **ResMLP** (residual MLP):
- Input: concat `pos` (3) + `velocity_in` flattened (5×3=15) = 18 features per point
- Architecture: Linear(18→256) → 6 × ResBlock(256) → LayerNorm → Linear(256→15)
- Output: reshape to [B, 5, N, 3]
- Does **not** use `t` or `idcs_airfoil` — easy wins for agents

This is deliberately simple. It treats each point independently (no spatial interaction). There is huge room for improvement.

## Submission

Predictions are saved per split as `.pt` files:
```
/mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<commit>/val.pt
```

The `predict.py` template handles this. `train.py` auto-runs `predict.py` after training.

## W&B Logging (required)

All runs log to W&B project `kagent-gram`:
- `train/loss` — MSE loss per batch
- `val/l2_error` — mean L2 velocity error (primary metric, used for leaderboard)
- `val/mae_Ux`, `val/mae_Uy`, `val/mae_Uz` — per-component MAE

## Rules

- Training timeout: controlled by `MAX_TIMEOUT_MIN` env var (default 30 min)
- VRAM: 96GB. Don't OOM.
- `data.py` is read-only
- No new packages beyond `pyproject.toml`

## Ideas to explore

**Architecture**:
- Transformers with local attention (full attention on 100k points is O(N²) — too expensive)
- U-Net style encode-decode with point cloud downsampling/upsampling 
- Diffusion models
- ab-upt and Transolver models

**Physics priors**:
- **No-slip enforcement**: zero out velocity at `idcs_airfoil` indices as a hard post-processing step
- **Residual prediction**: predict `delta = velocity_out - velocity_in[-1]` instead of absolute velocity. The last input timestep is a strong prior for the output.
- **Pressure** as auxiliary input (available in raw data but not in our preprocessed splits)

## Files

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `data.py` | Data loader (`load_data`, `collate_fn`, `GRAMDataset`) | No |
| `train.py` | Training with baseline ResMLP — improve the model | Yes |
| `predict.py` | Prediction on val split — adapt to your model | Yes |
| `README.md` | This file | Reference |
