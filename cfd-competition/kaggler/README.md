# CFD Surrogate Competition

## Problem

Train a neural network surrogate for computational fluid dynamics (CFD) on the [TandemFoilSet](https://openreview.net/forum?id=4Z0P4Nbosn) dataset. Given tandem-airfoil geometry and flow conditions, predict the full velocity (Ux, Uy) and pressure (p) field at every mesh node.

The baseline is a [Transolver](https://arxiv.org/abs/2402.02366) with physics-aware attention over irregular meshes. Beat it.

## Data

Pre-processed data lives on the PVC at `/mnt/new-pvc/datasets/tandemfoil/splits_v2/`:

```
splits_v2/
├── train/000000.pt ...           Each: {x: [N,24], y: [N,3], is_surface: [N]}
├── val_single_in_dist/...        Random holdout from single-foil (sanity check)
├── val_geom_camber_rc/...        Unseen front foil camber M=6-8 (raceCar)
├── val_geom_camber_cruise/...    Unseen front foil camber M=2-4 (cruise)
├── val_re_rand/...               Stratified Re holdout across all tandem domains
├── test_*/...                    4 test splits (same axes as val, no targets)
├── stats.json                    Normalization stats (x_mean, x_std, y_mean, y_std)
└── meta.json                     Split counts, domain groups
```

### Input features (x, 24 dimensions)

| Dims | Feature |
|------|---------|
| 0-1 | Node position (x, z) |
| 2-3 | Signed arc-length (saf) |
| 4-11 | Distance-based shape descriptor (dsdf) |
| 12 | Is surface node (0/1) |
| 13 | log(Re) |
| 14 | AoA foil 1 (radians) |
| 15-17 | NACA foil 1 (camber, position, thickness) |
| 18 | AoA foil 2 (radians, 0 for single-foil) |
| 19-21 | NACA foil 2 (0,0,0 for single-foil) |
| 22 | Gap between foils (0 for single-foil) |
| 23 | Stagger between foils (0 for single-foil) |

### Targets (y, 3 dimensions)

| Channel | Description |
|---------|-------------|
| 0 | Ux — velocity x-component |
| 1 | Uy — velocity y-component |
| 2 | p — kinematic pressure (p/rho, m²/s²) |

### Loading data

```python
from data import load_data

train_ds, val_splits, stats, sample_weights = load_data()
# train_ds[i] → (x, y, is_surface)
# val_splits["val_single_in_dist"][i] → (x, y, is_surface)
# stats = {x_mean, x_std, y_mean, y_std}
# sample_weights → for balanced domain sampling
```

### Batching and padding

Samples have **variable mesh sizes** (74K to 242K nodes). The dataloader pads each batch to the largest sample using `pad_collate`, which returns:

```python
x, y, is_surface, mask = batch
# x:          [B, N_max, 24]  — padded with zeros
# y:          [B, N_max, 3]   — padded with zeros
# is_surface: [B, N_max]      — False for padding
# mask:       [B, N_max]      — True for real nodes, False for padding
```

**The `mask` tensor is critical.** Your model output includes predictions for padding positions — these must be excluded from loss and metrics. The training template handles this correctly. If you write custom loss or pooling, always use `mask` to ignore padding.

### Dataset domains

The training data spans three physical domains with different mesh sizes and flow regimes:

| Domain | Samples | Mesh nodes (mean) | Description |
|--------|---------|-------------------|-------------|
| RaceCar single | ~688 train | ~86K | Single airfoil, Re ~700K–2M, AoA ±10° |
| RaceCar tandem | ~510 train | ~127K | Dual foils (Parts 1+3), Re ~700K–2M |
| Cruise | ~408 train | ~208K | Tandem cruise foils (Parts 1+3), Re 802K–1.475M |

The three domains are **equally weighted** in training via a balanced sampler — otherwise raceCar single would dominate.

### Physics context

Each sample is a 2D CFD simulation over an overset mesh with up to 3 zones:

```
┌─────────────────────────────────────────────────┐
│  Zone 0 — coarse background (full domain)       │
│                                                   │
│       ┌──────────────┐   ┌──────────────┐        │
│       │  Zone 1       │   │  Zone 2       │       │
│       │  (dense,      │   │  (dense,      │       │
│       │  foil 1)      │   │  foil 2)      │       │
│       └──────────────┘   └──────────────┘        │
│                                                   │
└─────────────────────────────────────────────────┘
```

**Boundary types** in `is_surface`:
- IDs 5, 6 = foil 1 surface (upper/lower)
- ID 7 = foil 2 surface (tandem only)
- IDs 0–4 = interior, inlet, outlet, top/bottom walls

**Value ranges** vary dramatically across domains:

| Domain | Re range | y range (approx) | y std |
|--------|----------|-------------------|-------|
| Cruise Part1 (Re=1.475M) | 1.475M | [-1,278, 233] | 55 |
| Cruise Part2 (Re=4.445M) | 4.445M | [-2,360, 2,118] | 304 |
| Cruise Part3 (Re=802K) | 802K | [-300, 69] | 17 |
| RaceCar single | ~700K–2M | [-874, 467] | 141 |
| RaceCar tandem | ~700K–2M | [-4,277, 668] | 235 |

The wide pressure range across Re numbers is a key challenge — the model must handle both low-Re (small values) and high-Re (extreme values) regimes.

### Parameter space

- **Reynolds number**: 802K to 4.445M (training sees up to ~1.5M; Part2 at 4.445M is OOD)
- **NACA profiles**: 4-digit codes encoding camber, position, thickness. Single-foil sweeps ~2205–2209; tandem has fixed front foils per Part (2412/6416/9412)
- **AoA**: ±8° (cruise) to ±10° (raceCar), per-foil for tandem
- **Gap/stagger**: tandem geometry parameters — gap ~[-0.8, 1.3], stagger ~[0.7, 2.0]

## Metrics

**Goal: lowest validation losses.** We track:
- **Surface MAE** — mean absolute error on airfoil surface nodes (Ux, Uy, p). **Most important.**
- **Volume MAE** — mean absolute error on volume (field) nodes.
- **val/loss** — combined: `vol_loss + surf_weight * surf_loss`.

Lower is better. Surface pressure accuracy matters most.

Four validation tracks test different failure modes:

| Track | Tests |
|-------|-------|
| `val_single_in_dist` | Random holdout from single-foil (sanity check) |
| `val_geom_camber_rc` | Unseen front foil camber M=6-8 — geometry interpolation (raceCar) |
| `val_geom_camber_cruise` | Unseen front foil camber M=2-4 — geometry interpolation (cruise) |
| `val_re_rand` | Stratified Re holdout across all tandem domains — cross-regime generalization |

## Model contract

Your model must:
- **Input**: `{"x": tensor [B, N, 24]}` — batch of normalized node features
- **Output**: `{"preds": tensor [B, N, 3]}` — predicted `[Ux, Uy, p]` in **normalized** space (same space as `(y - y_mean) / y_std`)

The training template handles normalization for you:
```python
# Normalize inputs
x = (x - stats["x_mean"]) / stats["x_std"]
# Normalize targets for loss computation
y_norm = (y - stats["y_mean"]) / stats["y_std"]
# Your model predicts in normalized space
pred = model({"x": x})["preds"]  # [B, N, 3] in normalized space
# Denormalize for MAE computation
pred_phys = pred * stats["y_std"] + stats["y_mean"]
```

## Submission

After training, generate predictions on the hidden test set. The test samples are in `splits_v2/test_*/` (4 test splits matching the val axes) — they contain `{x, is_surface}` but **no y** (targets).

Your `predict.py` must:
1. Load each test sample's `x` tensor `[N, 24]`
2. Normalize: `(x - x_mean) / x_std`
3. Run your model to get predictions `[N, 3]` in normalized space
4. **Denormalize** back to physical units: `pred * y_std + y_mean`
5. Save per-split `.pt` files (list of `[N_i, 3]` tensors, one per sample)

The output layout must be:
```
/mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<commit>/
├── test_single_in_dist.pt      # list of 200 tensors [N_i, 3]
├── test_geom_camber_rc.pt
├── test_geom_camber_cruise.pt
└── test_re_rand.pt
```

Each `.pt` file contains:
```python
# List of 200 tensors in PHYSICAL units (not normalized)
# Channels: [Ux (m/s), Uy (m/s), p (m²/s²)]
predictions: list[torch.Tensor]  # len=200, each shape [N_i, 3]
```

The `predict.py` template handles padding, batching, denormalization, and saving — you just need to plug in your model.

```bash
# Commit your code first (predict.py uses the commit hash for the output path)
git add -A && git commit -m "my model v1" && git push
uv run predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
```

## W&B Logging (required)

A W&B skill is available at `.claude/skills/wandb-primary/` with helpers for querying runs, comparing configs, and analyzing metrics. Use it to review your previous results between iterations.

All training runs **must** log to W&B project `kagent-v2` with these metrics:

```python
import wandb

run = wandb.init(
    entity="wandb-applied-ai-team",
    project="kagent-v2",
    name="<your-name>/<description>",
    tags=["<your-name>"],
)
```

### Required metrics

Log these every epoch with `global_step` as the x-axis:

**Training (per epoch):**
- `train/loss` — combined training loss
- `train/vol_loss` — volume MSE
- `train/surf_loss` — surface MSE

**Validation (per epoch, per split):**

For each split in `[val_single_in_dist, val_geom_camber_rc, val_geom_camber_cruise, val_re_rand]`:
- `{split}/loss` — combined: vol_loss + surf_weight * surf_loss
- `{split}/vol_loss`, `{split}/surf_loss` — component losses
- `{split}/mae_surf_Ux`, `{split}/mae_surf_Uy`, `{split}/mae_surf_p` — surface MAE per channel (physical units)
- `{split}/mae_vol_Ux`, `{split}/mae_vol_Uy`, `{split}/mae_vol_p` — volume MAE per channel

**Aggregated:**
- `val/loss` — mean of `{split}/loss` across all 4 splits (used for checkpoint selection)

**How to compute MAE in physical units:**

```python
# Denormalize predictions back to physical units
pred_phys = pred_norm * stats["y_std"] + stats["y_mean"]
err = (pred_phys - y).abs()  # y is already in physical units

# Surface MAE: average over surface nodes
mae_surf = (err * surf_mask.unsqueeze(-1)).sum(dim=(0,1)) / surf_mask.sum()
# Volume MAE: average over non-surface nodes
mae_vol = (err * vol_mask.unsqueeze(-1)).sum(dim=(0,1)) / vol_mask.sum()
```

### W&B setup boilerplate

```python
wandb.define_metric("global_step")
wandb.define_metric("train/*", step_metric="global_step")
wandb.define_metric("val/*", step_metric="global_step")
for split in ["val_single_in_dist", "val_geom_camber_rc", "val_geom_camber_cruise", "val_re_rand"]:
    wandb.define_metric(f"{split}/*", step_metric="global_step")
```

## Rules

- **Timeout**: Training capped at 30 minutes. Don't override this.
- **VRAM**: GPUs have 96GB. Don't OOM.
- **Simplicity**: All else equal, simpler is better.
- **No new packages** beyond `pyproject.toml`.
- **`data.py` is read-only** — don't change the data loading interface.
- **Log to W&B** with the required metrics above. Runs without proper logging can't be compared.
- Write your own `train.py` — you choose the architecture, loss, optimizer, augmentation. Use `data.py` to load data.

## Files

| File | Purpose | Modifiable? |
|------|---------|-------------|
| `data.py` | Data loader (`load_data`, `pad_collate`) | No |
| `train.py` | Training template — fill in your model | Yes — this is your playground |
| `predict.py` | Prediction template — adapt to your model | Yes |
| `viz.py` | Flow field visualization | Yes |
| `README.md` | This file — competition description + rules | Reference |
