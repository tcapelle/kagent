# CFD Competition — Organizer Guide

How the data was prepared, baseline was trained, and predictions are scored.

## 1. Data Preparation

### Raw data

The TandemFoilSet lives on the PVC at `/mnt/new-pvc/datasets/tandemfoil/` — 7 pickle files (~40GB total) containing PyG graph objects with CFD simulation results for single and tandem airfoil configurations.

| File | Samples | Description |
|------|---------|-------------|
| `raceCar_single_randomFields.pickle` | 899 | Single foil, variable Re |
| `raceCar_randomFields_mgn_Part{1,2,3}.pickle` | 300 each | Tandem, front foils: NACA2412/6416/9412 |
| `cruise_randomFields_mgn_Part{1,2,3}.pickle` | 300 each | Tandem cruise, Re: 1.475M/4.445M/802K |

### Split strategy

`prepare_splits.py` splits data into train/val/test with `SAMPLE_FRACTION=0.85` (15% held out for test).

The validation set is structured into 4 tracks testing different generalization axes:

- **val_single_in_dist** — random holdout from single-foil (sanity check)
- **val_geom_camber_rc** — unseen front foil camber M=6-8 (raceCar tandem Part2, full holdout)
- **val_geom_camber_cruise** — unseen front foil camber M=2-4 (cruise tandem Part2, full holdout)
- **val_re_rand** — stratified Re holdout across all tandem domains

See [SPLITS.md](SPLITS.md) for the full split design rationale.

### Preprocessing

Each raw PyG sample is converted to a 24-dim input feature vector per mesh node:
`[pos(2), saf(2), dsdf(8), is_surface(1), log_Re(1), AoA0(1), NACA0(3), AoA1(1), NACA1(3), gap(1), stagger(1)]`

Surface IDs include boundary 7 (foil 2) which was missing in the original prepare.py.

### Running the split

```bash
# On the cluster (needs PVC access):
uv run k8s/launch.py --tag <tag> --prepare

# Or directly on a pod with PVC:
uv run prepare_splits.py
```

Output: `/mnt/new-pvc/datasets/tandemfoil/splits_v2/` with per-sample .pt files, stats.json, and meta.json.

The job takes ~5 minutes on 8 CPU cores.

### Result (with SAMPLE_FRACTION=0.85)

| Split | Samples |
|-------|---------|
| train | 1,499 |
| val (4 x 100) | 400 |
| test (4 x 200) | 800 |
| **Total** | **2,699** |

## 2. Baseline Training

The baseline Transolver model is in `organizer/train.py` (not given to kagglers). It was trained for 10 epochs via a one-off k8s job. To re-run it, copy `data.py` and `viz.py` from `kaggler/` into the same directory, then:

```bash
cd cfd-competition/organizer
python train.py --agent baseline --wandb_name "baseline/transolver" --epochs 10
```

W&B project: `wandb-applied-ai-team/kagent-v2`

Key model config:
- `n_hidden=128`, `n_layers=5`, `n_head=4`, `slice_num=64`
- `lr=5e-4`, `weight_decay=1e-4`, `batch_size=4`, `surf_weight=10.0`
- Balanced domain sampling (racecar_single / racecar_tandem / cruise equally weighted)

## 3. Scoring Predictions

Kagglers submit predictions via `predict.py`, which saves to `/mnt/new-pvc/predictions/<tag>/<agent>/<model-id>/predictions.pt`.

Score with:

```bash
uv run score.py --predictions /mnt/new-pvc/predictions/<tag>/<agent>/<model-id>/predictions.pt
```

Output is a markdown table with per-domain and overall MAE for surface and volume nodes.

## Files

| File | Purpose |
|------|---------|
| `prepare_splits.py` | One-time data preprocessing + split |
| `score.py` | Score predictions against hidden ground truth |
| `README.md` | This file |
