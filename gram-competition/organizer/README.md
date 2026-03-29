# GRaM Competition — Organizer Guide

## 1. Data Preparation

### Source
181 simulations of 3D airflow around F1 front wings, from the [GRaM ICLR 2026 competition](https://github.com/gram-competition/iclr-2026). Dataset: `gram-competition/warped-ifw` on HuggingFace (~14GB, gated).

### Download
```bash
# Requires HF_TOKEN with access to gated dataset
huggingface-cli download gram-competition/warped-ifw --repo-type dataset --local-dir /mnt/new-pvc/datasets/gram/raw
```

### Split and preprocess
```bash
uv run k8s/launch.py --tag <tag> --competition gram-competition --prepare
```

Splits by simulation ID (90/10 train/val) with seed 42. No test split — the real competition holds out its own. Output: `/mnt/new-pvc/datasets/gram/splits/`.

## 2. Scoring

```bash
python score.py --score_all
```

Metric: mean L2 velocity error — `(pred - gt).norm(dim=2).mean()` averaged over samples.

## 3. Launch

```bash
uv run k8s/launch.py --tag <tag> --competition gram-competition --n_kagglers 4 --organizer
```
