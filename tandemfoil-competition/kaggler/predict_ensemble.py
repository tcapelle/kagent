"""Ensemble inference: average predictions from multiple checkpoints.

Usage:
  python predict_ensemble.py --checkpoints ckpt1.pt ckpt2.pt ... --agent <name>
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from tqdm import tqdm

from data import X_DIM
from train import Transolver

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    checkpoints: list[str]
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4
    weights: list[float] | None = None  # parallel to checkpoints; defaults to uniform
    weights_file: str | None = None  # JSON: {split_name: [w1, w2, ...]} for per-split weights


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

models = []
for ckpt_path_s in cfg.checkpoints:
    ckpt_path = Path(ckpt_path_s)
    with open(ckpt_path.parent / "config.yaml") as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device).eval()
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    models.append(m)
    print(f"Loaded {ckpt_path}")
print(f"Ensemble size: {len(models)}")

per_split_weights = None
if cfg.weights_file:
    with open(cfg.weights_file) as f:
        raw = json.load(f)
    # Support keys named with or without the "test_" prefix; map to TEST_SPLITS keys.
    per_split_weights = {}
    for split in TEST_SPLITS:
        key = split if split in raw else split.replace("test_", "val_")
        ws = raw[key]
        assert len(ws) == len(models), f"weights[{key}] must align with checkpoints"
        t = torch.tensor(ws, dtype=torch.float32, device=device)
        per_split_weights[split] = t / t.sum()
    print(f"Per-split weights loaded from {cfg.weights_file}")
elif cfg.weights is not None:
    assert len(cfg.weights) == len(models), "weights must align with checkpoints"
    weights_t = torch.tensor(cfg.weights, dtype=torch.float32, device=device)
    weights_t = weights_t / weights_t.sum()
    print(f"Weights: {weights_t.tolist()}")
else:
    weights_t = torch.full((len(models),), 1.0 / len(models), device=device)
    print(f"Weights: uniform")

with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

# Write test_single_in_dist last so the scorer's glob (which keys on it)
# only detects this commit-dir once all other splits are already saved.
WRITE_ORDER = TEST_SPLITS[1:] + TEST_SPLITS[:1]
for split in WRITE_ORDER:
    test_dir = splits_dir / split
    test_files = sorted(test_dir.glob("*.pt"))
    print(f"{split}: {len(test_files)} samples")

    if per_split_weights is not None:
        w_split = per_split_weights[split]
    else:
        w_split = weights_t

    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_files), cfg.batch_size), desc=split, leave=False):
            batch_files = test_files[i:i + cfg.batch_size]
            samples = [torch.load(f, weights_only=True) for f in batch_files]
            xs = [s["x"] for s in samples]

            max_n = max(x.shape[0] for x in xs)
            B = len(xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            for j, x in enumerate(xs):
                x_pad[j, :x.shape[0]] = x.to(device)

            x_norm = (x_pad - x_mean) / x_std
            preds_norm = []
            for m in models:
                preds_norm.append(m({"x": x_norm})["preds"])
            stacked = torch.stack(preds_norm, dim=0)  # [K, B, N, 3]
            avg_norm = (w_split[:, None, None, None] * stacked).sum(dim=0)
            pred = avg_norm * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    tmp_path = output_dir / f"{split}.pt.tmp"
    torch.save(predictions, tmp_path)
    tmp_path.rename(output_dir / f"{split}.pt")
    print(f"  → {output_dir / f'{split}.pt'} ({len(predictions)} samples)")

print(f"\nAll ensemble predictions saved to {output_dir}")
