"""Ensemble: average predictions across trained checkpoints (with weights).

Loads each checkpoint with its own runtime.yaml (so Cp / velocity normalization
specific to that run is honored), runs inference, and averages the per-sample
prediction tensors before writing one consolidated output set.

Run:
  python predict_ensemble.py --checkpoints "models/model-A/checkpoint.pt,models/model-B/checkpoint.pt" --agent askeladd
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
from model import Transolver

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
    checkpoints: str  # comma-separated checkpoint paths
    weights: str | None = None  # comma-separated weights, default uniform
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

ckpt_paths = [Path(p.strip()) for p in cfg.checkpoints.split(",") if p.strip()]
if cfg.weights:
    ws = [float(w) for w in cfg.weights.split(",")]
    assert len(ws) == len(ckpt_paths), "weights must match checkpoints"
    total = sum(ws)
    weights = [w / total for w in ws]
else:
    weights = [1.0 / len(ckpt_paths)] * len(ckpt_paths)
print(f"Ensembling {len(ckpt_paths)} checkpoints with weights {weights}")

# Load each model with its runtime info.
models = []
for ckpt in ckpt_paths:
    with open(ckpt.parent / "config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    rt_path = ckpt.parent / "runtime.yaml"
    rt = {"cp_normalize": False, "velocity_norm": False, "log_re_ref": 14.0}
    if rt_path.exists():
        with open(rt_path) as f:
            rt = yaml.safe_load(f) or rt
    m = Transolver(**model_cfg).to(device)
    m.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    m.eval()
    print(f"  - {ckpt} cp={rt.get('cp_normalize')} vel={rt.get('velocity_norm')}")
    models.append((m, rt))

# Base stats (used as fallback if a runtime.yaml doesn't override).
with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean_base = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std_base = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean_base = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std_base = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)


def _predict_one(model, rt, x_pad):
    cp = bool(rt.get("cp_normalize", False))
    vel = bool(rt.get("velocity_norm", False))
    log_re_ref = float(rt.get("log_re_ref", 14.0))
    if "y_mean" in rt:
        y_mean = torch.tensor(rt["y_mean"], dtype=torch.float32, device=device)
        y_std = torch.tensor(rt["y_std"], dtype=torch.float32, device=device)
    else:
        y_mean = y_mean_base
        y_std = y_std_base

    pred_norm = model({"x": (x_pad - x_mean_base) / x_std_base})["preds"]
    pred = pred_norm * y_std + y_mean
    log_re = x_pad[:, 0, 13]
    if cp:
        rfp = torch.exp(2.0 * (log_re - log_re_ref)).view(-1, 1, 1)
        pred = torch.cat([pred[..., :2], pred[..., 2:3] * rfp], dim=-1)
    if vel:
        rfv = torch.exp(log_re - log_re_ref).view(-1, 1, 1)
        pred = torch.cat([pred[..., :2] * rfv, pred[..., 2:3]], dim=-1)
    return pred


# Save predictions keyed by agent + commit hash.
agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

for split in TEST_SPLITS:
    test_dir = splits_dir / split
    test_files = sorted(test_dir.glob("*.pt"))
    print(f"{split}: {len(test_files)} samples")

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

            preds = [_predict_one(m, rt, x_pad) for m, rt in models]
            ws = torch.tensor(weights, device=device).view(-1, 1, 1, 1)
            pred_avg = (torch.stack(preds, dim=0) * ws).sum(dim=0)

            for j, x in enumerate(xs):
                predictions.append(pred_avg[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path}")

print(f"\nAll predictions saved to {output_dir}")
