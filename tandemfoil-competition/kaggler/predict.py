"""Generate predictions on the hidden test splits.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <name>

  # Ensemble of multiple checkpoints (comma-separated, predictions averaged):
  python predict.py --checkpoint a/checkpoint.pt,b/checkpoint.pt --agent <name>
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
    """Generate test predictions from a trained checkpoint (or comma-separated ensemble)."""
    checkpoint: str  # path to best model checkpoint, or comma-separated list
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4
    weights: str | None = None  # comma-separated ensemble weights (default: equal)


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

ckpt_paths = [Path(p.strip()) for p in cfg.checkpoint.split(",") if p.strip()]
if cfg.weights:
    weights = [float(w) for w in cfg.weights.split(",")]
    assert len(weights) == len(ckpt_paths), "weights count must match ckpts"
    ws = sum(weights)
    weights = [w / ws for w in weights]
else:
    weights = [1.0 / len(ckpt_paths)] * len(ckpt_paths)
models = []
for ckpt_path in ckpt_paths:
    config_path = ckpt_path.parent / "config.yaml"
    with open(config_path) as f:
        mcfg = yaml.safe_load(f)
    m = Transolver(**mcfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    models.append(m)
    print(f"Loaded model from {ckpt_path}")
print(f"Ensemble of {len(models)} model(s)")

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

for split in TEST_SPLITS:
    test_dir = splits_dir / split
    test_files = sorted(test_dir.glob("*.pt"))
    print(f"{split}: {len(test_files)} samples")

    predictions = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in tqdm(range(0, len(test_files), cfg.batch_size), desc=split, leave=False):
            batch_files = test_files[i:i + cfg.batch_size]
            samples = [torch.load(f, weights_only=True) for f in batch_files]
            xs = [s["x"] for s in samples]

            max_n = max(x.shape[0] for x in xs)
            B = len(xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
            for j, x in enumerate(xs):
                x_pad[j, :x.shape[0]] = x.to(device)
                mask[j, :x.shape[0]] = True

            x_norm = (x_pad - x_mean) / x_std
            pred_norm = sum(
                w * m({"x": x_norm, "mask": mask})["preds"].float()
                for w, m in zip(weights, models)
            )
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
