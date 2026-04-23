"""Generate predictions on the hidden test splits.

Loads a trained Transolver checkpoint (with accompanying config.yaml),
runs inference on each test split, and writes per-split .pt files to PVC.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
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
from train import Transolver, apply_no_slip

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
    checkpoint: str  # path to best model checkpoint
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 2
    no_slip_bc: bool = False


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load architecture config from the checkpoint's sibling yaml
ckpt_path = Path(cfg.checkpoint)
config_path = ckpt_path.parent / "config.yaml"
with open(config_path) as f:
    model_config = yaml.safe_load(f)

model = Transolver(**model_config).to(device)
state = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded model from {cfg.checkpoint}  ({model_config})")

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
    with torch.no_grad():
        for i in tqdm(range(0, len(test_files), cfg.batch_size), desc=split, leave=False):
            batch_files = test_files[i:i + cfg.batch_size]
            samples = [torch.load(f, weights_only=True) for f in batch_files]
            xs = [s["x"] for s in samples]
            surfs = [s["is_surface"] for s in samples]

            max_n = max(x.shape[0] for x in xs)
            B = len(xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            surf_pad = torch.zeros(B, max_n, dtype=torch.bool, device=device)
            for j, (x, sf) in enumerate(zip(xs, surfs)):
                x_pad[j, :x.shape[0]] = x.to(device)
                surf_pad[j, :sf.shape[0]] = sf.to(device)

            pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
            if cfg.no_slip_bc:
                pred_norm = apply_no_slip(pred_norm, surf_pad, y_mean, y_std)
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
