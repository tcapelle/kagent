"""Generate predictions on the hidden test splits.

Loads the Transolver model defined in train.py, runs inference on the four
test splits, and writes per-split prediction tensors to the PVC.
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
from models import Transolver

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
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4
    enforce_noslip: bool = True


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

ckpt_dir = Path(cfg.checkpoint).parent
with open(ckpt_dir / "config.yaml") as f:
    model_config = yaml.safe_load(f)

model = Transolver(**model_config).to(device)
state = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded model from {cfg.checkpoint}")

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

# Feature index 12 is the is_surface flag inside x.
SURF_FEAT_IDX = 12

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
            for j, x in enumerate(xs):
                x_pad[j, :x.shape[0]] = x.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
            pred_norm = pred_norm.float()
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(xs):
                p = pred[j, :x.shape[0]].cpu()
                if cfg.enforce_noslip:
                    sf = surfs[j].bool().cpu()
                    p[sf, 0] = 0.0
                    p[sf, 1] = 0.0
                predictions.append(p)

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
