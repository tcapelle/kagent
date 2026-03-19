"""Generate predictions on the hidden test set.

Adapt this to your model. The key contract:
  - Load your model from a checkpoint
  - Run inference on test/*.pt files
  - Save predictions to PVC at /mnt/new-pvc/predictions/<agent>/<commit>/predictions.pt

Run:
  uv run predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm

from data import X_DIM

PREDICTIONS_DIR = Path("/mnt/new-pvc/predictions")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits")


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str  # path to best model checkpoint
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None  # kaggler name for output path
    batch_size: int = 4
    hidden: int = 320
    n_blocks: int = 8


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

from train import CFDModel
model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks).to(device)
state_dict = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
# Strip _orig_mod. prefix from torch.compile checkpoints
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.eval()
print(f"Loaded model from {cfg.checkpoint}")

# Load stats
with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

# Load test inputs
test_files = sorted((splits_dir / "test").glob("*.pt"))
print(f"Test samples: {len(test_files)}")

# Run inference
predictions = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_files), cfg.batch_size), desc="Predicting"):
        batch_files = test_files[i:i + cfg.batch_size]
        samples = [torch.load(f, weights_only=True) for f in batch_files]
        xs = [s["x"] for s in samples]

        max_n = max(x.shape[0] for x in xs)
        B = len(xs)
        x_pad = torch.zeros(B, max_n, X_DIM, device=device)
        for j, x in enumerate(xs):
            x_pad[j, :x.shape[0]] = x.to(device)

        pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
        pred = pred_norm * y_std + y_mean

        for j, x in enumerate(xs):
            predictions.append(pred[j, :x.shape[0]].cpu())

# Save predictions keyed by agent + commit hash
agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"

output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "predictions.pt"
torch.save(predictions, output_path)
print(f"Saved {len(predictions)} predictions to {output_path}")
