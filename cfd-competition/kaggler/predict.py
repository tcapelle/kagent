"""Generate predictions on the hidden test set.

Supports single checkpoint or ensemble of multiple checkpoints.

Run:
  uv run predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
  uv run predict.py --checkpoint_list ensemble_checkpoints.txt --agent <your-name>
"""

import json
import subprocess
from dataclasses import dataclass, field
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
    checkpoint: str = ""  # path to best model checkpoint
    checkpoint_list: str = ""  # path to file with list of checkpoints (one per line)
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None  # kaggler name for output path
    batch_size: int = 4
    hidden: int = 256
    n_blocks: int = 8
    top_k: int = 10  # number of checkpoints to ensemble (if using checkpoint_list)


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

from train import CFDModel

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

# Pre-load all test data to avoid re-reading for each model
test_samples = [torch.load(f, weights_only=True) for f in tqdm(test_files, desc="Loading test data")]
test_xs = [s["x"] for s in test_samples]
test_ns = [x.shape[0] for x in test_xs]


def predict_with_model(model):
    """Run inference with a single model, return list of per-sample predictions."""
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_xs), cfg.batch_size):
            batch_xs = test_xs[i:i + cfg.batch_size]
            max_n = max(x.shape[0] for x in batch_xs)
            B = len(batch_xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            for j, x in enumerate(batch_xs):
                x_pad[j, :x.shape[0]] = x.to(device)

            pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(batch_xs):
                preds.append(pred[j, :x.shape[0]].cpu())
    return preds


def load_model(checkpoint_path):
    """Load a single model from checkpoint."""
    model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


# Determine checkpoints to use
if cfg.checkpoint_list:
    with open(cfg.checkpoint_list) as f:
        all_checkpoints = [line.strip() for line in f if line.strip()]
    # Use top_k checkpoints (randomly sampled if more than top_k)
    import random
    if len(all_checkpoints) > cfg.top_k:
        checkpoints = random.sample(all_checkpoints, cfg.top_k)
    else:
        checkpoints = all_checkpoints
    print(f"Ensemble: {len(checkpoints)} checkpoints")
elif cfg.checkpoint:
    checkpoints = [cfg.checkpoint]
    print(f"Single checkpoint: {cfg.checkpoint}")
else:
    raise ValueError("Must specify --checkpoint or --checkpoint_list")

# Run ensemble prediction
ensemble_preds = None
for idx, ckpt in enumerate(checkpoints):
    print(f"Model {idx+1}/{len(checkpoints)}: {ckpt}")
    model = load_model(ckpt)
    preds = predict_with_model(model)
    del model
    torch.cuda.empty_cache()

    if ensemble_preds is None:
        ensemble_preds = preds
    else:
        for i in range(len(preds)):
            ensemble_preds[i] = ensemble_preds[i] + preds[i]

# Average
predictions = [p / len(checkpoints) for p in ensemble_preds]

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
