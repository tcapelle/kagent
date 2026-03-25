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
    checkpoint: str = ""  # path to best model checkpoint (unused for ensemble)
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None  # kaggler name for output path
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

from train import CFDModel

# 31-model ensemble: 6 architectures x various surface weights
configs_and_checkpoints = [
    # h576x4 (L1-finetuned, sw20) - 5 models
    (576, 4, 'models/model-2m8nw7ay/checkpoint.pt'),
    (576, 4, 'models/model-zvebdom8/checkpoint.pt'),
    (576, 4, 'models/model-a1t4433a/checkpoint.pt'),
    (576, 4, 'models/model-k3w2fzq7/checkpoint.pt'),
    (576, 4, 'models/model-5u41rg2i/checkpoint.pt'),
    # h384x8 - 6 models (various sw)
    (384, 8, 'models/model-zlurpzuy/checkpoint.pt'),  # sw20
    (384, 8, 'models/model-6dymv1xq/checkpoint.pt'),  # sw20
    (384, 8, 'models/model-ic7ik3y7/checkpoint.pt'),  # sw10
    (384, 8, 'models/model-85bifouh/checkpoint.pt'),  # sw15
    (384, 8, 'models/model-99444jke/checkpoint.pt'),  # sw25
    (384, 8, 'models/model-07mei2kq/checkpoint.pt'),  # sw5
    # h256x12 - 5 models (various sw)
    (256, 12, 'models/model-fnjk0rb9/checkpoint.pt'),  # sw20
    (256, 12, 'models/model-1srfjih9/checkpoint.pt'),  # sw15
    (256, 12, 'models/model-rv20cdeb/checkpoint.pt'),  # sw10
    (256, 12, 'models/model-jasa0nov/checkpoint.pt'),  # sw25
    (256, 12, 'models/model-vte4od4j/checkpoint.pt'),  # sw5
    # h448x6 - 5 models (various sw)
    (448, 6, 'models/model-ykoir39f/checkpoint.pt'),  # sw20
    (448, 6, 'models/model-n6zkj9lt/checkpoint.pt'),  # sw15
    (448, 6, 'models/model-jmmn1xmo/checkpoint.pt'),  # sw10
    (448, 6, 'models/model-0farx8wu/checkpoint.pt'),  # sw25
    (448, 6, 'models/model-yhrdxpm6/checkpoint.pt'),  # sw5
    # h192x16 - 5 models (various sw)
    (192, 16, 'models/model-aevg54og/checkpoint.pt'),  # sw20
    (192, 16, 'models/model-itehj77h/checkpoint.pt'),  # sw10
    (192, 16, 'models/model-szaqzjj2/checkpoint.pt'),  # sw5
    (192, 16, 'models/model-8qn718k1/checkpoint.pt'),  # sw15
    (192, 16, 'models/model-nzwkykno/checkpoint.pt'),  # sw25
    # h640x3 - 5 models (various sw)
    (640, 3, 'models/model-p16uv4jl/checkpoint.pt'),  # sw20
    (640, 3, 'models/model-b1n9slsl/checkpoint.pt'),  # sw10
    (640, 3, 'models/model-qeynup6w/checkpoint.pt'),  # sw5
    (640, 3, 'models/model-1bz9b0x6/checkpoint.pt'),  # sw15
    (640, 3, 'models/model-unkgk0bq/checkpoint.pt'),  # sw25
]

# Load all models
models = []
for hidden, n_blocks, ckpt in configs_and_checkpoints:
    m = CFDModel(in_dim=X_DIM, out_dim=3, hidden=hidden, n_blocks=n_blocks).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    models.append(m)
    print(f"Loaded {ckpt} (h{hidden}x{n_blocks})")

print(f"Ensemble: {len(models)} models")

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

# Run inference with trimmed mean (trim=3)
TRIM = 4
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

        x_norm = (x_pad - x_mean) / x_std

        # Get predictions from all models
        all_preds = []
        for m in models:
            pred_norm = m({"x": x_norm})["preds"]
            pred = pred_norm * y_std + y_mean
            all_preds.append(pred)

        # Trimmed mean: sort, remove top/bottom TRIM, average rest
        stacked = torch.stack(all_preds, dim=0)  # [M, B, N, 3]
        sorted_preds = stacked.sort(dim=0).values
        trimmed = sorted_preds[TRIM:-TRIM]  # remove top/bottom TRIM
        ensemble_pred = trimmed.mean(dim=0)

        for j, x in enumerate(xs):
            predictions.append(ensemble_pred[j, :x.shape[0]].cpu())

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
