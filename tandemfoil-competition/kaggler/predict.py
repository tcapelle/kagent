"""Generate predictions on the hidden test splits.

Output layout:
  /mnt/new-pvc/predictions/<tag>/<agent>/<commit>/
  ├── test_single_in_dist.pt
  ├── test_geom_camber_rc.pt
  ├── test_geom_camber_cruise.pt
  └── test_re_rand.pt

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
"""

import json
import os
import shutil
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
PVC_CHECKPOINT_ROOT = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}")

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint.

    Pass a single `--checkpoint` for one-model prediction, or
    `--checkpoints a,b,c` (comma-separated) to ensemble (mean of preds).
    All ensembled checkpoints must share the same model_config.
    """
    checkpoint: str = ""
    checkpoints: str = ""  # comma-separated paths for ensemble
    weights: str = ""  # comma-separated weights for ensemble (default uniform)
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1  # bs=1 avoids padding-driven attention degradation


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

ckpt_paths = [Path(p) for p in cfg.checkpoints.split(",") if p.strip()] \
    if cfg.checkpoints else [Path(cfg.checkpoint)]
assert ckpt_paths, "Must pass --checkpoint or --checkpoints"
weights = [float(w) for w in cfg.weights.split(",") if w.strip()] \
    if cfg.weights else [1.0] * len(ckpt_paths)
assert len(weights) == len(ckpt_paths), f"Got {len(weights)} weights for {len(ckpt_paths)} checkpoints"
total_w = sum(weights)
ckpt_path = ckpt_paths[0]  # for downstream config + mirror
cfg_path = ckpt_path.parent / "config.yaml"
with open(cfg_path) as f:
    model_config = yaml.safe_load(f)

models = []
for p in ckpt_paths:
    m = Transolver(**model_config).to(device)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    m.eval()
    models.append(m)
    print(f"Loaded model from {p}")
print(f"Ensemble of {len(models)} model(s), weights={weights}")

with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_amp else torch.float32

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

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                x_in = (x_pad - x_mean) / x_std
                pred_norm = sum(w * m({"x": x_in})["preds"].float() for w, m in zip(weights, models)) / total_w
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")

pvc_ckpt_dir = PVC_CHECKPOINT_ROOT / agent_name / "checkpoints" / ckpt_path.parent.name
pvc_ckpt_dir.mkdir(parents=True, exist_ok=True)
shutil.copy(ckpt_path, pvc_ckpt_dir / "checkpoint.pt")
shutil.copy(cfg_path, pvc_ckpt_dir / "config.yaml")
print(f"Checkpoint mirrored to {pvc_ckpt_dir}")
