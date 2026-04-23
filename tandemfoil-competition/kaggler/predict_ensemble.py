"""Generate ensemble predictions by averaging output of multiple checkpoints.

Each checkpoint must have a matching config.yaml alongside it that encodes the
model_config dict used to instantiate Transolver.

Run:
  python predict_ensemble.py --agent thorfinn \
    --checkpoints /path/a/checkpoint.pt /path/b/checkpoint.pt ...
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
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
    checkpoints: list[str] = field(default_factory=list)  # list of checkpoint.pt paths
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 2


cfg = sp.parse(Config)
assert len(cfg.checkpoints) >= 1, "provide at least one --checkpoints path"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

models = []
for ckpt in cfg.checkpoints:
    p = Path(ckpt)
    with open(p.parent / "config.yaml") as f:
        mcfg = yaml.safe_load(f)
    m = Transolver(**mcfg).to(device)
    m.load_state_dict(torch.load(p, map_location=device, weights_only=True))
    m.eval()
    models.append(m)
    print(f"Loaded {ckpt}")

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

            x_norm = (x_pad - x_mean) / x_std
            preds_sum = None
            for m in models:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    pred_norm = m({"x": x_norm})["preds"].float()
                if preds_sum is None:
                    preds_sum = pred_norm
                else:
                    preds_sum = preds_sum + pred_norm
            pred_norm = preds_sum / len(models)
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
