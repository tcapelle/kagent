"""Generate ensemble predictions on the hidden test splits.

Averages predictions (in physical units) across multiple checkpoints.

Run:
  python predict_ensemble.py \
      --checkpoints models/model-A/checkpoint.pt models/model-B/checkpoint.pt \
      --weights 1.0 1.0 \
      --agent edward
"""

import json
import os
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

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

if not cfg.checkpoints:
    raise ValueError("Provide at least one checkpoint via --checkpoints")
weights = cfg.weights or [1.0] * len(cfg.checkpoints)
if len(weights) != len(cfg.checkpoints):
    raise ValueError("len(weights) must match len(checkpoints)")
total_w = sum(weights)
weights = [w / total_w for w in weights]
print(f"Ensemble: {len(cfg.checkpoints)} models, weights={weights}")

models = []
for ckpt in cfg.checkpoints:
    ckpt_path = Path(ckpt)
    config_path = ckpt_path.parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
    print(f"Loaded: {ckpt}")

# Stats
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

            max_n = max(x.shape[0] for x in xs)
            B = len(xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            for j, x in enumerate(xs):
                x_pad[j, :x.shape[0]] = x.to(device)

            x_norm = (x_pad - x_mean) / x_std
            ensembled = None
            for m, w in zip(models, weights):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = m({"x": x_norm})["preds"].float()
                p_phys = p * y_std + y_mean
                ensembled = p_phys * w if ensembled is None else ensembled + p_phys * w

            for j, x in enumerate(xs):
                predictions.append(ensembled[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll ensemble predictions saved to {output_dir}")
