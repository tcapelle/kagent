"""Generate predictions averaging across multiple model checkpoints.

For pressure-focused leaderboards, an ensemble of diverse pressure predictors
often outperforms any single model. Iter4/iter5 ckpts come from the same
architecture, so we just average their normalized outputs.

Run:
  python predict_ensemble.py --checkpoints /pvc/.../ckptA.pt /pvc/.../ckptB.pt --agent fern
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
    checkpoints: list[str]
    weights: list[float] | None = None  # optional per-checkpoint weights
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 2


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

if cfg.weights is None:
    weights = [1.0 / len(cfg.checkpoints)] * len(cfg.checkpoints)
else:
    assert len(cfg.weights) == len(cfg.checkpoints)
    s = sum(cfg.weights)
    weights = [w / s for w in cfg.weights]
print(f"Weights: {weights}")
weights_t = torch.tensor(weights, device=device).view(-1, 1, 1, 1)

# Load each model
models = []
for ckpt in cfg.checkpoints:
    ckpt_path = Path(ckpt)
    config_path = ckpt_path.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            mc = yaml.safe_load(f)
    else:
        mc = dict(space_dim=2, fun_dim=X_DIM - 2, out_dim=3, n_hidden=256, n_layers=8,
                  n_head=8, slice_num=96, mlp_ratio=2, dropout=0.0)
    mc["use_checkpoint"] = False
    m = Transolver(**mc).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    models.append(m)
    print(f"Loaded {ckpt}")
print(f"Ensemble of {len(models)} models")

with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

agent_name = cfg.agent or "unknown"
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

amp_dtype = torch.bfloat16

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

            preds_norm = []
            for m in models:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pn = m({"x": x_norm})["preds"]
                preds_norm.append(pn.float())
            avg = (torch.stack(preds_norm, dim=0) * weights_t).sum(dim=0)
            pred = avg * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
