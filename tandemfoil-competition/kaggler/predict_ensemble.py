"""Snapshot-ensemble predict: average predictions from many checkpoints in a dir.

Usage:
  python predict_ensemble.py --ckpt_dir checkpoints/snaps --config checkpoints/config.yaml --agent edward

Saves per-split prediction tensors to /mnt/new-pvc/predictions/$RESEARCH_TAG/<agent>/<HEAD>/.
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
    ckpt_dir: str                          # dir containing snapshot .pt files
    config: str = "checkpoints/config.yaml"
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4
    extra_ckpt: str | None = None          # also include this single checkpoint (e.g. checkpoints/best.pt)


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

with open(cfg.config) as f:
    model_config = yaml.safe_load(f)

ckpt_paths = sorted(Path(cfg.ckpt_dir).glob("*.pt"))
if cfg.extra_ckpt:
    ckpt_paths.append(Path(cfg.extra_ckpt))
assert ckpt_paths, f"No checkpoints found in {cfg.ckpt_dir}"
print(f"Ensembling {len(ckpt_paths)} checkpoints:")
for p in ckpt_paths:
    print(f"  {p}")

# Stats
with open(splits_dir / "stats.json") as f:
    stats = json.load(f)
x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats["y_std"], dtype=torch.float32, device=device)

# Output dir keyed by HEAD commit
agent = cfg.agent or "unknown"
head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out_dir = PREDICTIONS_DIR / agent / head
out_dir.mkdir(parents=True, exist_ok=True)
print(f"Output: {out_dir}")

# We need one Transolver instance to load each checkpoint into; per-split we
# accumulate the sum of predictions across all checkpoints and divide at the end.
model = Transolver(**model_config).to(device)

for split in TEST_SPLITS:
    test_files = sorted((splits_dir / split).glob("*.pt"))
    n = len(test_files)
    print(f"{split}: {n} samples × {len(ckpt_paths)} ckpts")

    # Pre-load all sample x tensors (lives on CPU; small enough).
    samples = [torch.load(f, weights_only=True)["x"] for f in test_files]

    accum: list[torch.Tensor | None] = [None] * n
    for ckpt_path in ckpt_paths:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()

        with torch.no_grad():
            for i in tqdm(range(0, n, cfg.batch_size), desc=ckpt_path.name, leave=False):
                xs = samples[i:i + cfg.batch_size]
                max_n = max(x.shape[0] for x in xs)
                B = len(xs)
                x_pad = torch.zeros(B, max_n, X_DIM, device=device)
                for j, x in enumerate(xs):
                    x_pad[j, :x.shape[0]] = x.to(device)
                pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
                pred = pred_norm * y_std + y_mean
                for j, x in enumerate(xs):
                    contrib = pred[j, :x.shape[0]].cpu()
                    if accum[i + j] is None:
                        accum[i + j] = contrib
                    else:
                        accum[i + j] = accum[i + j] + contrib

    avg = [a / len(ckpt_paths) for a in accum]
    out = out_dir / f"{split}.pt"
    torch.save(avg, out)
    print(f"  → {out}")

print(f"\nDone. Saved to {out_dir}")
