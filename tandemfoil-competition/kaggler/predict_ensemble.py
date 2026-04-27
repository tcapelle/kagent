"""Ensemble prediction: average predictions across multiple checkpoints."""

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
    checkpoints: list[str] = field(default_factory=list)  # paths to checkpoint.pt
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 2
    bf16: bool = True


cfg = sp.parse(Config)
device = torch.device("cuda")
splits_dir = Path(cfg.splits_dir)

if not cfg.checkpoints:
    raise SystemExit("Pass --checkpoints path1 path2 ...")

# Load all models. They must share the same architecture (same config.yaml).
models = []
for c in cfg.checkpoints:
    cp = Path(c)
    cfg_yaml = yaml.safe_load(open(cp.parent / "config.yaml"))
    m = Transolver(**cfg_yaml).to(device).eval()
    m.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
    models.append(m)
    print(f"Loaded {cp}")

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

autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if cfg.bf16 else torch.amp.autocast("cuda", enabled=False)

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

            # Average predictions in normalized space, then denormalize once.
            pred_acc = torch.zeros(B, max_n, 3, device=device)
            for m in models:
                with autocast_ctx:
                    p = m({"x": x_norm})["preds"]
                pred_acc += p.float()
            pred_norm = pred_acc / len(models)
            pred = pred_norm * y_std + y_mean

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nEnsemble of {len(models)} checkpoints saved to {output_dir}")
