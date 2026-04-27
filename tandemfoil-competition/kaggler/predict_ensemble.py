"""Generate ensemble predictions on the hidden test splits.

Loads multiple Transolver checkpoints, averages their (denormalized)
predictions per node, and saves to PVC under the current commit.

Run:
  python predict_ensemble.py --checkpoints A.pt,B.pt,C.pt --agent askeladd
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
    checkpoints: str  # comma-separated paths
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4
    bf16: bool = False  # fp32 inference is ~0.1pt better on val/avg_surf_p


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)


def _load_model(ckpt_path: str) -> Transolver:
    p = Path(ckpt_path)
    mc_path = p.parent / "config.yaml"
    if mc_path.exists():
        with open(mc_path) as f:
            mc = yaml.safe_load(f)
    else:
        mc = dict(
            space_dim=2, fun_dim=X_DIM - 2, out_dim=3,
            n_hidden=192, n_layers=6, n_head=6, slice_num=128, mlp_ratio=2,
            output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1],
        )
    m = Transolver(**mc).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


ckpts = [c.strip() for c in cfg.checkpoints.split(",") if c.strip()]
print(f"Loading {len(ckpts)} checkpoints for ensemble:")
models = []
for c in ckpts:
    print(f"  - {c}")
    models.append(_load_model(c))

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
print(f"Writing to {output_dir}")

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

            # Average predictions in physical units across models.
            pred_sum = torch.zeros(B, max_n, 3, device=device)
            for m in models:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                    pred_norm = m({"x": x_norm})["preds"]
                pred_sum += pred_norm.float() * y_std + y_mean
            pred = pred_sum / len(models)

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
