"""Generate predictions on hidden test samples.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>

For a multi-seed ensemble, pass a comma-separated list of checkpoint paths:
  python predict.py --checkpoint path1.pt,path2.pt,path3.pt --agent <your-name>
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

TEST_SPLITS = ["val"]


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint (or comma-separated ensemble)."""
    checkpoint: str  # path to best model checkpoint, or comma-separated list for ensemble
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

ckpt_paths = [p.strip() for p in cfg.checkpoint.split(",") if p.strip()]
print(f"Ensemble of {len(ckpt_paths)} checkpoint(s)" if len(ckpt_paths) > 1 else f"Single checkpoint")

from model import VoxelFlowNet

def load_model(path):
    m = VoxelFlowNet(
        vel_mean=torch.zeros(3), vel_std=torch.ones(3),
        grid_res=96, grid_ch=24, n_grid_blocks=4,
        point_hidden=384, n_point_blocks=6, point_dropout=0.15,
    ).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    print(f"  loaded {path}")
    return m

models = [load_model(p) for p in ckpt_paths]

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

for split in TEST_SPLITS:
    ds = GRAMDataset(splits_dir / split)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{split}: {len(ds)} samples")

    predictions = []
    total_l2 = 0.0
    n_samples = 0
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs, sdf in tqdm(loader, desc=split, leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            sdf = sdf.to(device, non_blocking=True)

            # Average predictions across ensemble
            pred_sum = None
            for m in models:
                p = m(v_in, pos, t, idcs, sdf)  # [B, 5, N, 3]
                pred_sum = p if pred_sum is None else pred_sum + p
            pred = pred_sum / len(models)

            l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
            total_l2 += l2_err.sum().item()
            n_samples += pred.shape[0]

            for j in range(pred.shape[0]):
                predictions.append(pred[j].cpu())

    mean_l2 = total_l2 / max(n_samples, 1)
    print(f"  {split}/l2_error = {mean_l2:.4f}")
    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
