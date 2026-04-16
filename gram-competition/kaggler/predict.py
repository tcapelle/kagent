"""Generate predictions on hidden test samples.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
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
from train import BaselineMLP

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

TEST_SPLITS = ["val"]


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1
    hidden: int = 256
    n_blocks: int = 8
    grid_size: int = 32
    n_fourier: int = 8


def main():
    cfg = sp.parse(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(cfg.splits_dir)

    with open(splits_dir / "stats.json") as f:
        _s = json.load(f)
    vel_mean = torch.tensor(_s["vel_mean"], dtype=torch.float32)
    vel_std = torch.tensor(_s["vel_std"], dtype=torch.float32)

    model = BaselineMLP(
        hidden=cfg.hidden,
        n_blocks=cfg.n_blocks,
        grid_size=cfg.grid_size,
        n_fourier=cfg.n_fourier,
        vel_mean=vel_mean,
        vel_std=vel_std,
    ).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded model from {cfg.checkpoint}")

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
        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in tqdm(loader, desc=split, leave=False):
                v_in = v_in.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs)
                for j in range(pred.shape[0]):
                    predictions.append(pred[j].cpu())

        output_path = output_dir / f"{split}.pt"
        torch.save(predictions, output_path)
        print(f"  -> {output_path} ({len(predictions)} samples)")

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
