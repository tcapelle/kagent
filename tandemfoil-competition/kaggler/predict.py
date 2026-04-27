"""Generate predictions on the hidden test splits.

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

from data import X_DIM
from train import Transolver, MODEL_CONFIG

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
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 2


def main():
    cfg = sp.parse(Config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(cfg.splits_dir)

    model = Transolver(**MODEL_CONFIG).to(device)
    sd = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    print(f"Loaded model from {cfg.checkpoint}")

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
                mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)
                for j, x in enumerate(xs):
                    n = x.shape[0]
                    x_pad[j, :n] = x.to(device)
                    mask[j, :n] = True

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred_norm = model({"x": (x_pad - x_mean) / x_std, "mask": mask})["preds"]
                pred_norm = pred_norm.float()
                pred = pred_norm * y_std + y_mean

                for j, x in enumerate(xs):
                    predictions.append(pred[j, :x.shape[0]].cpu())

        output_path = output_dir / f"{split}.pt"
        torch.save(predictions, output_path)
        print(f"  -> {output_path} ({len(predictions)} samples)")

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
