"""Generate predictions on hidden test samples.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --config models/model-<id>/config.pt --agent <your-name>
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn
from train import VelocityPredictor

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

TEST_SPLITS = ["val"]


@dataclass
class Config:
    checkpoint: str
    config: str = ""
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load model config
if cfg.config and Path(cfg.config).exists():
    model_cfg = torch.load(cfg.config, map_location="cpu", weights_only=True)
else:
    model_cfg = {"hidden": 256, "n_blocks": 8}

model = VelocityPredictor(
    hidden=model_cfg["hidden"], n_blocks=model_cfg["n_blocks"],
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

            with torch.amp.autocast("cuda"):
                pred = model(v_in, pos, t, idcs)
            pred = pred.float()

            for j in range(pred.shape[0]):
                predictions.append(pred[j].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
