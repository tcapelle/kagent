"""Generate predictions on hidden test samples.

Supports single-checkpoint prediction and optional model ensembling across
multiple checkpoints (passed as `--checkpoints a.pt,b.pt,c.pt`). Ensembling
averages each model's prediction (with y-reflection TTA) over all checkpoints.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
  python predict.py --checkpoints a.pt,b.pt,c.pt --agent <your-name>
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

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

TEST_SPLITS = ["val"]


@dataclass
class Config:
    """Generate test predictions from one or more trained checkpoints."""
    checkpoint: str | None = None            # single-checkpoint mode
    checkpoints: str | None = None           # comma-separated list for ensembling
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

from train import BaselineMLP, predict_tta

ckpt_paths: list[str] = []
if cfg.checkpoints:
    ckpt_paths = [p.strip() for p in cfg.checkpoints.split(",") if p.strip()]
elif cfg.checkpoint:
    ckpt_paths = [cfg.checkpoint]
else:
    raise SystemExit("Pass --checkpoint or --checkpoints")

models: list[BaselineMLP] = []
for p in ckpt_paths:
    m = BaselineMLP(hidden=384, n_blocks=8).to(device)
    state = torch.load(p, map_location=device, weights_only=True)
    # strict=False lets us load older checkpoints (no SDF branch) into the
    # current architecture; their sdf_embed stays zero-init -> no-op branch.
    missing, unexpected = m.load_state_dict(state, strict=False)
    m.eval()
    print(f"Loaded {p} (missing={len(missing)}, unexpected={len(unexpected)})")
    models.append(m)

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
    print(f"{split}: {len(ds)} samples, {len(models)} models")

    predictions = []
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in tqdm(loader, desc=split, leave=False):
            v_in = v_in.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            preds = []
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for m in models:
                    preds.append(predict_tta(m, v_in, pos, t, idcs).float())
            pred = torch.stack(preds, dim=0).mean(dim=0)  # [B, 5, N, 3]
            for j in range(pred.shape[0]):
                predictions.append(pred[j].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
