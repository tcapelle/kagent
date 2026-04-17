"""Ensemble prediction: average forward passes from multiple checkpoints, each with y-flip TTA.

Run: python ensemble_predict.py --checkpoints a.pt b.pt --agent nezuko
"""
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)
    grid_sizes: list[int] = field(default_factory=list)
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1


cfg = sp.parse(Config)
assert len(cfg.checkpoints) >= 2, "need at least 2 checkpoints"
if cfg.grid_sizes:
    assert len(cfg.grid_sizes) == len(cfg.checkpoints), "grid_sizes must match checkpoints"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

from train import VoxelUNet, MODEL_CFG
with open(Path(cfg.splits_dir) / "stats.json") as f:
    _stats_raw = json.load(f)
_vel_mean = torch.tensor(_stats_raw["vel_mean"], dtype=torch.float32)
_vel_std = torch.tensor(_stats_raw["vel_std"], dtype=torch.float32)

models = []
for i, ckpt_path in enumerate(cfg.checkpoints):
    model_cfg = dict(MODEL_CFG)
    if cfg.grid_sizes:
        model_cfg["grid_size"] = cfg.grid_sizes[i]
    m = VoxelUNet(**model_cfg, vel_mean=_vel_mean, vel_std=_vel_std).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    m.eval()
    models.append(m)
    print(f"Loaded: {ckpt_path} (grid={model_cfg['grid_size']})")

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

for split in ["val"]:
    ds = GRAMDataset(splits_dir / split)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{split}: {len(ds)} samples")

    predictions = []
    l2_errs = []
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in tqdm(loader, desc=split, leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out_gpu = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            v_f = v_in.clone(); v_f[..., 1].neg_()
            pos_f = pos.clone(); pos_f[..., 1].neg_()

            pred_sum = None
            for m in models:
                p1 = m(v_in, pos, t, idcs)
                p2 = m(v_f, pos_f, t, idcs)
                p2 = p2.clone(); p2[..., 1].neg_()
                p = 0.5 * (p1 + p2)
                pred_sum = p if pred_sum is None else pred_sum + p
            pred = pred_sum / len(models)

            l2 = (pred - v_out_gpu).norm(dim=3).mean(dim=(1, 2))
            l2_errs.append(l2.cpu())

            for j in range(pred.shape[0]):
                predictions.append(pred[j].cpu())

    l2_tensor = torch.cat(l2_errs)
    print(f"  {split} l2_error (with y-flip TTA + {len(models)}-model avg): {l2_tensor.mean().item():.4f}")

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
