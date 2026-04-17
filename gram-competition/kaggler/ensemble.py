"""Ensemble predictions across multiple checkpoints.

Loads k checkpoints, runs val inference on each, averages predictions,
reports val/l2_error of the ensemble, and saves averaged predictions
to the standard predictions directory.

Run:
  python ensemble.py --checkpoints ckpt1.pt ckpt2.pt [...] --agent <name>
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GRAMDataset, load_data
from train import VoxelResidualModel, compute_sdf, SDFDataset, collate_sdf

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1


cfg = sp.parse(Config)
assert len(cfg.checkpoints) >= 2, "Need at least 2 checkpoints to ensemble"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

_, val_splits, stats = load_data(cfg.splits_dir)

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

for split_name, base in val_splits.items():
    print(f"{split_name}: {len(base)} samples")

    print("  Precomputing SDF...")
    sdfs = [compute_sdf(base[i][2], base[i][4], device)
            for i in tqdm(range(len(base)), desc=f"{split_name} SDF")]
    ds = SDFDataset(base, sdfs)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sdf)

    # Accumulate predictions and ground truth across all checkpoints
    ensemble_preds: list[torch.Tensor] | None = None
    gt_all: list[torch.Tensor] = []

    for ck_idx, ckpt_path in enumerate(cfg.checkpoints):
        print(f"  [{ck_idx+1}/{len(cfg.checkpoints)}] Loading {ckpt_path}")
        model = VoxelResidualModel(
            vel_mean=stats["vel_mean"], vel_std=stats["vel_std"],
            hidden=256, voxel_res=64, voxel_mid=64,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()

        preds_this: list[torch.Tensor] = []
        with torch.no_grad():
            for v_in, v_out, pos, t, idcs, sdf in tqdm(loader, desc=f"ckpt{ck_idx+1}", leave=False):
                v_in = v_in.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                sdf = sdf.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs, sdf)  # [B, 5, N, 3]
                for j in range(pred.shape[0]):
                    preds_this.append(pred[j].cpu())
                if ck_idx == 0:
                    for j in range(v_out.shape[0]):
                        gt_all.append(v_out[j])

        if ensemble_preds is None:
            ensemble_preds = preds_this
        else:
            ensemble_preds = [a + b for a, b in zip(ensemble_preds, preds_this)]

    # Average
    ensemble_preds = [p / len(cfg.checkpoints) for p in ensemble_preds]

    # Compute val/l2
    total_l2 = 0.0
    for p, g in zip(ensemble_preds, gt_all):
        total_l2 += (p - g).norm(dim=2).mean().item()
    mean_l2 = total_l2 / len(gt_all)
    print(f"  {split_name} ENSEMBLE val/l2 = {mean_l2:.4f}  ({len(cfg.checkpoints)} models)")

    output_path = output_dir / f"{split_name}.pt"
    torch.save(ensemble_preds, output_path)
    print(f"  -> {output_path} ({len(ensemble_preds)} samples)")

print(f"\nEnsemble predictions saved to {output_dir}")
