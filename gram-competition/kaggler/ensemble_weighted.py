"""Weighted ensemble: compute per-seed val/l2 on-the-fly and test several
weighting schemes. Reports val/l2 for each scheme, saves the best one.

Schemes tried:
  uniform: baseline (same as ensemble.py)
  inv_loss: w_i ~ 1 / l2_i
  inv_loss_sq: w_i ~ 1 / l2_i^2
  softmax_neg_loss: w_i ~ exp(-l2_i / T), T=0.02
  softmax_neg_loss_cold: w_i ~ exp(-l2_i / T), T=0.005 (sharper)

Run:
  python ensemble_weighted.py --checkpoints ckpt1.pt ckpt2.pt [...] --agent <name>
"""

import math
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
assert len(cfg.checkpoints) >= 2, "Need at least 2 checkpoints"
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


def compute_l2(preds, gts):
    total = 0.0
    for p, g in zip(preds, gts):
        total += (p - g).norm(dim=2).mean().item()
    return total / len(gts)


for split_name, base in val_splits.items():
    print(f"\n{split_name}: {len(base)} samples")
    print("  Precomputing SDF...")
    sdfs = [compute_sdf(base[i][2], base[i][4], device)
            for i in tqdm(range(len(base)), desc=f"{split_name} SDF")]
    ds = SDFDataset(base, sdfs)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sdf)

    all_preds: list[list[torch.Tensor]] = []   # per-ckpt list of per-sample preds
    gt_all: list[torch.Tensor] = []
    per_seed_l2: list[float] = []

    for ck_idx, ckpt_path in enumerate(cfg.checkpoints):
        print(f"  [{ck_idx+1}/{len(cfg.checkpoints)}] {ckpt_path}")
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

                pred = model(v_in, pos, t, idcs, sdf)
                for j in range(pred.shape[0]):
                    preds_this.append(pred[j].cpu())
                if ck_idx == 0:
                    for j in range(v_out.shape[0]):
                        gt_all.append(v_out[j])

        all_preds.append(preds_this)
        seed_l2 = compute_l2(preds_this, gt_all)
        per_seed_l2.append(seed_l2)
        print(f"    solo val/l2 = {seed_l2:.4f}")

    l2 = torch.tensor(per_seed_l2)
    print(f"\n  Per-seed solo val/l2: min={l2.min():.4f}, max={l2.max():.4f}, mean={l2.mean():.4f}")

    # Weighting schemes: softmax(-l2/T) normalized
    schemes = {
        "uniform": torch.ones(len(cfg.checkpoints)) / len(cfg.checkpoints),
        "inv_loss": (1.0 / l2) / (1.0 / l2).sum(),
        "inv_loss_sq": (1.0 / l2 ** 2) / (1.0 / l2 ** 2).sum(),
        "softmax_T0.02": torch.softmax(-l2 / 0.02, dim=0),
        "softmax_T0.005": torch.softmax(-l2 / 0.005, dim=0),
        "softmax_T0.001": torch.softmax(-l2 / 0.001, dim=0),
    }

    best_name, best_loss, best_preds = None, float("inf"), None
    for name, w in schemes.items():
        # Weighted average predictions
        n_samples = len(all_preds[0])
        ensemble_preds = []
        for si in range(n_samples):
            acc = torch.zeros_like(all_preds[0][si])
            for ci in range(len(cfg.checkpoints)):
                acc += w[ci].item() * all_preds[ci][si]
            ensemble_preds.append(acc)

        loss = compute_l2(ensemble_preds, gt_all)
        print(f"  {name:20s}: val/l2 = {loss:.4f}   weights[min,max]=[{w.min():.3f}, {w.max():.3f}]")
        if loss < best_loss:
            best_loss = loss
            best_name = name
            best_preds = ensemble_preds

    print(f"\n  BEST: {best_name}  val/l2 = {best_loss:.4f}")
    output_path = output_dir / f"{split_name}.pt"
    torch.save(best_preds, output_path)
    print(f"  -> {output_path} ({len(best_preds)} samples, scheme={best_name})")
