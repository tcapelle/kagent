"""Search optimal per-model weights for an ensemble on the val set.

Runs each model once with y-flip TTA, stacks predictions, then uses L-BFGS
to minimize the mean L2 velocity error over non-negative weights.
"""
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.optimize import minimize

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
    save: bool = False


cfg = sp.parse(Config)
assert len(cfg.checkpoints) >= 2
assert len(cfg.grid_sizes) == len(cfg.checkpoints)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

from train import VoxelUNet, MODEL_CFG
with open(Path(cfg.splits_dir) / "stats.json") as f:
    _stats_raw = json.load(f)
_vel_mean = torch.tensor(_stats_raw["vel_mean"], dtype=torch.float32)
_vel_std = torch.tensor(_stats_raw["vel_std"], dtype=torch.float32)

ds = GRAMDataset(splits_dir / "val")
loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(f"val: {len(ds)} samples")

all_preds = []  # list of [M, n_val, 5, N, 3] eventually; collect per-model
targets = []

for ci, ckpt_path in enumerate(cfg.checkpoints):
    model_cfg = dict(MODEL_CFG)
    model_cfg["grid_size"] = cfg.grid_sizes[ci]
    m = VoxelUNet(**model_cfg, vel_mean=_vel_mean, vel_std=_vel_std).to(device)
    m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    m.eval()
    print(f"Loaded: {ckpt_path} (grid={model_cfg['grid_size']})")

    model_preds = []
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in tqdm(loader, desc=f"m{ci}", leave=False):
            v_in = v_in.to(device, non_blocking=True); pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            v_f = v_in.clone(); v_f[..., 1].neg_()
            pos_f = pos.clone(); pos_f[..., 1].neg_()
            p1 = m(v_in, pos, t, idcs)
            p2 = m(v_f, pos_f, t, idcs)
            p2 = p2.clone(); p2[..., 1].neg_()
            p = 0.5 * (p1 + p2)
            model_preds.append(p.cpu())
            if ci == 0:
                targets.append(v_out)
    all_preds.append(torch.cat(model_preds, dim=0))  # [n_val, 5, N, 3]
    del m
    torch.cuda.empty_cache()

P = torch.stack(all_preds, dim=0).float()  # [M, n_val, 5, N, 3]
Y = torch.cat(targets, dim=0).float()       # [n_val, 5, N, 3]
M = P.shape[0]
print(f"P shape: {P.shape}, Y shape: {Y.shape}")

# Uniform baseline
pred_uniform = P.mean(dim=0)
l2_uniform = (pred_uniform - Y).norm(dim=3).mean(dim=(1, 2))
print(f"\nUniform ensemble val l2: {l2_uniform.mean().item():.4f}")

# Search for optimal weights w (M-dim, sum to 1, >= 0) that minimize mean L2 error
# Use scipy with simplex constraint
def objective(w):
    w = torch.tensor(w, dtype=torch.float32)
    pred = (P * w.view(M, 1, 1, 1, 1)).sum(dim=0)
    l2 = (pred - Y).norm(dim=3).mean(dim=(1, 2))
    return l2.mean().item()

# Initialize at uniform
w0 = np.ones(M) / M
# Constraint: sum(w) = 1
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
bounds = [(0.0, 1.0)] * M
res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"ftol": 1e-8, "maxiter": 200})
print(f"\nOptimized weights: {[f'{w:.4f}' for w in res.x]}")
print(f"Optimized val l2: {res.fun:.4f}")

# Also try unconstrained (allow weights > 1 or negative)
res2 = minimize(objective, w0, method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 500})
print(f"\nUnconstrained weights: {[f'{w:.4f}' for w in res2.x]}")
print(f"Unconstrained val l2: {res2.fun:.4f}")

# Save the best
best_w = torch.tensor(res.x if res.fun <= res2.fun else res2.x, dtype=torch.float32)
best_l2 = min(res.fun, res2.fun)
print(f"\nBest: l2={best_l2:.4f}, weights={best_w.tolist()}")

if cfg.save:
    agent_name = cfg.agent or "unknown"
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    output_dir = PREDICTIONS_DIR / agent_name / commit
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_best = (P * best_w.view(M, 1, 1, 1, 1)).sum(dim=0)
    predictions = [pred_best[j] for j in range(pred_best.shape[0])]
    output_path = output_dir / "val.pt"
    torch.save(predictions, output_path)
    print(f"-> {output_path}")
