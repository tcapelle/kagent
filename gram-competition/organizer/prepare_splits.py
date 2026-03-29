"""Prepare GRAM competition splits from raw HuggingFace .npz files.

Downloads from gram-competition/warped-ifw, splits by simulation ID,
saves per-sample .pt files and normalization stats.

Run:
  python prepare_splits.py [--data_dir /path/to/npz/files]
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
from rich.console import Console

console = Console()

SEED = 42
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# remaining 0.10 goes to test

DATA_ROOT = Path("/mnt/new-pvc/datasets/gram")


@dataclass
class Config:
    data_dir: str = str(DATA_ROOT / "raw")  # directory with .npz files
    out_dir: str = str(DATA_ROOT / "splits")


cfg = sp.parse(Config)
data_dir = Path(cfg.data_dir)
out_dir = Path(cfg.out_dir)

# --- Scan and group by simulation ID ---
console.rule("Scanning .npz files")
npz_files = sorted(data_dir.glob("*.npz"))
print(f"Found {len(npz_files)} .npz files")

sim_to_files: dict[str, list[Path]] = {}
for f in npz_files:
    # filename: <sim_id>-<window_idx>.npz e.g. "1021_1-0.npz"
    sim_id = f.stem.rsplit("-", 1)[0]
    sim_to_files.setdefault(sim_id, []).append(f)

sim_ids = sorted(sim_to_files.keys())
print(f"Found {len(sim_ids)} unique simulations, {len(npz_files)} total samples")

# --- Split by simulation ID ---
console.rule("Splitting by simulation ID")
random.seed(SEED)
random.shuffle(sim_ids)

n_train = int(len(sim_ids) * TRAIN_FRAC)
n_val = int(len(sim_ids) * VAL_FRAC)
train_sims = sim_ids[:n_train]
val_sims = sim_ids[n_train:n_train + n_val]
test_sims = sim_ids[n_train + n_val:]

print(f"Train: {len(train_sims)} sims, Val: {len(val_sims)} sims, Test: {len(test_sims)} sims")


def save_split(split_name: str, sims: list[str]):
    """Convert .npz files for given simulation IDs to .pt files."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = out_dir / f".{split_name}_gt"
    is_test = split_name == "test"
    if is_test:
        gt_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    for sim_id in sorted(sims):
        for npz_path in sorted(sim_to_files[sim_id]):
            data = np.load(npz_path)

            sample = {
                "velocity_in": torch.from_numpy(data["velocity_in"].astype(np.float32)),   # [5, N, 3]
                "velocity_out": torch.from_numpy(data["velocity_out"].astype(np.float32)), # [5, N, 3]
                "pos": torch.from_numpy(data["pos"].astype(np.float32)),                   # [N, 3]
                "t": torch.from_numpy(data["t"].astype(np.float32)),                       # [10]
                "idcs_airfoil": torch.from_numpy(data["idcs_airfoil"].astype(np.int64)),   # [M]
            }

            torch.save(sample, split_dir / f"{idx:06d}.pt")

            if is_test:
                gt = {"velocity_out": sample["velocity_out"]}
                torch.save(gt, gt_dir / f"{idx:06d}.pt")

            idx += 1

    print(f"  {split_name}: {idx} samples")
    return idx


# --- Save splits ---
console.rule("Saving splits")
n_train_samples = save_split("train", train_sims)
n_val_samples = save_split("val", val_sims)
n_test_samples = save_split("test", test_sims)

# --- Compute normalization stats on training set ---
console.rule("Computing velocity stats (train set)")
train_dir = out_dir / "train"
train_files = sorted(train_dir.glob("*.pt"))

vel_sum = torch.zeros(3, dtype=torch.float64)
vel_sq_sum = torch.zeros(3, dtype=torch.float64)
n_total = 0

for f in train_files:
    s = torch.load(f, weights_only=True)
    # Combine velocity_in and velocity_out for stats
    vel = torch.cat([s["velocity_in"], s["velocity_out"]], dim=0)  # [10, N, 3]
    vel = vel.reshape(-1, 3).double()
    vel_sum += vel.sum(0)
    vel_sq_sum += (vel ** 2).sum(0)
    n_total += vel.shape[0]

vel_mean = vel_sum / n_total
vel_std = ((vel_sq_sum / n_total) - vel_mean ** 2).sqrt()
vel_std = vel_std.clamp(min=1e-6)

stats = {
    "vel_mean": vel_mean.tolist(),
    "vel_std": vel_std.tolist(),
}
with open(out_dir / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# --- Summary ---
console.rule("Summary")
meta = {
    "n_train": n_train_samples,
    "n_val": n_val_samples,
    "n_test": n_test_samples,
    "n_simulations": len(sim_ids),
    "train_sims": train_sims,
    "val_sims": val_sims,
    "test_sims": test_sims,
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"Train: {n_train_samples} samples ({len(train_sims)} sims)")
print(f"Val:   {n_val_samples} samples ({len(val_sims)} sims)")
print(f"Test:  {n_test_samples} samples ({len(test_sims)} sims)")
print(f"Stats: vel_mean={vel_mean.tolist()}, vel_std={vel_std.tolist()}")
print(f"\nSaved to {out_dir}")
