"""Sweep ensemble combinations: load N candidate ckpts, eval all k-combos on val.

Usage:
  python sweep_ens.py k=3 ckpt1 ckpt2 ... ckptM
  Picks best k-combo by val avg surf_p MAE (averaged equally across 4 splits).

Vectorized: per (model, split), flatten surface predictions for surf_p across all
samples into a single 1D tensor. Combos then reduce to a single mean over stacked tensors.
"""

import itertools
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, pad_collate, SplitDataset
from train import Transolver

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")

device = torch.device("cuda")
with open(SPLITS_DIR / "stats.json") as f:
    sd_ = json.load(f)
x_mean = torch.tensor(sd_["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(sd_["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(sd_["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(sd_["y_std"], dtype=torch.float32, device=device)

args = sys.argv[1:]
k_values = []
ckpt_args = []
top_n = 10
for a in args:
    if a.startswith("k="):
        k_values = [int(x) for x in a.split("=")[1].split(",")]
    elif a.startswith("top="):
        top_n = int(a.split("=")[1])
    else:
        ckpt_args.append(a)
if not k_values:
    k_values = [4]
ckpt_dirs = [Path(p) for p in ckpt_args]

# Storage: per (model, split) → flat 1D tensor of surface-only normalized p preds across all samples
# Truth: per split → flat 1D tensor of surface-only physical p across all samples
print(f"Pre-computing predictions for {len(ckpt_dirs)} candidates over val splits...", flush=True)
preds_flat = {}   # name -> {split: tensor[Nsurf_total]} on GPU, normalized space, p channel only
truths_flat = {}  # split -> tensor[Nsurf_total] physical p

loaders = {split: DataLoader(SplitDataset(SPLITS_DIR / split), batch_size=4, shuffle=False,
                              collate_fn=pad_collate, num_workers=2) for split in VAL_SPLIT_NAMES}

# Pre-collect surface masks and truths into flat tensors.
surf_lists = {split: [] for split in VAL_SPLIT_NAMES}  # bool 1D per sample (used to slice preds later)
truth_p_lists = {split: [] for split in VAL_SPLIT_NAMES}
sample_n_lists = {split: [] for split in VAL_SPLIT_NAMES}
for split in VAL_SPLIT_NAMES:
    for x, y, is_surface, mask in loaders[split]:
        for j in range(x.shape[0]):
            n = mask[j].sum().item()
            surf_lists[split].append(is_surface[j, :n].clone())
            truth_p_lists[split].append(y[j, :n, 2].clone())  # physical p
            sample_n_lists[split].append(n)
    surf_cat = torch.cat(surf_lists[split], dim=0).to(device)
    p_truth_cat = torch.cat(truth_p_lists[split], dim=0).to(device)
    truths_flat[split] = p_truth_cat[surf_cat]  # only surface positions

for d in ckpt_dirs:
    ckpt = d / "checkpoint.pt"
    cfg_yaml = d / "config.yaml"
    if not ckpt.exists() or not cfg_yaml.exists():
        continue
    with open(cfg_yaml) as f:
        mc = yaml.safe_load(f)
    try:
        m = Transolver(**mc).to(device).eval()
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    except (TypeError, RuntimeError):
        continue
    name = d.name
    preds_flat[name] = {}
    with torch.no_grad():
        for split in VAL_SPLIT_NAMES:
            chunks = []
            for x, y, is_surface, mask in loaders[split]:
                x = x.to(device)
                x_norm = (x - x_mean) / x_std
                p_norm = m({"x": x_norm})["preds"]  # [B, N, 3] normalized
                # Denormalize p channel right away
                p_p = p_norm[..., 2] * y_std[2] + y_mean[2]  # physical p, [B, N]
                for j in range(x.shape[0]):
                    n = mask[j].sum().item()
                    chunks.append(p_p[j, :n])
            cat = torch.cat(chunks, dim=0)  # [Ntotal]
            surf_cat = torch.cat(surf_lists[split], dim=0).to(device)
            preds_flat[name][split] = cat[surf_cat].clone()  # [Nsurf_total]
    del m
    torch.cuda.empty_cache()
    print(f"  {name}: cached", flush=True)


def score_ensemble(names):
    """Returns (avg_over_splits, per_split_dict). Vectorized."""
    per_split = {}
    total = 0.0
    for split in VAL_SPLIT_NAMES:
        stacked = torch.stack([preds_flat[nm][split] for nm in names], dim=0)  # [K, Nsurf]
        avg = stacked.mean(dim=0)
        mae = (avg - truths_flat[split]).abs().mean().item()
        per_split[split] = mae
        total += mae
    return total / len(VAL_SPLIT_NAMES), per_split


# Singles
singles = []
for nm in preds_flat:
    avg, per = score_ensemble([nm])
    singles.append((nm, avg))
print("\n=== Singles (sorted by val avg surf_p MAE) ===", flush=True)
for nm, s in sorted(singles, key=lambda x: x[1])[:30]:
    print(f"{s:7.3f}  {nm}", flush=True)

# k-combos
top_names = [nm for nm, _ in sorted(singles, key=lambda x: x[1])[:top_n]]
print(f"\nTop-{top_n} candidates: {top_names}", flush=True)
for k in k_values:
    print(f"\n=== {k}-combos from top {top_n} singles ===", flush=True)
    combo_results = []
    for combo in itertools.combinations(top_names, k):
        avg, per = score_ensemble(list(combo))
        combo_results.append((combo, avg, per))
    for combo, avg, per in sorted(combo_results, key=lambda x: x[1])[:25]:
        parts = " ".join(combo)
        print(f"{avg:7.3f}  {parts}", flush=True)
