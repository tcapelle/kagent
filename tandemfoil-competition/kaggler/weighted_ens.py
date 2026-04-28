"""Optimize ensemble weights on val to minimize surf_p MAE.

Given a set of K candidate ckpts, we solve a constrained NNLS / convex weighted-mean
problem to find weights summing to 1 that minimize MAE on surface pressure.

Usage:
  python weighted_ens.py ckpt1 ckpt2 ... ckptK
"""

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

ckpt_dirs = [Path(p) for p in sys.argv[1:]]

print(f"Caching {len(ckpt_dirs)} candidates over val splits...", flush=True)
loaders = {split: DataLoader(SplitDataset(SPLITS_DIR / split), batch_size=4, shuffle=False,
                              collate_fn=pad_collate, num_workers=2) for split in VAL_SPLIT_NAMES}

# Pre-collect surface masks and truths; surf-flat per split
surf_lists = {split: [] for split in VAL_SPLIT_NAMES}
truth_p_lists = {split: [] for split in VAL_SPLIT_NAMES}
for split in VAL_SPLIT_NAMES:
    for x, y, is_surface, mask in loaders[split]:
        for j in range(x.shape[0]):
            n = mask[j].sum().item()
            surf_lists[split].append(is_surface[j, :n].clone())
            truth_p_lists[split].append(y[j, :n, 2].clone())
truths_flat = {}
for split in VAL_SPLIT_NAMES:
    surf_cat = torch.cat(surf_lists[split], dim=0).to(device)
    p_cat = torch.cat(truth_p_lists[split], dim=0).to(device)
    truths_flat[split] = p_cat[surf_cat]

# preds_flat[split] = [K, Nsurf] tensor in physical p
preds_per_split = {split: [] for split in VAL_SPLIT_NAMES}
names = []
for d in ckpt_dirs:
    ckpt = d / "checkpoint.pt"
    cfg_yaml = d / "config.yaml"
    with open(cfg_yaml) as f:
        mc = yaml.safe_load(f)
    m = Transolver(**mc).to(device).eval()
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    name = d.name
    names.append(name)
    with torch.no_grad():
        for split in VAL_SPLIT_NAMES:
            chunks = []
            for x, y, is_surface, mask in loaders[split]:
                x = x.to(device)
                x_norm = (x - x_mean) / x_std
                p_norm = m({"x": x_norm})["preds"]
                p_p = p_norm[..., 2] * y_std[2] + y_mean[2]
                for j in range(x.shape[0]):
                    n = mask[j].sum().item()
                    chunks.append(p_p[j, :n])
            cat = torch.cat(chunks, dim=0)
            surf_cat = torch.cat(surf_lists[split], dim=0).to(device)
            preds_per_split[split].append(cat[surf_cat])
    del m
    torch.cuda.empty_cache()
    print(f"  {name}: cached", flush=True)

# Stack: split → [K, N] tensor
preds_stack = {split: torch.stack(preds_per_split[split], dim=0) for split in VAL_SPLIT_NAMES}

K = len(names)
print(f"\nOptimizing weights on val (uniform avg as init, K={K})...", flush=True)


def avg_mae(weights):
    """Returns avg surf_p MAE across the 4 splits using softmax-normalized weights."""
    w = torch.softmax(weights, dim=0)
    total = 0.0
    for split in VAL_SPLIT_NAMES:
        avg = (w[:, None] * preds_stack[split]).sum(dim=0)
        mae = (avg - truths_flat[split]).abs().mean()
        total = total + mae
    return total / len(VAL_SPLIT_NAMES)


# Init with log(uniform) i.e. zeros → softmax(0) = 1/K
w = torch.zeros(K, device=device, requires_grad=True)
opt = torch.optim.Adam([w], lr=0.05)

best = float("inf")
best_w = None
for it in range(2000):
    opt.zero_grad()
    loss = avg_mae(w)
    loss.backward()
    opt.step()
    if loss.item() < best:
        best = loss.item()
        best_w = torch.softmax(w, dim=0).detach().cpu().numpy().copy()
    if it % 100 == 0:
        print(f"  iter {it}: avg_mae={loss.item():.4f}", flush=True)

print(f"\nBest weighted avg_mae = {best:.4f}", flush=True)
print("Weights (sum=1.0):", flush=True)
for n, wv in zip(names, best_w):
    print(f"  {n}: {wv:.4f}", flush=True)

# Compare to uniform
uniform = torch.softmax(torch.zeros(K, device=device), dim=0)
total = 0.0
for split in VAL_SPLIT_NAMES:
    avg = (uniform[:, None] * preds_stack[split]).sum(dim=0)
    total += (avg - truths_flat[split]).abs().mean().item()
print(f"\nUniform avg_mae = {total / len(VAL_SPLIT_NAMES):.4f}", flush=True)
print(f"Improvement: {total / len(VAL_SPLIT_NAMES) - best:.4f}", flush=True)
