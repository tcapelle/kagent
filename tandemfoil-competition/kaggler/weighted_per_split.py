"""Optimize per-split ensemble weights on val to minimize per-split surf_p MAE.

Each test split gets its own softmax weight vector across the K candidates.
At inference time we apply the corresponding weights based on which split a sample comes from.

Usage:
  python weighted_per_split.py ckpt1 ckpt2 ... ckptK
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

preds_stack = {split: torch.stack(preds_per_split[split], dim=0) for split in VAL_SPLIT_NAMES}

K = len(names)
print(f"\nOptimizing per-split weights on val (K={K} candidates × 4 splits)...", flush=True)


def mae():
    total = 0.0
    for i, split in enumerate(VAL_SPLIT_NAMES):
        w = torch.softmax(W[i], dim=0)
        avg = (w[:, None] * preds_stack[split]).sum(dim=0)
        total = total + (avg - truths_flat[split]).abs().mean()
    return total / len(VAL_SPLIT_NAMES)


W = torch.zeros(len(VAL_SPLIT_NAMES), K, device=device, requires_grad=True)
opt = torch.optim.Adam([W], lr=0.05)

best = float("inf")
best_W = None
for it in range(3000):
    opt.zero_grad()
    loss = mae()
    loss.backward()
    opt.step()
    if loss.item() < best:
        best = loss.item()
        best_W = torch.softmax(W, dim=1).detach().cpu().numpy().copy()
    if it % 200 == 0:
        print(f"  iter {it}: avg_mae={loss.item():.4f}", flush=True)

print(f"\nBest per-split avg_mae = {best:.4f}", flush=True)
print("Per-split weights:", flush=True)
for i, split in enumerate(VAL_SPLIT_NAMES):
    print(f"\n[{split}]")
    for nm, wv in zip(names, best_W[i]):
        print(f"  {nm}: {wv:.4f}")

# Compare against uniform
uniform = torch.full((K,), 1.0 / K, device=device)
total = 0.0
for split in VAL_SPLIT_NAMES:
    avg = (uniform[:, None] * preds_stack[split]).sum(dim=0)
    total += (avg - truths_flat[split]).abs().mean().item()
print(f"\nUniform (no per-split) avg_mae = {total / len(VAL_SPLIT_NAMES):.4f}")
print(f"Improvement vs uniform: {total / len(VAL_SPLIT_NAMES) - best:.4f}")
