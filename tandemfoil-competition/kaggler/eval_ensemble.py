"""Evaluate ensembles of checkpoints on validation splits.

Reports avg surface pressure MAE for each ensemble combination, mirroring
the leaderboard metric.
"""

import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_data, pad_collate
from model import Transolver

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")

# Define candidate checkpoints
CKPTS = {
    "iter4": "models/model-kjb26vxt/checkpoint.pt",  # slice 128, val 73.21
    "iter5b": "models/model-hahrr3i7/checkpoint.pt",  # full mesh, val 50.53
    "iter6": "models/model-rz1akevm/checkpoint.pt",  # chain lr 5e-7, val 49.61
    "iter7": "models/model-q5t9h880/checkpoint.pt",  # chain lr 2e-7, val 49.33
    "iter8": "models/model-viigighk/checkpoint.pt",  # camber+full, val 47.71
    "iter9": "models/model-4cx1uha0/checkpoint.pt",  # camber chain, val 47.25
}

# Try ensembles
ENSEMBLES = [
    ["iter9"],
    ["iter8", "iter9"],
    ["iter7", "iter8", "iter9"],
    ["iter5b", "iter8", "iter9"],
    ["iter5b", "iter6", "iter7", "iter8", "iter9"],
]

train_ds, val_splits, stats, _ = load_data(splits_dir, debug=False)
stats = {k: v.to(device) for k, v in stats.items()}

# Load all candidate models
loaded = {}
for name, path in CKPTS.items():
    cpp = Path(path)
    if not cpp.exists():
        print(f"Skip {name}: {path} missing")
        continue
    with open(cpp.parent / "config.yaml") as f:
        mc = yaml.safe_load(f)
    m = Transolver(**mc).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    loaded[name] = m
    print(f"Loaded {name} (slice_num={mc.get('slice_num')})")

# Cache per-split per-model normalized predictions to avoid re-running models
val_loaders = {
    name: DataLoader(ds, batch_size=4, shuffle=False, collate_fn=pad_collate, num_workers=2)
    for name, ds in val_splits.items()
}

# pred_cache[split_name][ckpt_name] = list of [B, N_max, 3] tensors (one per batch, normalized)
pred_cache: dict = {sp: {} for sp in val_splits}
gt_cache: dict = {sp: {"y": [], "is_surface": [], "mask": []} for sp in val_splits}

for split_name, loader in val_loaders.items():
    print(f"\n--- caching {split_name} ---")
    for name in loaded:
        pred_cache[split_name][name] = []
    with torch.no_grad():
        for x, y, is_surface, mask in tqdm(loader):
            x = x.to(device)
            y_dev = y.to(device)
            is_surf = is_surface.to(device)
            mask_dev = mask.to(device)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            for name, m in loaded.items():
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    p = m({"x": x_norm})["preds"].float()
                pred_cache[split_name][name].append(p.cpu())
            gt_cache[split_name]["y"].append(y_dev.cpu())
            gt_cache[split_name]["is_surface"].append(is_surf.cpu())
            gt_cache[split_name]["mask"].append(mask_dev.cpu())

# Score ensembles
print("\n=== Ensemble surface-p MAE ===")
y_mean = stats["y_mean"].cpu()
y_std = stats["y_std"].cpu()
results = {}
for ens in ENSEMBLES:
    if any(n not in loaded for n in ens):
        continue
    per_split = {}
    for split_name in val_splits:
        n_batches = len(gt_cache[split_name]["y"])
        sum_p = 0.0
        n_p = 0
        for b in range(n_batches):
            preds = [pred_cache[split_name][n][b] for n in ens]
            pred_norm = torch.stack(preds).mean(0)
            pred = pred_norm * y_std + y_mean
            y = gt_cache[split_name]["y"][b]
            is_surf = gt_cache[split_name]["is_surface"][b]
            mask = gt_cache[split_name]["mask"][b]
            surf_mask = mask & is_surf
            err_p = (pred[..., 2] - y[..., 2]).abs() * surf_mask.float()
            sum_p += err_p.sum().item()
            n_p += surf_mask.sum().item()
        per_split[split_name] = sum_p / max(n_p, 1)
    avg = sum(per_split.values()) / len(per_split)
    results["+".join(ens)] = (avg, per_split)
    print(f"\n{'+'.join(ens):40s} avg={avg:.3f}")
    for k, v in per_split.items():
        print(f"  {k}: {v:.3f}")

print("\n=== Ranked ===")
for name, (avg, _) in sorted(results.items(), key=lambda kv: kv[1][0]):
    print(f"  {avg:7.3f}  {name}")
