"""Evaluate a list of checkpoints on the val split, print val/l2 for each.

Run:
  python eval_ckpts.py
"""
import os, time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn, load_data
from model import build_from_checkpoint


CKPT_DIR = Path("/mnt/new-pvc/kagent/apr16/edward/checkpoints")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

CANDIDATES = [
    "model-w8rxftm4",
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, _, stats = load_data(str(SPLITS_DIR))

val_ds = GRAMDataset(SPLITS_DIR / "val")
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
print(f"val: {len(val_ds)} samples")

results = []
for name in CANDIDATES:
    path = CKPT_DIR / name / "checkpoint.pt"
    if not path.exists():
        print(f"{name}: MISSING")
        continue
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_from_checkpoint(state, stats, device)
    model.eval()
    total_l2 = 0.0
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in val_loader:
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            pred = model(v_in, pos, t, idcs)
            l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
            total_l2 += l2_err.sum().item()
            n += v_in.shape[0]
    val_l2 = total_l2 / n
    dt = time.time() - t0
    print(f"{name}: val/l2={val_l2:.4f}  ({dt:.0f}s)")
    results.append((name, val_l2))
    del model

results.sort(key=lambda x: x[1])
print("\nRanked:")
for name, l2 in results:
    print(f"  {l2:.4f}  {name}")
