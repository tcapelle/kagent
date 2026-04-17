"""Quick val-L2 evaluator for one-or-more checkpoints (optional ensemble).

Mirrors predict.py's forward path (predict_tta, bf16 autocast) but ALSO
compares against the val ground truth — useful for answering "does the
ensemble beat the best single checkpoint?" without touching the real
leaderboard submission path.

Run:
  python eval.py --checkpoints a.pt,b.pt,c.pt
"""

import os
from dataclasses import dataclass

import simple_parsing as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GRAMDataset, collate_fn
from train import BaselineMLP, predict_tta

SPLITS_DIR = os.environ.get("GRAM_SPLITS", "/mnt/new-pvc/datasets/gram/splits")


@dataclass
class Config:
    checkpoints: str = ""  # comma-separated
    batch_size: int = 1


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

paths = [p.strip() for p in cfg.checkpoints.split(",") if p.strip()]
assert paths, "--checkpoints required"

models = []
for p in paths:
    m = BaselineMLP(hidden=384, n_blocks=8).to(device)
    state = torch.load(p, map_location=device, weights_only=True)
    missing, unexpected = m.load_state_dict(state, strict=False)
    m.eval()
    print(f"Loaded {p}  (missing={len(missing)}, unexpected={len(unexpected)})")
    models.append(m)

ds = GRAMDataset(f"{SPLITS_DIR}/val")
loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

total_l2 = 0.0
total_mae = torch.zeros(3, device=device, dtype=torch.float64)
n = 0

with torch.no_grad():
    for v_in, v_out, pos, t, idcs in tqdm(loader, leave=False):
        v_in = v_in.to(device, non_blocking=True)
        v_out = v_out.to(device, non_blocking=True)
        pos = pos.to(device, non_blocking=True)
        t = t.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = [predict_tta(m, v_in, pos, t, idcs).float() for m in models]
        pred = torch.stack(preds, 0).mean(0)
        l2 = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
        total_l2 += l2.sum().item()
        total_mae += (pred - v_out).abs().mean(dim=(1, 2)).double().sum(0)
        n += v_in.shape[0]

mean_l2 = total_l2 / n
mean_mae = total_mae / n
print(f"\nval/l2_error = {mean_l2:.4f}  (n={n} samples, {len(models)} models)")
print(f"  mae Ux={mean_mae[0]:.4f}  Uy={mean_mae[1]:.4f}  Uz={mean_mae[2]:.4f}")
