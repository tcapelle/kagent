"""Ensemble v6-family checkpoints: average predictions, save to PVC.

Run:
  python ensemble.py --agent edward
"""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GRAMDataset, collate_fn, load_data
from model import build_from_checkpoint


RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "apr16")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")
CKPT_DIR = Path("/mnt/new-pvc/kagent/apr16/edward/checkpoints")

# Ensemble members (val/l2 in isolation):
#   jnseenin 1.1681 (v6 canonical)
#   9vaz4wrn 1.1812 (v11 EMA)
#   w8rxftm4 1.1828 (v9 long-train)
#   oxopax5h 1.1176 (v13 dropout=0.1)
#   fnyc7p7z 1.1041 (v14 dropout=0.2)
MEMBERS = [
    CKPT_DIR / "model-jnseenin" / "checkpoint.pt",
    CKPT_DIR / "model-9vaz4wrn" / "checkpoint.pt",
    CKPT_DIR / "model-w8rxftm4" / "checkpoint.pt",
    CKPT_DIR / "model-oxopax5h" / "checkpoint.pt",
    CKPT_DIR / "model-fnyc7p7z" / "checkpoint.pt",
]


@dataclass
class Config:
    agent: str = "edward"
    batch_size: int = 1
    splits_dir: str = str(SPLITS_DIR)


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

_, _, stats = load_data(cfg.splits_dir)

val_ds = GRAMDataset(splits_dir / "val")
val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=2, pin_memory=True)
print(f"val: {len(val_ds)} samples   |   {len(MEMBERS)} ensemble members")

# For each sample: accumulate predictions from each model, average at end.
# We iterate loader once per model (keeps memory low — only one model on GPU at a time).
sum_preds: list[torch.Tensor] = []
l2_per_model: list[float] = []
v_outs: list[torch.Tensor] = []

for m_idx, ckpt in enumerate(MEMBERS):
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model = build_from_checkpoint(state, stats, device)
    model.eval()
    print(f"[{m_idx+1}/{len(MEMBERS)}] {ckpt.parent.name}")

    total_l2 = 0.0
    n = 0
    with torch.no_grad():
        for i, (v_in, v_out, pos, t, idcs) in enumerate(tqdm(val_loader, leave=False)):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            pred = model(v_in, pos, t, idcs)

            l2 = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
            total_l2 += l2.sum().item()
            n += v_in.shape[0]

            if m_idx == 0:
                sum_preds.append(pred.cpu().clone())
                v_outs.append(v_out.cpu().clone())
            else:
                sum_preds[i] = sum_preds[i] + pred.cpu()

    l2_per_model.append(total_l2 / n)
    print(f"  single-model val/l2 = {l2_per_model[-1]:.4f}")
    del model
    torch.cuda.empty_cache()

# Compute ensemble predictions (mean) and val L2
K = len(MEMBERS)
total_l2_ens = 0.0
n_ens = 0
ensemble_preds = []
for i in range(len(sum_preds)):
    avg = sum_preds[i] / K
    ensemble_preds.append(avg.squeeze(0))  # (T_OUT, N, 3)
    l2 = (avg - v_outs[i]).norm(dim=3).mean(dim=(1, 2))
    total_l2_ens += l2.sum().item()
    n_ens += avg.shape[0]

val_l2_ens = total_l2_ens / n_ens
print(f"\nEnsemble (mean of {K} models) val/l2 = {val_l2_ens:.4f}")
print(f"Individual:  " + "  ".join(f"{x:.4f}" for x in l2_per_model))
print(f"Mean of individuals: {sum(l2_per_model)/K:.4f}")

# Save ensemble predictions in the same format predict.py uses
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / cfg.agent / commit
output_dir.mkdir(parents=True, exist_ok=True)
out_path = output_dir / "val.pt"
torch.save(ensemble_preds, out_path)
print(f"\nSaved ensemble predictions: {out_path}")
