"""Grid search: for top-k subset, find optimal softmax temperature T.

Usage: python ensemble_topk_grid.py --checkpoints ... --agent ...
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import load_data
from train import VoxelResidualModel, compute_sdf, SDFDataset, collate_sdf, infer_arch_from_state_dict

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


def weighted_mean(preds_all, w):
    n = len(preds_all[0])
    out = []
    for si in range(n):
        acc = torch.zeros_like(preds_all[0][si])
        for ci in range(len(preds_all)):
            acc += w[ci].item() * preds_all[ci][si]
        out.append(acc)
    return out


for split_name, base in val_splits.items():
    print(f"\n{split_name}: {len(base)} samples")
    sdfs = [compute_sdf(base[i][2], base[i][4], device)
            for i in tqdm(range(len(base)), desc=f"{split_name} SDF")]
    ds = SDFDataset(base, sdfs)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_sdf)

    all_preds: list[list[torch.Tensor]] = []
    gt_all: list[torch.Tensor] = []
    per_seed_l2: list[float] = []

    for ck_idx, ckpt_path in enumerate(cfg.checkpoints):
        print(f"  [{ck_idx+1}/{len(cfg.checkpoints)}] {ckpt_path}")
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        arch = infer_arch_from_state_dict(sd)
        model = VoxelResidualModel(
            vel_mean=stats["vel_mean"], vel_std=stats["vel_std"], **arch,
        ).to(device)
        model.load_state_dict(sd)
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

    K = len(cfg.checkpoints)
    l2 = torch.tensor(per_seed_l2)
    print(f"\n  K={K} per-seed: min={l2.min():.4f}, max={l2.max():.4f}")

    sorted_idx = torch.argsort(l2).tolist()

    results = []
    for k in [8, 10, 12, 14, 15, 16, 18, 20, K]:
        if k > K:
            continue
        keep = sorted_idx[:k]
        sub_preds = [all_preds[i] for i in keep]
        sub_l2 = l2[torch.tensor(keep)]
        # Also try uniform weighting for each k (baseline)
        w_uni = torch.ones(k) / k
        loss_uni = compute_l2(weighted_mean(sub_preds, w_uni), gt_all)
        row = [f"top{k}_uniform", loss_uni, None]
        results.append(row)
        best_row = row
        for T in [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030]:
            w = torch.softmax(-sub_l2 / T, dim=0)
            preds = weighted_mean(sub_preds, w)
            loss = compute_l2(preds, gt_all)
            row = [f"top{k}_T{T}", loss, preds]
            results.append(row)
            if loss < best_row[1]:
                best_row = row
        print(f"  top{k}: best={best_row[0]} {best_row[1]:.4f} (uniform={loss_uni:.4f})")

    results.sort(key=lambda x: x[1])
    print(f"\n  Top 10 schemes:")
    for name, loss, _ in results[:10]:
        print(f"    {name:22s} {loss:.4f}")

    # Save best
    best_name, best_loss, best_preds = results[0]
    print(f"\n  BEST: {best_name}  val/l2 = {best_loss:.4f}")
    if best_preds is not None:
        output_path = output_dir / f"{split_name}.pt"
        torch.save(best_preds, output_path)
        print(f"  -> {output_path} (scheme={best_name})")
