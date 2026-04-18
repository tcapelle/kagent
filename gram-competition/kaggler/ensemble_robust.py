"""Robust ensemble methods: median, trimmed-mean, per-point top-k.

Compares mean (weighted by softmax), median, and trimmed-mean for the top-k
subset of seeds. For quadratic L2 error, mean is optimal in expectation but
fragile to per-point outlier predictions; median is robust to single-model
outliers at the cost of ignoring the 'wisdom of the crowd'.

Usage: python ensemble_robust.py --checkpoints ... --agent ...
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


def pointwise_median(preds_all):
    """Per-point per-component median across ckpts."""
    n = len(preds_all[0])
    out = []
    for si in range(n):
        stacked = torch.stack([preds_all[ci][si] for ci in range(len(preds_all))], dim=0)
        out.append(torch.median(stacked, dim=0).values)
    return out


def pointwise_trimmed_mean(preds_all, trim_frac=0.125):
    """Per-point trimmed mean: drop top and bottom k, average the rest."""
    n = len(preds_all[0])
    K = len(preds_all)
    trim = max(1, int(K * trim_frac))
    out = []
    for si in range(n):
        stacked = torch.stack([preds_all[ci][si] for ci in range(K)], dim=0)  # [K, T, N, 3]
        sorted_vals, _ = torch.sort(stacked, dim=0)
        kept = sorted_vals[trim:K - trim]
        out.append(kept.mean(dim=0))
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
    # For each k: try mean(uniform), mean(softmax), median, trimmed-mean
    for k in [12, 14, 15, 16, 18, 20, K]:
        if k > K:
            continue
        keep = sorted_idx[:k]
        sub_preds = [all_preds[i] for i in keep]
        sub_l2 = l2[torch.tensor(keep)]

        # Mean + softmax T=0.02 (baseline)
        w_sm = torch.softmax(-sub_l2 / 0.02, dim=0)
        preds_sm = weighted_mean(sub_preds, w_sm)
        loss_sm = compute_l2(preds_sm, gt_all)
        results.append([f"top{k}_mean_T0.02", loss_sm, preds_sm])

        # Median
        preds_med = pointwise_median(sub_preds)
        loss_med = compute_l2(preds_med, gt_all)
        results.append([f"top{k}_median", loss_med, preds_med])

        # Trimmed mean (drop 12.5% top/bottom)
        if k >= 8:
            preds_tm = pointwise_trimmed_mean(sub_preds, trim_frac=0.125)
            loss_tm = compute_l2(preds_tm, gt_all)
            results.append([f"top{k}_trim12.5", loss_tm, preds_tm])

        print(f"  top{k}: mean_T0.02={loss_sm:.4f}  median={loss_med:.4f}"
              + (f"  trim12.5={loss_tm:.4f}" if k >= 8 else ""))

    results.sort(key=lambda x: x[1])
    print(f"\n  Top 10 schemes:")
    for name, loss, _ in results[:10]:
        print(f"    {name:25s} {loss:.4f}")

    best_name, best_loss, best_preds = results[0]
    print(f"\n  BEST: {best_name}  val/l2 = {best_loss:.4f}")
    if best_preds is not None:
        output_path = output_dir / f"{split_name}.pt"
        torch.save(best_preds, output_path)
        print(f"  -> {output_path} (scheme={best_name})")
