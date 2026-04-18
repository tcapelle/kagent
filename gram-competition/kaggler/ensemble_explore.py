"""Explore ensemble schemes beyond weighted mean.

Loads N checkpoints, runs val inference once per checkpoint, then tries:
  - temperature grid (softmax weighting with T in {0.008..0.04})
  - median-of-predictions (per-scalar median)
  - top-k subset weighted-mean (pick top-k by solo val/l2, softmax-weight)
  - trim worst-k (drop k weakest, weighted-mean the rest)

Reports val/l2 for each, saves the best to the standard predictions path.
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GRAMDataset, load_data
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
assert len(cfg.checkpoints) >= 2
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


def median_ens(preds_all):
    n = len(preds_all[0])
    out = []
    for si in range(n):
        stacked = torch.stack([p[si] for p in preds_all])  # [K, 5, N, 3]
        med = stacked.median(dim=0).values
        out.append(med)
    return out


for split_name, base in val_splits.items():
    print(f"\n{split_name}: {len(base)} samples")
    print("  Precomputing SDF...")
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
    print(f"\n  K={K} seeds. Per-seed solo: min={l2.min():.4f}, max={l2.max():.4f}")

    results: list[tuple[str, float, list[torch.Tensor]]] = []

    # (1) Temperature grid
    print("\n  === softmax temperature grid ===")
    for T in [0.008, 0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030, 0.040]:
        w = torch.softmax(-l2 / T, dim=0)
        preds = weighted_mean(all_preds, w)
        loss = compute_l2(preds, gt_all)
        print(f"  T={T:.3f} loss={loss:.4f} wmin={w.min():.3f} wmax={w.max():.3f}")
        results.append((f"softmax_T{T}", loss, preds))

    # (2) Median
    print("\n  === median ensemble ===")
    preds = median_ens(all_preds)
    loss = compute_l2(preds, gt_all)
    print(f"  median loss={loss:.4f}")
    results.append(("median", loss, preds))

    # (3) Top-k subset + softmax T=0.02
    print("\n  === top-k subset (softmax T=0.02) ===")
    sorted_idx = torch.argsort(l2).tolist()
    for k in [5, 8, 10, 12, 15, K]:
        if k > K:
            continue
        keep = sorted_idx[:k]
        sub_preds = [all_preds[i] for i in keep]
        sub_l2 = l2[torch.tensor(keep)]
        w = torch.softmax(-sub_l2 / 0.02, dim=0)
        preds = weighted_mean(sub_preds, w)
        loss = compute_l2(preds, gt_all)
        print(f"  top-{k} loss={loss:.4f} (dropped {K-k} weakest)")
        results.append((f"top{k}_T0.02", loss, preds))

    # (4) Trim-worst-k (alias for top-k but named differently)
    # same as top-k, skip

    # (5) Median over top-k
    print("\n  === median over top-k ===")
    for k in [5, 8, 10, 12]:
        if k > K:
            continue
        keep = sorted_idx[:k]
        sub_preds = [all_preds[i] for i in keep]
        preds = median_ens(sub_preds)
        loss = compute_l2(preds, gt_all)
        print(f"  top-{k} median loss={loss:.4f}")
        results.append((f"top{k}_median", loss, preds))

    # Pick best
    results.sort(key=lambda x: x[1])
    print("\n  Top 5:")
    for name, loss, _ in results[:5]:
        print(f"    {name:20s} {loss:.4f}")

    best_name, best_loss, best_preds = results[0]
    print(f"\n  BEST: {best_name}  val/l2 = {best_loss:.4f}")
    output_path = output_dir / f"{split_name}.pt"
    torch.save(best_preds, output_path)
    print(f"  -> {output_path} ({len(best_preds)} samples, scheme={best_name})")
