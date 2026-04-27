"""Evaluate a prediction-averaging ensemble across val splits.

Loads multiple checkpoints, averages their normalized outputs, computes
surface-pressure MAE per split, and reports the ensemble vs. each member.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, X_DIM, load_data, pad_collate
from model import Transolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Optional per-checkpoint weights for the ensemble (will be normalized).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()
    if args.weights is None:
        args.weights = [1.0] * len(args.checkpoints)
    assert len(args.weights) == len(args.checkpoints), "weights/checkpoints length mismatch"
    w_total = sum(args.weights)
    args.weights = [w / w_total for w in args.weights]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            mc = yaml.safe_load(f)
    else:
        mc = dict(space_dim=2, fun_dim=X_DIM - 2, out_dim=3, n_hidden=256, n_layers=8,
                  n_head=8, slice_num=96, mlp_ratio=2, dropout=0.0)
    mc["use_checkpoint"] = False

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=2, pin_memory=True)
        for name, ds in val_splits.items()
    }

    models = []
    for path in args.checkpoints:
        m = Transolver(**mc).to(device)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
        print(f"Loaded {path}")

    amp_dtype = torch.bfloat16

    def eval_pred_set(get_pred_norm):
        sum_p = 0.0
        per_split = {}
        for split_name, vloader in val_loaders.items():
            mae_surf = torch.zeros(3, device=device)
            n_surf = 0
            with torch.no_grad():
                for x, y, is_surface, mask in vloader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    is_surface = is_surface.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    xn = (x - stats["x_mean"]) / stats["x_std"]
                    pred_norm = get_pred_norm(xn)
                    pred_orig = pred_norm * stats["y_std"] + stats["y_mean"]
                    surf_mask = mask & is_surface
                    err = (pred_orig - y).abs()
                    mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                    n_surf += surf_mask.sum().item()
            mae_surf /= max(n_surf, 1)
            per_split[split_name] = mae_surf[2].item()
            sum_p += mae_surf[2].item()
        return sum_p / len(val_loaders), per_split

    # Per-model
    print("\n=== Individual ===")
    for path, m in zip(args.checkpoints, models):
        def gp(xn, _m=m):
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                return _m({"x": xn})["preds"].float()
        mean_p, per = eval_pred_set(gp)
        print(f"  {Path(path).parent.name}: mean={mean_p:.4f} per_split={per}")

    # Ensemble
    print(f"\n=== Ensemble weights={args.weights} ===")
    weights = torch.tensor(args.weights, device=device).view(-1, 1, 1, 1)
    def gp_ens(xn):
        outs = []
        for m in models:
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                outs.append(m({"x": xn})["preds"].float())
        return (torch.stack(outs, 0) * weights).sum(0)
    mean_p, per = eval_pred_set(gp_ens)
    print(f"  ensemble: mean={mean_p:.4f} per_split={per}")


if __name__ == "__main__":
    main()
