"""Average multiple model checkpoints (Stochastic Weight Averaging)
and evaluate against the per-checkpoint baseline. Optionally write the
SWA model as the new `checkpoints/best.pt` if it beats the inputs.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, X_DIM, load_data, pad_collate
from model import Transolver


def load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def average_state_dicts(states):
    keys = states[0].keys()
    avg = {}
    for k in keys:
        ts = [s[k] for s in states]
        if ts[0].dtype.is_floating_point:
            avg[k] = torch.stack(ts, dim=0).mean(dim=0)
        else:
            avg[k] = ts[0].clone()
    return avg


def evaluate(model, val_loaders, stats, device, amp_dtype=torch.bfloat16):
    model.eval()
    out = {}
    sum_p = 0.0
    for split_name, vloader in val_loaders.items():
        mae_surf = torch.zeros(3, device=device)
        n_surf = 0
        with torch.no_grad():
            for x, y, is_surface, mask in vloader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                is_surface = is_surface.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                xn = (x - stats["x_mean"]) / stats["x_std"]
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pred = model({"x": xn})["preds"]
                pred = pred.float()
                pred_orig = pred * stats["y_std"] + stats["y_mean"]
                surf_mask = mask & is_surface
                err = (pred_orig - y).abs()
                mae_surf += (err * surf_mask.unsqueeze(-1)).sum(dim=(0, 1))
                n_surf += surf_mask.sum().item()
        mae_surf /= max(n_surf, 1)
        out[split_name] = {
            f"{split_name}/mae_surf_Ux": mae_surf[0].item(),
            f"{split_name}/mae_surf_Uy": mae_surf[1].item(),
            f"{split_name}/mae_surf_p": mae_surf[2].item(),
        }
        sum_p += mae_surf[2].item()
    return out, sum_p / len(val_loaders)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="Paths to checkpoints to average")
    ap.add_argument("--config", default=None,
                    help="Path to model config.yaml; defaults to baseline")
    ap.add_argument("--output", default="checkpoints/swa.pt")
    ap.add_argument("--splits_dir", default="/mnt/new-pvc/datasets/tandemfoil/splits_v2")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--update_best", action="store_true",
                    help="Overwrite checkpoints/best.pt if SWA beats best individual")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            model_config = yaml.safe_load(f)
    else:
        model_config = dict(
            space_dim=2, fun_dim=X_DIM - 2, out_dim=3, n_hidden=256, n_layers=8,
            n_head=8, slice_num=96, mlp_ratio=2, dropout=0.0,
        )
    model_config["use_checkpoint"] = False

    train_ds, val_splits, stats, _ = load_data(args.splits_dir, debug=False)
    stats = {k: v.to(device) for k, v in stats.items()}
    val_loaders = {
        name: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         collate_fn=pad_collate, num_workers=2, pin_memory=True)
        for name, ds in val_splits.items()
    }

    # Evaluate each individual checkpoint
    states = [load_ckpt(p) for p in args.checkpoints]
    individual_scores = {}
    for path, state in zip(args.checkpoints, states):
        m = Transolver(**model_config).to(device)
        m.load_state_dict(state)
        _, mean_p = evaluate(m, val_loaders, stats, device)
        individual_scores[path] = mean_p
        print(f"  {path}: mean_surf_p = {mean_p:.4f}")
        del m

    # Average and evaluate
    avg_state = average_state_dicts(states)
    m = Transolver(**model_config).to(device)
    m.load_state_dict(avg_state)
    swa_split, swa_mean = evaluate(m, val_loaders, stats, device)
    print(f"\nSWA: mean_surf_p = {swa_mean:.4f}")
    for split_name, sm in swa_split.items():
        print(f"  {split_name}/mae_surf_p = {sm[f'{split_name}/mae_surf_p']:.2f}")

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(avg_state, args.output)
    print(f"\nSaved SWA ckpt to {args.output}")

    best_individual = min(individual_scores.values())
    if args.update_best and swa_mean < best_individual:
        torch.save(avg_state, "checkpoints/best.pt")
        print(f"\n[update] best.pt replaced (SWA {swa_mean:.4f} < best {best_individual:.4f})")
    elif args.update_best:
        print(f"\nSWA did not beat best individual ({swa_mean:.4f} >= {best_individual:.4f}); best.pt unchanged")


if __name__ == "__main__":
    main()
