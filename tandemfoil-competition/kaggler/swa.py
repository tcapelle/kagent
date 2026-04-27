"""Stochastic Weight Averaging across model checkpoints.

Loads N checkpoints, averages their state_dicts (with optional weights),
saves the averaged checkpoint. Run predict.py with the resulting model.

Usage:
  python swa.py --checkpoints /tmp/iter4_best.pt /tmp/iter6_best.pt \
                --weights 0.5 0.5 \
                --out models/swa-iter4-6/checkpoint.pt
"""

import argparse
from pathlib import Path

import torch
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    p.add_argument("--out", required=True, help="output checkpoint path")
    p.add_argument("--config", default=None,
                   help="path to config.yaml; defaults to first checkpoint's sibling")
    args = p.parse_args()

    w = args.weights or [1.0] * len(args.checkpoints)
    assert len(w) == len(args.checkpoints)
    tot = sum(w)
    w = [x / tot for x in w]

    sds = [torch.load(c, map_location="cpu", weights_only=True) for c in args.checkpoints]
    avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sds[0].items()}
    for sd, weight in zip(sds, w):
        for k in avg:
            avg[k] += weight * sd[k].float()
    for k in avg:
        avg[k] = avg[k].to(sds[0][k].dtype)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, out_path)
    print(f"Saved SWA of {len(args.checkpoints)} checkpoints to {out_path}")

    cfg_src = Path(args.config) if args.config else Path(args.checkpoints[0]).parent / "config.yaml"
    if cfg_src.exists():
        import shutil
        shutil.copy2(cfg_src, out_path.parent / "config.yaml")
        print(f"Copied config from {cfg_src}")


if __name__ == "__main__":
    main()
