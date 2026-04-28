"""Weight-average several checkpoints into one (stochastic weight averaging).

Usage:
  python swa.py --ckpts path1 path2 ... --out checkpoints/swa.pt

Then run predict.py against `--out`. The averaged ckpt typically generalizes
slightly better than any single checkpoint when they come from late chained
fine-tunes around a basin minimum.
"""

import argparse
from pathlib import Path

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="Paths to checkpoint .pt files (state_dicts).")
    ap.add_argument("--out", required=True, help="Output .pt path for averaged weights.")
    ap.add_argument("--config_from", default=None,
                    help="Path to a config.yaml to copy alongside the SWA ckpt.")
    args = ap.parse_args()

    paths = [Path(p) for p in args.ckpts]
    print(f"Averaging {len(paths)} checkpoints:")
    for p in paths:
        print(f"  - {p}")

    states = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    keys = states[0].keys()
    avg = {}
    for k in keys:
        stack = torch.stack([s[k].float() for s in states])
        avg[k] = stack.mean(dim=0).to(states[0][k].dtype)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, out)
    print(f"Saved averaged weights → {out}")

    if args.config_from:
        cfg_src = Path(args.config_from)
        cfg_dst = out.parent / "config.yaml"
        cfg_dst.write_text(cfg_src.read_text())
        print(f"Copied config → {cfg_dst}")


if __name__ == "__main__":
    main()
