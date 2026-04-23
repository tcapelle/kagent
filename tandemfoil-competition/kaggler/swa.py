"""Stochastic Weight Averaging across multiple checkpoints.

Averages state dicts element-wise and saves to a new checkpoint directory
that predict.py can load directly (has checkpoint.pt + config.yaml).

Usage:
  python swa.py --checkpoints /tmp/iter17_best.pt /tmp/iter19_best.pt /tmp/iter21_best.pt \
                --config_yaml /workspace/kagent/tandemfoil-competition/kaggler/models/model-ogycayte/config.yaml \
                --out_dir models/model-swa
"""

import argparse
import shutil
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--config_yaml", required=True, help="path to a config.yaml (all checkpoints must share arch)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--weights", nargs="+", type=float, default=None)
    args = p.parse_args()

    ws = args.weights or [1.0] * len(args.checkpoints)
    assert len(ws) == len(args.checkpoints)
    tot = sum(ws)
    ws = [w / tot for w in ws]
    print(f"Averaging {len(args.checkpoints)} checkpoints with weights {ws}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    avg = None
    for ckpt, w in zip(args.checkpoints, ws):
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        if avg is None:
            avg = {k: w * v.float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg[k] = avg[k] + w * v.float()
        print(f"  loaded {ckpt}")

    torch.save(avg, out_dir / "checkpoint.pt")
    shutil.copy(args.config_yaml, out_dir / "config.yaml")
    print(f"Saved SWA checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
