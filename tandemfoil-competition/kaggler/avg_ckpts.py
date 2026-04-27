"""Average state-dicts of compatible checkpoints (Polyak / SWA-style).

Use when several models trained from a common warm-start sit in the same loss
basin — the average often generalizes better than any single one.

Run:
  python avg_ckpts.py --inputs ck1.pt ck2.pt --output ck_avg.pt
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    sds = []
    for p in args.inputs:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sds.append(sd)

    keys = sds[0].keys()
    for sd in sds[1:]:
        if sd.keys() != keys:
            raise SystemExit(f"State-dict keys mismatch between inputs")

    avg = {}
    for k in keys:
        stacked = torch.stack([sd[k].float() for sd in sds])
        avg[k] = stacked.mean(0).to(sds[0][k].dtype)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(avg, args.output)
    print(f"Averaged {len(sds)} checkpoints → {args.output}")


if __name__ == "__main__":
    main()
