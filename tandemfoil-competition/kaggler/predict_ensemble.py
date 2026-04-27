"""Average predictions across multiple checkpoints (output-space ensemble).

Run all checkpoints sequentially on every test sample and average the resulting
3-channel tensors. Use when individual models in the same architecture family
already reached the basin floor — averaging predictions adds genuine diversity
that weight-averaging in the same basin does not.

Layout:
  /mnt/new-pvc/predictions/<tag>/<agent>/<commit>/
  ├── test_single_in_dist.pt
  ├── test_geom_camber_rc.pt
  ├── test_geom_camber_cruise.pt
  └── test_re_rand.pt

Run:
  python predict_ensemble.py \
    --checkpoints ck1.pt ck2.pt ck3.pt \
    --configs cfg1.yaml cfg2.yaml cfg3.yaml \
    --agent <name>
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from data import X_DIM
from train import Transolver

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", type=Path,
                    help="Per-checkpoint config.yaml. Defaults to alongside the ckpt.")
    ap.add_argument("--splits_dir", type=Path, default=SPLITS_DIR)
    ap.add_argument("--agent", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="Per-checkpoint weights for the average (default: equal).")
    args = ap.parse_args()

    device = torch.device("cuda")
    splits_dir = Path(args.splits_dir)

    if args.configs is None:
        args.configs = [c.parent / "config.yaml" for c in args.checkpoints]
    assert len(args.configs) == len(args.checkpoints)

    if args.weights is None:
        args.weights = [1.0] * len(args.checkpoints)
    assert len(args.weights) == len(args.checkpoints)
    w_sum = sum(args.weights)
    weights = [w / w_sum for w in args.weights]

    # Load every model into GPU (they're small — ~7MB each).
    models = []
    for ckpt_path, cfg_path in zip(args.checkpoints, args.configs):
        with open(cfg_path) as f:
            mc = yaml.safe_load(f)
        m = Transolver(**mc).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        m.load_state_dict(sd, strict=False)
        m.eval()
        models.append(m)
        print(f"Loaded {ckpt_path}")

    with open(splits_dir / "stats.json") as f:
        stats_data = json.load(f)
    x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
    y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"
    output_dir = PREDICTIONS_DIR / args.agent / commit
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in TEST_SPLITS:
        test_dir = splits_dir / split
        test_files = sorted(test_dir.glob("*.pt"))
        print(f"{split}: {len(test_files)} samples")

        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(test_files), args.batch_size), desc=split, leave=False):
                batch_files = test_files[i:i + args.batch_size]
                samples = [torch.load(f, weights_only=True) for f in batch_files]
                xs = [s["x"] for s in samples]

                max_n = max(x.shape[0] for x in xs)
                B = len(xs)
                x_pad = torch.zeros(B, max_n, X_DIM, device=device)
                for j, x in enumerate(xs):
                    x_pad[j, :x.shape[0]] = x.to(device)

                x_in = (x_pad - x_mean) / x_std
                ens = torch.zeros(B, max_n, 3, device=device)
                for w, m in zip(weights, models):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_norm = m({"x": x_in})["preds"]
                    pred = pred_norm.float() * y_std + y_mean
                    ens = ens + w * pred

                for j, x in enumerate(xs):
                    predictions.append(ens[j, :x.shape[0]].cpu())

        output_path = output_dir / f"{split}.pt"
        torch.save(predictions, output_path)
        print(f"  → {output_path} ({len(predictions)} samples)")

    print(f"\nEnsemble of {len(models)} ckpts saved to {output_dir}")


if __name__ == "__main__":
    main()
