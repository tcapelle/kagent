"""Average predictions from several checkpoints (output-space ensembling).

Usage:
  python predict_ensemble.py --ckpts path1 path2 ... --agent thorfinn

Each checkpoint is expected to have a sibling config.yaml. All checkpoints must
use the same architecture (same Transolver hyperparameters) — otherwise output
shapes won't align.
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
from model import Transolver

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
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--agent", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_bf16", action="store_true", default=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for ckpt_path in args.ckpts:
        ckpt_path = Path(ckpt_path)
        cfg_path = ckpt_path.parent / "config.yaml"
        with open(cfg_path) as f:
            mc = yaml.safe_load(f)
        m = Transolver(**mc).to(device)
        m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        m.eval()
        models.append(m)
        print(f"Loaded {ckpt_path}")
    print(f"Ensemble size: {len(models)}")

    with open(SPLITS_DIR / "stats.json") as f:
        stats = json.load(f)
    x_mean = torch.tensor(stats["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(stats["x_std"], dtype=torch.float32, device=device)
    y_mean = torch.tensor(stats["y_mean"], dtype=torch.float32, device=device)
    y_std = torch.tensor(stats["y_std"], dtype=torch.float32, device=device)

    agent_name = args.agent or os.environ.get("KAGGLER_NAME", "unknown")
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"
    output_dir = PREDICTIONS_DIR / agent_name / commit
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) \
        if args.use_bf16 and device.type == "cuda" \
        else (lambda: torch.cuda.amp.autocast(enabled=False))

    for split in TEST_SPLITS:
        test_dir = SPLITS_DIR / split
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
                preds_norm = None
                for m in models:
                    with autocast_ctx():
                        p = m({"x": x_in})["preds"].float()
                    preds_norm = p if preds_norm is None else preds_norm + p
                preds_norm /= len(models)
                pred_phys = preds_norm * y_std + y_mean

                for j, x in enumerate(xs):
                    predictions.append(pred_phys[j, :x.shape[0]].cpu())

        out = output_dir / f"{split}.pt"
        torch.save(predictions, out)
        print(f"  → {out} ({len(predictions)} samples)")

    print(f"\nAll ensemble predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
