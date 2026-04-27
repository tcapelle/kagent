"""Prediction averaging: forward each checkpoint, average outputs in physical units.

Unlike model soup (weight averaging), this averages model OUTPUTS — robust to
non-linearity. Useful when the constituent models are too correlated for weight
averaging but still have decorrelated errors.
"""

import json
import os
import subprocess
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import X_DIM, pad_collate, load_data
from train import Transolver

CKPT_DIR = Path("/mnt/new-pvc/kagent/apr27/frieren/checkpoints")
RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
TEST_SPLITS = [
    "test_single_in_dist", "test_geom_camber_rc",
    "test_geom_camber_cruise", "test_re_rand",
]
# Members to ensemble (newer ckpts have more weight).
MEMBERS = [
    ("model-4ydobzth", 1.0),  # iter5, val=47.17
    ("model-6ulvj74p", 1.5),  # iter6, val=46.14
    ("model-4hbvu7xe", 2.0),  # iter7, val=45.46
    ("model-qsywg20y", 2.5),  # iter9, val=44.73
]

device = torch.device("cuda")


def load_models():
    models = []
    weights = []
    for name, w in MEMBERS:
        with open(CKPT_DIR / name / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        m = Transolver(**cfg).to(device).eval()
        sd = torch.load(CKPT_DIR / name / "checkpoint.pt", map_location=device, weights_only=True)
        m.load_state_dict(sd)
        models.append(m)
        weights.append(w)
    total = sum(weights)
    weights = [w / total for w in weights]
    print(f"Ensemble: {[(n, f'{w:.3f}') for (n, _), w in zip(MEMBERS, weights)]}")
    return models, weights


def evaluate(models, weights, val_loaders, stats):
    out = {}
    for split, vl in val_loaders.items():
        s_err = 0.0; n_surf = 0
        with torch.no_grad():
            for x, y, is_surface, mask in vl:
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device); mask = mask.to(device)
                xn = (x - stats["x_mean"]) / stats["x_std"]
                pred_phys = None
                for m, w in zip(models, weights):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pn = m({"x": xn})["preds"].float()
                    p = pn * stats["y_std"] + stats["y_mean"]
                    pred_phys = w * p if pred_phys is None else pred_phys + w * p
                surf_mask = mask & is_surface
                err = (pred_phys[..., 2] - y[..., 2]).abs()
                s_err += (err * surf_mask.float()).sum().item()
                n_surf += surf_mask.sum().item()
        out[split] = s_err / max(n_surf, 1)
    return out


def predict_test(models, weights, stats, output_dir, batch_size=2):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in TEST_SPLITS:
        test_dir = SPLITS_DIR / split
        test_files = sorted(test_dir.glob("*.pt"))
        predictions = []
        for i in tqdm(range(0, len(test_files), batch_size), desc=split):
            batch_files = test_files[i:i + batch_size]
            samples = [torch.load(f, weights_only=True) for f in batch_files]
            xs = [s["x"] for s in samples]
            max_n = max(x.shape[0] for x in xs)
            B = len(xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            for j, x in enumerate(xs):
                x_pad[j, :x.shape[0]] = x.to(device)
            pred_phys = None
            with torch.no_grad():
                for m, w in zip(models, weights):
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pn = m({"x": (x_pad - stats["x_mean"]) / stats["x_std"]})["preds"].float()
                    p = pn * stats["y_std"] + stats["y_mean"]
                    pred_phys = w * p if pred_phys is None else pred_phys + w * p
            for j, x in enumerate(xs):
                predictions.append(pred_phys[j, :x.shape[0]].cpu())
        torch.save(predictions, output_dir / f"{split}.pt")
        print(f"  → {split} ({len(predictions)} samples)")


def main():
    train_ds, val_splits, stats, _ = load_data()
    stats = {k: v.to(device) for k, v in stats.items()}
    val_loaders = {
        name: DataLoader(ds, batch_size=2, shuffle=False, collate_fn=pad_collate, num_workers=4)
        for name, ds in val_splits.items()
    }

    models, weights = load_models()
    per_split = evaluate(models, weights, val_loaders, stats)
    avg = sum(per_split.values()) / 4
    print(f"ensemble  avg_surf_p={avg:.3f}  " + " ".join(f"{k.replace('val_','')}={v:.2f}" for k, v in per_split.items()))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    output_dir = PREDICTIONS_DIR / "frieren" / commit
    print(f"Saving predictions to {output_dir}")
    predict_test(models, weights, stats, output_dir)
    print(f"Done. ensemble val_avg_surf_p={avg:.2f}")


if __name__ == "__main__":
    main()
