"""Model soup: average weights of multiple chain checkpoints to reduce variance.

Loads a sequence of fine-tuned checkpoints, averages their parameters, validates
the soup on the four val splits, and runs predictions on the test splits.
"""

import json
import os
import subprocess
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import X_DIM, VAL_SPLIT_NAMES, pad_collate, load_data
from train import Transolver

CKPT_DIR = Path("/mnt/new-pvc/kagent/apr27/frieren/checkpoints")
RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
TEST_SPLITS = [
    "test_single_in_dist", "test_geom_camber_rc",
    "test_geom_camber_cruise", "test_re_rand",
]
# Chain checkpoints (oldest → newest), all hid=192 L=6 S=64.
CHAIN = ["model-s8nqhr0q", "model-d215g7ng", "model-4ydobzth",
         "model-6ulvj74p", "model-4hbvu7xe"]

device = torch.device("cuda")


def load_soup(ckpt_names: list[str], weights: list[float] | None = None) -> tuple[Transolver, dict]:
    """Average weights from multiple checkpoints. Returns model + reference config."""
    weights = weights or [1.0 / len(ckpt_names)] * len(ckpt_names)
    assert abs(sum(weights) - 1.0) < 1e-6
    cfg = None
    avg = None
    for name, w in zip(ckpt_names, weights):
        with open(CKPT_DIR / name / "config.yaml") as f:
            c = yaml.safe_load(f)
        if cfg is None:
            cfg = c
        else:
            assert c == cfg, f"Architecture mismatch: {name}"
        sd = torch.load(CKPT_DIR / name / "checkpoint.pt", map_location=device, weights_only=True)
        if avg is None:
            avg = {k: v.float() * w for k, v in sd.items()}
        else:
            for k, v in sd.items():
                avg[k] = avg[k] + v.float() * w
    model = Transolver(**cfg).to(device).eval()
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in avg.items()})
    return model, cfg


def evaluate(model, val_loaders, stats):
    """Compute avg_surf_p MAE across all val splits."""
    model.eval()
    out = {}
    for split, vl in val_loaders.items():
        s_err = 0.0; n_surf = 0
        per_chan = torch.zeros(3, device=device); n_all = 0
        with torch.no_grad():
            for x, y, is_surface, mask in vl:
                x, y = x.to(device), y.to(device)
                is_surface = is_surface.to(device); mask = mask.to(device)
                xn = (x - stats["x_mean"]) / stats["x_std"]
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = model({"x": xn})["preds"].float()
                p = pred * stats["y_std"] + stats["y_mean"]
                surf_mask = mask & is_surface
                err = (p[..., 2] - y[..., 2]).abs()
                s_err += (err * surf_mask.float()).sum().item()
                n_surf += surf_mask.sum().item()
        out[split] = s_err / max(n_surf, 1)
    return out


def predict_test(model, stats, output_dir, batch_size=2):
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
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred_norm = model({"x": (x_pad - stats["x_mean"]) / stats["x_std"]})["preds"].float()
            pred = pred_norm * stats["y_std"] + stats["y_mean"]
            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())
        torch.save(predictions, output_dir / f"{split}.pt")
        print(f"  → {split} ({len(predictions)} samples)")


def main():
    train_ds, val_splits, stats, _ = load_data()
    stats = {k: v.to(device) for k, v in stats.items()}
    val_loaders = {
        name: DataLoader(ds, batch_size=2, shuffle=False, collate_fn=pad_collate, num_workers=4)
        for name, ds in val_splits.items()
    }

    # Try different soup compositions; pick the best by val.
    candidates = {
        "all-5": (CHAIN, None),
        "last-3": (CHAIN[-3:], None),
        "last-2": (CHAIN[-2:], None),
        "weighted-by-rank": (CHAIN, [0.05, 0.10, 0.20, 0.30, 0.35]),
        "iter7-only": (CHAIN[-1:], None),
    }
    best_name, best_avg, best_model = None, float("inf"), None
    for name, (members, w) in candidates.items():
        if w is None:
            w = [1.0 / len(members)] * len(members)
        model, _ = load_soup(members, w)
        per_split = evaluate(model, val_loaders, stats)
        avg = sum(per_split.values()) / 4
        print(f"{name:24s} avg_surf_p={avg:.3f}  " + " ".join(f"{k.replace('val_',''):8s}={v:.2f}" for k, v in per_split.items()))
        if avg < best_avg:
            best_avg = avg
            best_name = name
            best_model = model

    print(f"\nBest soup: {best_name} (avg_surf_p={best_avg:.3f})")

    # Save predictions under current commit hash.
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    output_dir = PREDICTIONS_DIR / "frieren" / commit
    print(f"Saving predictions to {output_dir}")
    predict_test(best_model, stats, output_dir)
    print(f"Done. soup={best_name}, val_avg_surf_p={best_avg:.2f}")


if __name__ == "__main__":
    main()
