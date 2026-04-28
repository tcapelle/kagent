"""Learn per-split per-channel SHIFT calibration from val residuals.

If the ensemble has systematic bias on some splits, a constant shift can correct
it. This is low-parameter (12 numbers: 4 splits × 3 channels) so unlikely to
overfit val. Optionally also a multiplicative scale (24 params).

Usage:
    python calibrate.py \
        --ckpts <ckpt1> <ckpt2> <ckpt3> \
        --weights 0.62 0.30 0.08 \
        --per_split_yaml per_split.yaml \
        --out_yaml calibration.yaml
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from torch.utils.data import DataLoader

from data import VAL_SPLIT_NAMES, load_data, pad_collate
from model import Transolver


@dataclass
class Config:
    ckpts: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    per_split_yaml: str | None = None  # if set, use per-split weights
    fit_scale: bool = False  # also fit multiplicative scale (else only shift)
    splits_dir: str = "/mnt/new-pvc/datasets/tandemfoil/splits_v2"
    out_yaml: str = "calibration.yaml"
    batch_size: int = 4
    surf_only: bool = True  # only optimize on surface nodes (since that's the metric)


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if cfg.per_split_yaml:
    with open(cfg.per_split_yaml) as f:
        ps_cfg = yaml.safe_load(f)
else:
    if not cfg.weights:
        cfg.weights = [1.0 / len(cfg.ckpts)] * len(cfg.ckpts)
    ps_cfg = None

# Load all unique checkpoints into a cache
all_ckpts = set()
if ps_cfg:
    for split, sc in ps_cfg.items():
        all_ckpts.update(sc["checkpoints"])
else:
    all_ckpts.update(cfg.ckpts)

cache_models: dict[str, torch.nn.Module] = {}
for ckpt in all_ckpts:
    config_path = Path(ckpt).parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    cache_models[ckpt] = m
    print(f"Loaded: {ckpt}")

_, val_splits, stats, _ = load_data(cfg.splits_dir)
stats = {k: v.to(device) for k, v in stats.items()}

val_loaders = {
    name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                     collate_fn=pad_collate, num_workers=2, pin_memory=True)
    for name, ds in val_splits.items()
}


@torch.no_grad()
def get_split_config(split_name):
    """Returns (checkpoints, weights) for a given val split."""
    if ps_cfg:
        # ps_cfg keys are test_*, val_* maps to test_* by suffix
        test_split = split_name.replace("val_", "test_")
        sc = ps_cfg[test_split]
        ckpts = sc["checkpoints"]
        weights = sc["weights"]
        weights = [w / sum(weights) for w in weights]
        return ckpts, weights
    return cfg.ckpts, [w / sum(cfg.weights) for w in cfg.weights]


calibration: dict = {}
print(f"\n{'split':<26} {'before_mae':<12} {'after_mae':<12} {'shift_Ux':<12} {'shift_Uy':<12} {'shift_p':<12}")
for split_name, vloader in val_loaders.items():
    ckpts, weights = get_split_config(split_name)
    sum_y, sum_p = torch.zeros(3, device=device), torch.zeros(3, device=device)
    n_pts = 0
    abs_err_before = torch.zeros(3, device=device)

    with torch.no_grad():
        for x, y, is_surface, mask in vloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            target_mask = mask & is_surface if cfg.surf_only else mask

            ensembled = None
            for ckpt, w in zip(ckpts, weights):
                m = cache_models[ckpt]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = m({"x": x_norm})["preds"].float()
                p_phys = p * stats["y_std"] + stats["y_mean"]
                ensembled = p_phys * w if ensembled is None else ensembled + p_phys * w

            mask3 = target_mask.unsqueeze(-1).expand_as(y)
            sum_y += y[mask3].view(-1, 3).sum(dim=0)
            sum_p += ensembled[mask3].view(-1, 3).sum(dim=0)
            n_pts += target_mask.sum().item()
            abs_err_before += (ensembled - y).abs()[mask3].view(-1, 3).sum(dim=0)

    # Shift = mean(y - pred) over surface nodes
    mean_y = sum_y / n_pts
    mean_p = sum_p / n_pts
    shift = mean_y - mean_p

    # Apply shift, recompute MAE
    abs_err_after = torch.zeros(3, device=device)
    n_pts2 = 0
    with torch.no_grad():
        for x, y, is_surface, mask in vloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_surface = is_surface.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_norm = (x - stats["x_mean"]) / stats["x_std"]
            target_mask = mask & is_surface if cfg.surf_only else mask

            ensembled = None
            for ckpt, w in zip(ckpts, weights):
                m = cache_models[ckpt]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = m({"x": x_norm})["preds"].float()
                p_phys = p * stats["y_std"] + stats["y_mean"]
                ensembled = p_phys * w if ensembled is None else ensembled + p_phys * w
            ensembled = ensembled + shift  # apply correction

            mask3 = target_mask.unsqueeze(-1).expand_as(y)
            abs_err_after += (ensembled - y).abs()[mask3].view(-1, 3).sum(dim=0)
            n_pts2 += target_mask.sum().item()

    mae_before = abs_err_before / n_pts
    mae_after = abs_err_after / n_pts2

    surf_p_before = mae_before[2].item()
    surf_p_after = mae_after[2].item()

    print(f"{split_name:<26} {surf_p_before:<12.4f} {surf_p_after:<12.4f} "
          f"{shift[0].item():<12.4f} {shift[1].item():<12.4f} {shift[2].item():<12.4f}")

    calibration[split_name.replace("val_", "test_")] = {
        "shift": [shift[0].item(), shift[1].item(), shift[2].item()],
    }

with open(cfg.out_yaml, "w") as f:
    yaml.dump(calibration, f)
print(f"\nSaved calibration to {cfg.out_yaml}")
