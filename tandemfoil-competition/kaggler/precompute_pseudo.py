"""Pre-compute ensemble pseudo-labels on training data for self-distillation.

Loads N checkpoints, runs uniform-weighted ensemble on each train sample,
saves predictions in NORMALIZED space (so the loss in train.py can use them
directly without re-normalization). Output saved under <out_dir>/000000.pt etc.
matching the train data filenames.

Run AFTER v22 if needed:
    python precompute_pseudo.py \
        --checkpoints models/model-izweqran/checkpoint.pt \
                      models/model-8fazsiug/checkpoint.pt \
                      models/model-bxvp8448/checkpoint.pt \
        --out_dir /workspace/kagent/tandemfoil-competition/kaggler/pseudo_train
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from tqdm import tqdm

from data import X_DIM
from model import Transolver

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")


@dataclass
class Config:
    checkpoints: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    out_dir: str = "/workspace/kagent/tandemfoil-competition/kaggler/pseudo_train"
    splits_dir: str = str(SPLITS_DIR)


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load stats
with open(splits_dir / "stats.json") as f:
    s = json.load(f)
x_mean = torch.tensor(s["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(s["x_std"], dtype=torch.float32, device=device)

# Load models
models = []
for ckpt in cfg.checkpoints:
    config_path = Path(ckpt).parent / "config.yaml"
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
    m = Transolver(**model_config).to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)
    print(f"Loaded: {ckpt}")

if not cfg.weights:
    cfg.weights = [1.0 / len(models)] * len(models)
weights = [w / sum(cfg.weights) for w in cfg.weights]
print(f"Weights: {weights}")

train_dir = splits_dir / "train"
out_dir = Path(cfg.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

train_files = sorted(train_dir.glob("*.pt"))
print(f"Pseudo-labeling {len(train_files)} train samples → {out_dir}")

with torch.no_grad():
    for f in tqdm(train_files):
        sample = torch.load(f, weights_only=True)
        x = sample["x"].to(device).unsqueeze(0)
        x_norm = (x - x_mean) / x_std

        ensembled = None
        for m, w in zip(models, weights):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p = m({"x": x_norm})["preds"].float()  # normalized space
            ensembled = p * w if ensembled is None else ensembled + p * w
        # Save in normalized space; store as float16 to save disk
        torch.save({"pseudo_y_norm": ensembled.squeeze(0).half().cpu()}, out_dir / f.name)
print("Done.")
