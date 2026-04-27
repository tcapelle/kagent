"""Per-split-aware predictions: each test split uses its own optimal model set.

Routing (based on per-split test scores):
  test_single_in_dist  -> ensemble [warm, iter1, iter2, iter3]   (40.28 floor)
  test_geom_camber_rc  -> warm-start solo                        (61.70 floor)
  test_geom_camber_cruise -> ensemble [warm, iter2, iter3]       (27.95 floor)
  test_re_rand         -> warm-start solo                        (51.05 floor)

Run:
  python predict_router.py --agent thorfinn
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import yaml
from tqdm import tqdm

from data import X_DIM
from model import Transolver

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")

WARM = "/mnt/new-pvc/kagent/apr27-bis/thorfinn/checkpoints/model-e3itadc2/checkpoint.pt"
ITER1 = "models/model-jbbynlph/checkpoint.pt"
ITER2 = "models/model-833lzt0u/checkpoint.pt"
ITER3 = "models/model-chbzghhz/checkpoint.pt"

ROUTES = {
    "test_single_in_dist": [WARM, ITER1, ITER2, ITER3],
    "test_geom_camber_rc": [WARM],
    "test_geom_camber_cruise": [WARM, ITER2, ITER3],
    "test_re_rand": [WARM],
}


@dataclass
class Config:
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)


def load(path: str) -> Transolver:
    p = Path(path)
    mc_path = p.parent / "config.yaml"
    if mc_path.exists():
        with open(mc_path) as f:
            mc = yaml.safe_load(f)
    else:
        mc = dict(space_dim=0, fun_dim=X_DIM, out_dim=3,
                  n_hidden=192, n_layers=6, n_head=6, slice_num=128, mlp_ratio=2,
                  output_fields=["Ux", "Uy", "p"], output_dims=[1, 1, 1])
    m = Transolver(**mc).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    return m


# Cache models so we don't reload them across splits.
model_cache: dict[str, Transolver] = {}


def get(path: str) -> Transolver:
    if path not in model_cache:
        model_cache[path] = load(path)
    return model_cache[path]


with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output: {output_dir}")

for split, ckpts in ROUTES.items():
    test_dir = splits_dir / split
    test_files = sorted(test_dir.glob("*.pt"))
    print(f"\n{split} (ensemble of {len(ckpts)} models): {len(test_files)} samples")

    models = [get(c) for c in ckpts]
    weights = [1.0 / len(models)] * len(models)

    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_files), cfg.batch_size), desc=split, leave=False):
            batch_files = test_files[i:i + cfg.batch_size]
            samples = [torch.load(f, weights_only=True) for f in batch_files]
            xs = [s["x"] for s in samples]

            max_n = max(x.shape[0] for x in xs)
            B = len(xs)
            x_pad = torch.zeros(B, max_n, X_DIM, device=device)
            for j, x in enumerate(xs):
                x_pad[j, :x.shape[0]] = x.to(device)

            x_norm = (x_pad - x_mean) / x_std

            pred_avg = None
            for m, w in zip(models, weights):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pn = m({"x": x_norm})["preds"].float()
                pp = pn * y_std + y_mean
                pred_avg = pp * w if pred_avg is None else pred_avg + pp * w

            for j, x in enumerate(xs):
                predictions.append(pred_avg[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nRouter predictions saved to {output_dir}")
