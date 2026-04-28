"""Generate predictions on the hidden test splits.

Loads the trained Transolver model and writes per-split predictions.

Output layout:
  /mnt/new-pvc/predictions/<RESEARCH_TAG>/<agent>/<commit>/
  ├── test_single_in_dist.pt
  ├── test_geom_camber_rc.pt
  ├── test_geom_camber_cruise.pt
  └── test_re_rand.pt
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

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 2
    use_amp: bool = True
    re_norm_k: float = 0.0  # match training; if >0 scales pressure prediction by Re factor
    re_ref_log: float = 14.58


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

# Load config alongside checkpoint if present, else use defaults
ckpt_path = Path(cfg.checkpoint)
config_path = ckpt_path.parent / "config.yaml"
if config_path.exists():
    with open(config_path) as f:
        model_config = yaml.safe_load(f)
else:
    model_config = dict(
        space_dim=2, fun_dim=X_DIM - 2, out_dim=3, n_hidden=256, n_layers=8,
        n_head=8, slice_num=96, mlp_ratio=2, dropout=0.0,
    )
# Disable gradient checkpointing for inference (speeds up)
model_config["use_checkpoint"] = False

model = Transolver(**model_config).to(device)
state = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print(f"Loaded model from {cfg.checkpoint}")

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

amp_dtype = torch.bfloat16

for split in TEST_SPLITS:
    test_dir = splits_dir / split
    test_files = sorted(test_dir.glob("*.pt"))
    print(f"{split}: {len(test_files)} samples")

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
            if cfg.use_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    pred_norm = model({"x": x_norm})["preds"]
                pred_norm = pred_norm.float()
            else:
                pred_norm = model({"x": x_norm})["preds"]
            pred = pred_norm * y_std + y_mean

            if cfg.re_norm_k > 0:
                # Apply Re factor to pressure (channel 2) — must match training scaling
                log_re_per_sample = torch.zeros(B, device=device)
                for j, x in enumerate(xs):
                    log_re_per_sample[j] = x[:, 13].mean().to(device)
                re_factor = torch.exp(cfg.re_norm_k * (log_re_per_sample - cfg.re_ref_log)).view(B, 1)
                pred = pred.clone()
                pred[..., 2] = pred[..., 2] * re_factor  # broadcasts [B, N] * [B, 1]

            for j, x in enumerate(xs):
                predictions.append(pred[j, :x.shape[0]].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  → {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
