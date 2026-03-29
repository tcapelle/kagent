"""Generate predictions on hidden test samples.

Run:
  python predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import DataLoader

from data import GRAMDataset, T_IN, T_OUT, collate_fn

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

TEST_SPLITS = ["val"]


# ---------------------------------------------------------------------------
# Model definition (duplicated from train.py to avoid import side effects)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class AirflowPredictor(nn.Module):
    def __init__(self, hidden=512, n_blocks=8, vel_mean=None, vel_std=None):
        super().__init__()
        in_dim = 3 + T_IN * 3 + T_IN
        out_dim = T_OUT * 3

        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std

        t_in = t[:, :T_IN]
        t_range = t_in[:, -1:] - t_in[:, :1] + 1e-6
        t_features = (t_in - t_in[:, :1]) / t_range
        t_features = t_features.unsqueeze(1).expand(B, N, T_IN)

        v_flat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)
        x = torch.cat([pos, v_flat, t_features], dim=-1)

        x = self.proj_in(x)
        x = self.blocks(x)
        delta_norm = self.proj_out(x)
        delta_norm = delta_norm.reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)

        last_in_norm = v_in_norm[:, -1:, :, :]
        pred_norm = last_in_norm + delta_norm

        pred = pred_norm * self.vel_std + self.vel_mean

        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                pred[i, :, idcs_airfoil[i], :] = 0.0

        return pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

with open(splits_dir / "stats.json") as f:
    stats_raw = json.load(f)
vel_mean = torch.tensor(stats_raw["vel_mean"], dtype=torch.float32)
vel_std = torch.tensor(stats_raw["vel_std"], dtype=torch.float32)

model = AirflowPredictor(hidden=512, n_blocks=8, vel_mean=vel_mean, vel_std=vel_std).to(device)
model.load_state_dict(torch.load(cfg.checkpoint, map_location=device, weights_only=True))

model.eval()
print(f"Loaded model from {cfg.checkpoint}")

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)

for split in TEST_SPLITS:
    ds = GRAMDataset(splits_dir / split)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{split}: {len(ds)} samples")

    predictions = []
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in tqdm(loader, desc=split, leave=False):
            v_in = v_in.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                pred = model(v_in, pos, t, idcs)

            for j in range(pred.shape[0]):
                predictions.append(pred[j].float().cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
