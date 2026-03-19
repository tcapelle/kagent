"""Generate predictions on the hidden test set.

Run:
  uv run predict.py --checkpoint models/model-<id>/checkpoint.pt --agent <your-name>
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
from tqdm import tqdm

from data import X_DIM

PREDICTIONS_DIR = Path("/mnt/new-pvc/predictions")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits")


class MultiscaleFourierFeatures(nn.Module):
    def __init__(self, in_dim, n_freq=66, scales=(1.0, 5.0, 25.0)):
        super().__init__()
        freqs_per_scale = n_freq // len(scales)
        parts = []
        for s in scales:
            parts.append(torch.randn(in_dim, freqs_per_scale) * s)
        self.register_buffer("B", torch.cat(parts, dim=1))

    def forward(self, x):
        proj = x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class FiLMResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, x, cond):
        h = self.norm(x)
        gamma, beta = self.film(cond).unsqueeze(1).chunk(2, dim=-1)
        h = gamma * h + beta
        h = self.linear2(torch.nn.functional.gelu(self.linear1(h)))
        return x + h


class CFDModel(nn.Module):
    def __init__(self, in_dim=24, out_dim=3, hidden=768, n_blocks=6,
                 n_fourier=66, cond_dim=192):
        super().__init__()
        self.local_dim = 13
        self.global_dim = in_dim - 13

        self.fourier_pos = MultiscaleFourierFeatures(2, n_fourier, scales=(1.0, 5.0, 25.0))
        self.fourier_geom = MultiscaleFourierFeatures(10, n_fourier, scales=(1.0, 5.0, 25.0))
        n_fourier_actual = (n_fourier // 3) * 3
        proj_in_dim = self.local_dim + 2 * n_fourier_actual * 2
        self.proj_in = nn.Linear(proj_in_dim, hidden)

        self.cond_enc = nn.Sequential(
            nn.Linear(self.global_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList([FiLMResBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.head_vel = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.head_p = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, data, **kwargs):
        x = data["x"]
        local = x[:, :, :self.local_dim]
        global_feat = x[:, 0, self.local_dim:]
        ff_pos = self.fourier_pos(x[:, :, :2])
        ff_geom = self.fourier_geom(x[:, :, 2:12])
        h = torch.cat([local, ff_pos, ff_geom], dim=-1)
        h = self.proj_in(h)
        cond = self.cond_enc(global_feat)
        for block in self.blocks:
            h = block(h, cond)
        vel = self.head_vel(h)
        p = self.head_p(h)
        return {"preds": torch.cat([vel, p], dim=-1)}


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 4
    # Model config - must match training
    hidden: int = 768
    n_blocks: int = 6
    n_fourier: int = 132
    cond_dim: int = 192


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

model = CFDModel(in_dim=X_DIM, out_dim=3, hidden=cfg.hidden, n_blocks=cfg.n_blocks,
                 n_fourier=cfg.n_fourier, cond_dim=cfg.cond_dim).to(device)
state = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
model.load_state_dict(state)

model.eval()
print(f"Loaded model from {cfg.checkpoint}")

with open(splits_dir / "stats.json") as f:
    stats_data = json.load(f)
x_mean = torch.tensor(stats_data["x_mean"], dtype=torch.float32, device=device)
x_std = torch.tensor(stats_data["x_std"], dtype=torch.float32, device=device)
y_mean = torch.tensor(stats_data["y_mean"], dtype=torch.float32, device=device)
y_std = torch.tensor(stats_data["y_std"], dtype=torch.float32, device=device)

test_files = sorted((splits_dir / "test").glob("*.pt"))
print(f"Test samples: {len(test_files)}")

predictions = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_files), cfg.batch_size), desc="Predicting"):
        batch_files = test_files[i:i + cfg.batch_size]
        samples = [torch.load(f, weights_only=True) for f in batch_files]
        xs = [s["x"] for s in samples]

        max_n = max(x.shape[0] for x in xs)
        B = len(xs)
        x_pad = torch.zeros(B, max_n, X_DIM, device=device)
        for j, x in enumerate(xs):
            x_pad[j, :x.shape[0]] = x.to(device)

        pred_norm = model({"x": (x_pad - x_mean) / x_std})["preds"]
        pred = pred_norm * y_std + y_mean

        for j, x in enumerate(xs):
            predictions.append(pred[j, :x.shape[0]].cpu())

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"

output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "predictions.pt"
torch.save(predictions, output_path)
print(f"Saved {len(predictions)} predictions to {output_path}")
