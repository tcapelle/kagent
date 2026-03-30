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
# Utilities
# ---------------------------------------------------------------------------

def chunked_knn(pos, k, chunk_size=5000):
    """Compute k-NN indices for point cloud. pos: [B, N, 3] -> [B, N, k]"""
    B, N, _ = pos.shape
    device = pos.device
    knn_idx = torch.empty(B, N, k, dtype=torch.long, device=device)
    pos_f = pos.float()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dists = torch.cdist(pos_f[:, start:end], pos_f)
        local_idx = torch.arange(end - start, device=device)
        dists[:, local_idx, local_idx + start] = float('inf')
        knn_idx[:, start:end] = dists.topk(k, dim=-1, largest=False).indices
    return knn_idx


# ---------------------------------------------------------------------------
# Model (must match train.py)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class EdgeConvBlock(nn.Module):
    """DGCNN-style edge convolution for spatial interaction."""

    def __init__(self, dim, k=20):
        super().__init__()
        self.k = k
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos, knn_idx):
        B, N, D = x.shape
        k = knn_idx.shape[-1]

        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        x_j = x[batch_idx, knn_idx]
        pos_j = pos[batch_idx, knn_idx]

        diff = x_j - x.unsqueeze(2)
        rel_pos = pos_j - pos.unsqueeze(2)
        edge_feat = torch.cat([diff, rel_pos], dim=-1)

        edge_out = self.edge_mlp(edge_feat)
        agg = edge_out.mean(dim=2)

        return self.norm(x + agg)


class AirflowPredictor(nn.Module):
    def __init__(self, hidden=512, n_blocks=9, n_fourier=64, dropout=0.05,
                 k_neighbors=16, vel_mean=None, vel_std=None):
        super().__init__()
        self.n_fourier = n_fourier
        self.k_neighbors = k_neighbors

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=T_IN, padding=0),
            nn.GELU(),
        )
        temporal_feat_dim = 32

        pos_feat_dim = n_fourier * 2 * 3
        vel_feat_dim = T_IN * 3
        vel_mag_dim = T_IN
        deriv_feat_dim = (T_IN - 1) * 3
        time_feat_dim = T_IN
        surface_dim = 1
        in_dim = pos_feat_dim + temporal_feat_dim + vel_feat_dim + vel_mag_dim + deriv_feat_dim + time_feat_dim + surface_dim

        out_dim = T_OUT * 3

        freqs = torch.cat([
            torch.randn(3, n_fourier // 4) * 0.5,
            torch.randn(3, n_fourier // 2) * 2.0,
            torch.randn(3, n_fourier - n_fourier // 4 - n_fourier // 2) * 8.0,
        ], dim=1)
        self.fourier_freqs = nn.Parameter(freqs)

        self.proj_in = nn.Linear(in_dim, hidden)

        third = n_blocks // 3
        self.blocks_1 = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(third)])
        self.edge_conv_1 = EdgeConvBlock(hidden, k=k_neighbors)
        self.blocks_2 = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(third)])
        self.edge_conv_2 = EdgeConvBlock(hidden, k=k_neighbors)
        self.blocks_3 = nn.Sequential(*[ResBlock(hidden, dropout=dropout) for _ in range(n_blocks - 2 * third)])

        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

        if vel_mean is not None:
            self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
            self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))
        else:
            self.register_buffer("vel_mean", torch.zeros(1, 1, 1, 3))
            self.register_buffer("vel_std", torch.ones(1, 1, 1, 3))

    def _fourier_encode(self, pos):
        B, N, _ = pos.shape
        feats = []
        for i in range(3):
            p = pos[:, :, i:i+1] * self.fourier_freqs[i:i+1, :]
            feats.append(torch.sin(p))
            feats.append(torch.cos(p))
        return torch.cat(feats, dim=-1)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape

        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std
        pos_feat = self._fourier_encode(pos)

        v_temp = v_in_norm.permute(0, 2, 3, 1).reshape(B * N, C, T)
        v_temp = self.temporal_conv(v_temp).squeeze(-1)
        v_temp = v_temp.reshape(B, N, -1)

        v_flat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)

        v_mag = v_in_norm.norm(dim=3)
        v_mag = v_mag.permute(0, 2, 1)

        v_deriv = v_in_norm[:, 1:] - v_in_norm[:, :-1]
        v_deriv_flat = v_deriv.permute(0, 2, 1, 3).reshape(B, N, (T-1) * C)

        t_in = t[:, :T_IN]
        t_range = t_in[:, -1:] - t_in[:, :1] + 1e-6
        t_features = (t_in - t_in[:, :1]) / t_range
        t_features = t_features.unsqueeze(1).expand(B, N, T_IN)

        surface_flag = torch.zeros(B, N, 1, device=pos.device, dtype=pos.dtype)
        for i in range(B):
            if idcs_airfoil[i] is not None and len(idcs_airfoil[i]) > 0:
                surface_flag[i, idcs_airfoil[i], 0] = 1.0

        x = torch.cat([pos_feat, v_temp, v_flat, v_mag, v_deriv_flat, t_features, surface_flag], dim=-1)

        with torch.amp.autocast("cuda", enabled=False):
            knn_idx = chunked_knn(pos.float(), self.k_neighbors)

        x = self.proj_in(x)
        x = self.blocks_1(x)
        x = self.edge_conv_1(x, pos, knn_idx)
        x = self.blocks_2(x)
        x = self.edge_conv_2(x, pos, knn_idx)
        x = self.blocks_3(x)
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
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1
    hidden: int = 512
    n_blocks: int = 9
    n_fourier: int = 64
    k_neighbors: int = 12


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

with open(splits_dir / "stats.json") as f:
    stats_raw = json.load(f)
vel_mean = torch.tensor(stats_raw["vel_mean"], dtype=torch.float32)
vel_std = torch.tensor(stats_raw["vel_std"], dtype=torch.float32)

model = AirflowPredictor(
    hidden=cfg.hidden, n_blocks=cfg.n_blocks, n_fourier=cfg.n_fourier,
    k_neighbors=cfg.k_neighbors,
    vel_mean=vel_mean, vel_std=vel_std,
).to(device)
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
