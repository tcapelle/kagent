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
from tqdm import tqdm

from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")

TEST_SPLITS = ["val"]


@dataclass
class Config:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str  # primary checkpoint
    # Optional extra checkpoints for ensembling (predictions averaged).
    checkpoint2: str | None = None
    checkpoint3: str | None = None
    checkpoint4: str | None = None
    checkpoint5: str | None = None
    checkpoint6: str | None = None
    checkpoint7: str | None = None
    checkpoint8: str | None = None
    checkpoint9: str | None = None
    checkpoint10: str | None = None
    checkpoint11: str | None = None
    checkpoint12: str | None = None
    checkpoint13: str | None = None
    checkpoint14: str | None = None
    checkpoint15: str | None = None
    checkpoint16: str | None = None
    # Comma-separated weights matching the number of checkpoints (default: equal).
    weights: str | None = None
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1
    # Average predictions with y-flipped input (wing is y-symmetric).
    yflip_tta: bool = False


cfg = sp.parse(Config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
splits_dir = Path(cfg.splits_dir)

with open(Path(cfg.splits_dir) / "stats.json") as _f:
    _stats_raw = json.load(_f)
_vel_mean = torch.tensor(_stats_raw["vel_mean"], dtype=torch.float32)
_vel_std = torch.tensor(_stats_raw["vel_std"], dtype=torch.float32)

from train import TransolverModel, VoxelUNetModel, compute_dist_to_airfoil


def _make_model(ckpt_path):
    """Auto-detect model type and architecture from checkpoint state_dict."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "unet.enc1.block.0.weight" in sd:
        ch_base = sd["unet.enc1.block.0.weight"].shape[0]
        if "unet._grid_buf" in sd:
            grid = tuple(sd["unet._grid_buf"].tolist())
        else:
            grid = (64, 32, 32)  # legacy default
        m = VoxelUNetModel(
            hidden=256, num_pre=2, num_post=4,
            grid=grid, ch_base=ch_base,
            num_pos_freqs=10, num_vel_freqs=3, num_dist_freqs=6,
            vel_mean=_vel_mean, vel_std=_vel_std,
        ).to(device)
        m.load_state_dict(sd, strict=False)
        print(f"Loaded UNet (ch={ch_base}, grid={grid}) from {ckpt_path}")
    else:
        m = TransolverModel(
            hidden=256, n_blocks=6, heads=8, slices=64,
            num_pos_freqs=10, num_vel_freqs=3, num_dist_freqs=6,
            vel_mean=_vel_mean, vel_std=_vel_std,
        ).to(device)
        m.load_state_dict(sd, strict=False)
        print(f"Loaded Transolver from {ckpt_path}")
    m.eval()
    return m


_ckpts = [cfg.checkpoint]
for c in [cfg.checkpoint2, cfg.checkpoint3, cfg.checkpoint4, cfg.checkpoint5,
          cfg.checkpoint6, cfg.checkpoint7, cfg.checkpoint8, cfg.checkpoint9,
          cfg.checkpoint10, cfg.checkpoint11, cfg.checkpoint12,
          cfg.checkpoint13, cfg.checkpoint14, cfg.checkpoint15, cfg.checkpoint16]:
    if c:
        _ckpts.append(c)

models = [_make_model(p) for p in _ckpts]
if cfg.weights:
    weights = [float(s) for s in cfg.weights.split(",")]
    assert len(weights) == len(models), "weights count must match checkpoints"
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    weights = weights / weights.sum()
else:
    weights = torch.full((len(models),), 1.0 / len(models), device=device)
print(f"Ensemble: {len(models)} models, weights={weights.tolist()}, TTA={cfg.yflip_tta}")

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

            dist = compute_dist_to_airfoil(pos, idcs)

            pred_sum = None
            for m, w in zip(models, weights):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    pred = m(v_in, pos, t, idcs, dist).float()
                if cfg.yflip_tta:
                    pos_f = pos.clone(); pos_f[..., 1] = -pos_f[..., 1]
                    v_in_f = v_in.clone(); v_in_f[..., 1] = -v_in_f[..., 1]
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        pred_f = m(v_in_f, pos_f, t, idcs, dist).float()
                    pred_f[..., 1] = -pred_f[..., 1]
                    pred = 0.5 * (pred + pred_f)
                contrib = w * pred
                pred_sum = contrib if pred_sum is None else pred_sum + contrib

            for j in range(pred_sum.shape[0]):
                predictions.append(pred_sum[j].cpu())

    output_path = output_dir / f"{split}.pt"
    torch.save(predictions, output_path)
    print(f"  -> {output_path} ({len(predictions)} samples)")

print(f"\nAll predictions saved to {output_dir}")
