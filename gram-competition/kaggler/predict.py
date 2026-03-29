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
class PredictConfig:
    """Generate test predictions from a trained checkpoint."""
    checkpoint: str
    splits_dir: str = str(SPLITS_DIR)
    agent: str | None = None
    batch_size: int = 1
    hidden: int = 256
    n_heads: int = 8
    n_transformer_blocks: int = 4
    n_mlp_blocks: int = 2
    pred_subsample: int = 16384  # subsample for prediction (transformer can't do 100k)
    n_ensembles: int = 4  # average over multiple random subsamples


def main():
    cfg = sp.parse(PredictConfig)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_dir = Path(cfg.splits_dir)

    with open(splits_dir / "stats.json") as f:
        stats_raw = json.load(f)

    vel_mean = torch.tensor(stats_raw["vel_mean"], dtype=torch.float32).to(device)
    vel_std = torch.tensor(stats_raw["vel_std"], dtype=torch.float32).to(device)

    from train import AirflowTransformer
    model = AirflowTransformer(
        hidden=cfg.hidden, n_heads=cfg.n_heads,
        n_transformer_blocks=cfg.n_transformer_blocks,
        n_mlp_blocks=cfg.n_mlp_blocks,
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
        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
        print(f"{split}: {len(ds)} samples")

        predictions = []
        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in tqdm(loader, desc=split, leave=False):
                v_in = v_in.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                N = pos.shape[1]

                if N <= cfg.pred_subsample:
                    # Can do full pass
                    with torch.cuda.amp.autocast():
                        pred = model(v_in, pos, t, idcs)
                    pred = pred.float()
                else:
                    # Ensemble of random subsamples + scatter back
                    pred_sum = torch.zeros(1, 5, N, 3, device=device)
                    pred_count = torch.zeros(1, 1, N, 1, device=device)

                    for _ in range(cfg.n_ensembles):
                        idx = torch.randperm(N, device=device)[:cfg.pred_subsample].sort().values

                        v_in_s = v_in[:, :, idx, :]
                        pos_s = pos[:, idx, :]

                        # Remap airfoil indices
                        idcs_s = []
                        for i_b in range(v_in.shape[0]):
                            if idcs[i_b] is not None and len(idcs[i_b]) > 0:
                                mask = torch.isin(idx, idcs[i_b].to(device))
                                idcs_s.append(torch.where(mask)[0])
                            else:
                                idcs_s.append(torch.tensor([], dtype=torch.long, device=device))

                        with torch.cuda.amp.autocast():
                            pred_s = model(v_in_s, pos_s, t, idcs_s).float()

                        pred_sum[:, :, idx, :] += pred_s
                        pred_count[:, :, idx, :] += 1

                    # For points not covered by any subsample, use copy-last
                    uncovered = (pred_count.squeeze() == 0)
                    pred_count = pred_count.clamp(min=1)
                    pred = pred_sum / pred_count

                    if uncovered.any():
                        last_vel = v_in[:, -1:, :, :].expand_as(pred)
                        pred = torch.where(uncovered.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand_as(pred),
                                          last_vel, pred)

                for j in range(pred.shape[0]):
                    predictions.append(pred[j].cpu())

        output_path = output_dir / f"{split}.pt"
        torch.save(predictions, output_path)
        print(f"  -> {output_path} ({len(predictions)} samples)")

    print(f"\nAll predictions saved to {output_dir}")


if __name__ == "__main__":
    main()
