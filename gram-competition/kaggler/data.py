"""Data loading for GRAM airflow prediction competition.

Loads pre-processed .pt samples from splits directory on PVC.
Each sample: velocity_in [5, N, 3], velocity_out [5, N, 3], pos [N, 3], t [10], idcs_airfoil.
"""

import json

import torch
from pathlib import Path
from torch.utils.data import Dataset

N_POINTS = 100_000
T_IN = 5   # input time steps
T_OUT = 5  # output time steps

SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")
VAL_SPLIT_NAMES = ["val"]


class GRAMDataset(Dataset):
    """Dataset of airflow velocity field samples."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = torch.load(self.files[idx], weights_only=True)
        return s["velocity_in"], s["velocity_out"], s["pos"], s["t"], s["idcs_airfoil"]


def collate_fn(batch):
    """Collate GRAM samples into batched tensors.

    All samples have N_POINTS=100k points, so no padding needed.
    idcs_airfoil varies per sample — kept as a list.

    Returns: velocity_in [B,5,N,3], velocity_out [B,5,N,3], pos [B,N,3], t [B,10], idcs_airfoil list[tensor]
    """
    v_in, v_out, pos, t, idcs = zip(*batch)
    return (
        torch.stack(v_in),
        torch.stack(v_out),
        torch.stack(pos),
        torch.stack(t),
        list(idcs),
    )


def load_data(
    splits_dir: str | Path = SPLITS_DIR,
    debug: bool = False,
) -> tuple[GRAMDataset, dict[str, GRAMDataset], dict[str, torch.Tensor]]:
    """Load competition data from pre-processed splits.

    Returns:
        train_ds:   GRAMDataset for training
        val_splits: {"val": GRAMDataset}
        stats:      normalization stats (velocity mean/std)
    """
    splits_dir = Path(splits_dir)

    with open(splits_dir / "stats.json") as f:
        stats_raw = json.load(f)

    train_ds = GRAMDataset(splits_dir / "train")
    val_splits = {name: GRAMDataset(splits_dir / name) for name in VAL_SPLIT_NAMES}

    if debug:
        train_ds.files = train_ds.files[:4]
        for ds in val_splits.values():
            ds.files = ds.files[:2]

    stats = {
        "vel_mean": torch.tensor(stats_raw["vel_mean"], dtype=torch.float32),
        "vel_std": torch.tensor(stats_raw["vel_std"], dtype=torch.float32),
    }

    print(
        f"Train: {len(train_ds)}, "
        + ", ".join(f"{k}: {len(v)}" for k, v in val_splits.items())
    )
    return train_ds, val_splits, stats
