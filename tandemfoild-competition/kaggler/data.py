"""Data loading for pre-processed competition splits.

Loads individual .pt sample files from the splits directory on PVC.
Kagglers use load_data() to get ready-to-train datasets.
"""

import json

import torch
from pathlib import Path
from torch.utils.data import Dataset

X_DIM = 24
VAL_SPLIT_NAMES = [
    "val_single_in_dist",
    "val_geom_camber_rc",
    "val_geom_camber_cruise",
    "val_re_rand",
]
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")


class SplitDataset(Dataset):
    """Dataset backed by individual .pt files in a directory."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.files = sorted(self.directory.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        s = torch.load(self.files[idx], weights_only=True)
        return s["x"], s["y"], s["is_surface"]


def pad_collate(batch):
    """Collate variable-length mesh samples into padded batches.

    Returns (x, y, is_surface, mask), each [B, N_max, ...].
    """
    xs, ys, surfs = zip(*batch)
    max_n = max(x.shape[0] for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_n, xs[0].shape[1])
    y_pad = torch.zeros(B, max_n, ys[0].shape[1])
    surf_pad = torch.zeros(B, max_n, dtype=torch.bool)
    mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, (x, y, sf) in enumerate(zip(xs, ys, surfs)):
        n = x.shape[0]
        x_pad[i, :n] = x
        y_pad[i, :n] = y
        surf_pad[i, :n] = sf
        mask[i, :n] = True
    return x_pad, y_pad, surf_pad, mask


def load_data(
    splits_dir: str | Path = SPLITS_DIR,
    debug: bool = False,
) -> tuple[SplitDataset, dict[str, SplitDataset], dict[str, torch.Tensor], torch.Tensor]:
    """Load competition data from pre-processed splits.

    Returns:
        train_ds:       SplitDataset for training
        val_splits:     {name: SplitDataset} for each validation track
        stats:          {x_mean, x_std, y_mean, y_std} as CPU float32 tensors
        sample_weights: per-sample weights for balanced domain sampling
    """
    splits_dir = Path(splits_dir)

    with open(splits_dir / "stats.json") as f:
        stats_raw = json.load(f)
    with open(splits_dir / "meta.json") as f:
        meta = json.load(f)

    train_ds = SplitDataset(splits_dir / "train")
    val_splits = {name: SplitDataset(splits_dir / name) for name in VAL_SPLIT_NAMES}

    if debug:
        train_ds.files = train_ds.files[:6]
        for ds in val_splits.values():
            ds.files = ds.files[:2]

    stats = {
        "x_mean": torch.tensor(stats_raw["x_mean"], dtype=torch.float32),
        "x_std": torch.tensor(stats_raw["x_std"], dtype=torch.float32),
        "y_mean": torch.tensor(stats_raw["y_mean"], dtype=torch.float32),
        "y_std": torch.tensor(stats_raw["y_std"], dtype=torch.float32),
    }

    # Balanced domain sampler weights
    domain_groups = meta["domain_groups"]
    group_sizes = {name: len(idxs) for name, idxs in domain_groups.items()}
    idx_to_group: dict[int, str] = {}
    for name, idxs in domain_groups.items():
        for i in idxs:
            idx_to_group[i] = name

    sample_weights = torch.tensor(
        [1.0 / group_sizes[idx_to_group[i]] for i in range(len(train_ds))],
        dtype=torch.float64,
    )

    print(
        f"Train: {len(train_ds)}, "
        + ", ".join(f"{k}: {len(v)}" for k, v in val_splits.items())
    )
    return train_ds, val_splits, stats, sample_weights
