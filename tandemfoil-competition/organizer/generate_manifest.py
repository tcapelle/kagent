"""One-time: generate the split manifest and check it into the repo.

Run locally or on a pod with access to the raw pickle files:
  python generate_manifest.py

Writes: cfd-competition/organizer/split_manifest.json
This file is checked into git and is the single source of truth for all splits.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch

SEED = 42
N_PER_VAL = 100
N_PER_TEST = 200
DATA_ROOT = Path("/mnt/new-pvc/datasets/tandemfoil")

PICKLE_FILES = [
    "raceCar_single_randomFields.pickle",         # 0
    "raceCar_randomFields_mgn_Part1.pickle",       # 1
    "raceCar_randomFields_mgn_Part2.pickle",       # 2
    "raceCar_randomFields_mgn_Part3.pickle",       # 3
    "cruise_randomFields_mgn_Part1.pickle",        # 4
    "cruise_randomFields_mgn_Part2.pickle",        # 5
    "cruise_randomFields_mgn_Part3.pickle",        # 6
]

VAL_SPLITS = [
    "val_single_in_dist",
    "val_geom_camber_rc",
    "val_geom_camber_cruise",
    "val_re_rand",
]
TEST_SPLITS = [s.replace("val_", "test_") for s in VAL_SPLITS]


def load_pickle(path: Path) -> list:
    return torch.load(path, map_location="cpu", weights_only=False)


def scan_metadata(pickle_paths: list[Path]):
    by_file: dict[int, list[dict]] = {}
    file_sizes: list[int] = []
    offset = 0
    for fi, path in enumerate(pickle_paths):
        raw = load_pickle(path)
        n = len(raw)
        file_sizes.append(n)
        by_file[fi] = []
        for li, sample in enumerate(raw):
            aoa = sample.AoA
            by_file[fi].append({
                "global_idx": offset + li,
                "re": float(sample.flowState["Re"]),
                "aoa0": float(aoa[0]) if isinstance(aoa, list) else float(aoa),
                "gap": float(sample.gap) if getattr(sample, "gap", None) is not None else None,
                "stagger": float(sample.stagger) if getattr(sample, "stagger", None) is not None else None,
            })
        print(f"  [{fi}] {path.name} → {n} samples")
        offset += n
        del raw
    return by_file, file_sizes


def assign_splits(by_file: dict[int, list[dict]]):
    rng = np.random.default_rng(SEED)

    all_split_names = ["train"] + VAL_SPLITS + TEST_SPLITS
    splits: dict[str, list[int]] = {k: [] for k in all_split_names}
    groups: dict[str, list[int]] = {
        "racecar_single": [], "racecar_tandem": [], "cruise": [],
    }

    def split_val_test(idxs: list[int], val_name: str, test_name: str) -> list[int]:
        arr = np.array(idxs)
        rng.shuffle(arr)
        splits[val_name].extend(arr[:N_PER_VAL].tolist())
        splits[test_name].extend(arr[N_PER_VAL:N_PER_VAL + N_PER_TEST].tolist())
        return arr[N_PER_VAL + N_PER_TEST:].tolist()

    # File 0: single foil — random holdout
    f0_idxs = [r["global_idx"] for r in by_file[0]]
    f0_rest = split_val_test(f0_idxs, "val_single_in_dist", "test_single_in_dist")
    splits["train"].extend(f0_rest)
    groups["racecar_single"].extend(f0_rest)

    # File 2: full geometry holdout (raceCar, M=6-8)
    f2_idxs = [r["global_idx"] for r in by_file[2]]
    f2_rest = split_val_test(f2_idxs, "val_geom_camber_rc", "test_geom_camber_rc")
    assert len(f2_rest) == 0

    # File 5: full geometry holdout (cruise, M=2-4)
    f5_idxs = [r["global_idx"] for r in by_file[5]]
    f5_rest = split_val_test(f5_idxs, "val_geom_camber_cruise", "test_geom_camber_cruise")
    assert len(f5_rest) == 0

    # Files 1,3,4,6: stratified Re holdout
    tandem_pool = []
    for fi in (1, 3, 4, 6):
        for r in by_file[fi]:
            tandem_pool.append(r)

    tandem_pool.sort(key=lambda r: r["re"])
    holdout_indices = []
    train_indices = []
    for i, r in enumerate(tandem_pool):
        if i % 4 == 0:
            holdout_indices.append(r["global_idx"])
        else:
            train_indices.append(r["global_idx"])

    assert len(holdout_indices) == N_PER_VAL + N_PER_TEST

    holdout_arr = np.array(holdout_indices)
    rng.shuffle(holdout_arr)
    splits["val_re_rand"].extend(holdout_arr[:N_PER_VAL].tolist())
    splits["test_re_rand"].extend(holdout_arr[N_PER_VAL:].tolist())

    file_offsets = [0]
    for fi in range(len(by_file)):
        file_offsets.append(file_offsets[-1] + len(by_file[fi]))

    for gidx in train_indices:
        splits["train"].append(gidx)
        if file_offsets[1] <= gidx < file_offsets[4]:
            groups["racecar_tandem"].append(gidx)
        else:
            groups["cruise"].append(gidx)

    return splits, groups


# --- Main ---

pickle_paths = [DATA_ROOT / f for f in PICKLE_FILES]

print("Scanning metadata...")
by_file, file_sizes = scan_metadata(pickle_paths)

print("Assigning splits...")
splits, domain_groups = assign_splits(by_file)

for k, v in splits.items():
    print(f"  {k:30s} {len(v):5d}")
total = sum(len(v) for v in splits.values())
print(f"  {'TOTAL':30s} {total:5d} / {sum(file_sizes)}")
assert total == sum(file_sizes)

# Build train-local domain groups (sequential index within train split)
train_gidx_to_seq = {gidx: i for i, gidx in enumerate(splits["train"])}

manifest = {
    "version": 2,
    "seed": SEED,
    "n_per_val": N_PER_VAL,
    "n_per_test": N_PER_TEST,
    "pickle_files": PICKLE_FILES,
    "file_sizes": file_sizes,
    "val_splits": VAL_SPLITS,
    "test_splits": TEST_SPLITS,
    "split_counts": {k: len(v) for k, v in splits.items()},
    "splits": {k: sorted(v) for k, v in splits.items()},
    "domain_groups": {
        name: sorted(train_gidx_to_seq[gidx] for gidx in idxs)
        for name, idxs in domain_groups.items()
    },
}

out_path = Path(__file__).parent / "split_manifest.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nWrote {out_path}")
print("Check this file into git. prepare_splits.py reads it to materialize .pt files.")
