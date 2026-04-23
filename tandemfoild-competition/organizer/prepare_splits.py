"""Materialize pre-processed .pt files from a checked-in split manifest.

Reads split_manifest.json (checked into git) and writes per-sample .pt files
to the PVC. No split logic here — the manifest is the single source of truth.

Run on organizer pod:
  python prepare_splits.py

Output on PVC:
  /mnt/new-pvc/datasets/tandemfoil/splits_v2/
  ├── train/000000.pt ...                  {x, y, is_surface}
  ├── val_single_in_dist/...               {x, y, is_surface}
  ├── val_geom_camber_rc/...               {x, y, is_surface}
  ├── val_geom_camber_cruise/...           {x, y, is_surface}
  ├── val_re_rand/...                      {x, y, is_surface}
  ├── test_single_in_dist/...              {x, is_surface}  (no y)
  ├── test_geom_camber_rc/...              {x, is_surface}
  ├── test_geom_camber_cruise/...          {x, is_surface}
  ├── test_re_rand/...                     {x, is_surface}
  ├── .test_*_gt/                          {y, is_surface}  (hidden ground truth)
  ├── stats.json
  └── meta.json
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import simple_parsing as sp
import torch
from rich.console import Console
from rich.panel import Panel

console = Console()

# --- Constants ---
SURFACE_IDS = (5, 6, 7)
X_DIM = 24
DATA_ROOT = Path("/mnt/new-pvc/datasets/tandemfoil")
MANIFEST_PATH = Path(__file__).parent / "split_manifest.json"


@dataclass
class Args:
    """Materialize TandemFoilSet splits from manifest."""
    data_root: str = str(DATA_ROOT)
    out_dir: str = str(DATA_ROOT / "splits_v2")
    manifest: str = str(MANIFEST_PATH)


# --- Raw data helpers ---

def load_pickle(path: Path) -> list:
    return torch.load(path, map_location="cpu", weights_only=False)


def parse_naca(s: str) -> tuple[float, float, float]:
    if len(s) == 4 and s.isdigit():
        return int(s[0]) / 9.0, int(s[1]) / 9.0, int(s[2:]) / 24.0
    return 0.0, 0.0, 0.0


def preprocess(sample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw PyG sample → (x [N,24], y [N,3], is_surface [N]).

    x layout: [pos(2), saf(2), dsdf(8), is_surface(1), log_Re(1),
               AoA0_rad(1), NACA0(3), AoA1_rad(1), NACA1(3), gap(1), stagger(1)]
    """
    n = sample.pos.shape[0]

    is_surface = torch.zeros(n, dtype=torch.bool)
    for sid in SURFACE_IDS:
        is_surface |= sample.boundary == sid

    aoa = sample.AoA
    aoa0 = float(aoa[0]) if isinstance(aoa, list) else float(aoa)
    aoa1 = float(aoa[1]) if isinstance(aoa, list) else 0.0

    naca0 = parse_naca(sample.NACA[0])
    naca1 = parse_naca(sample.NACA[1]) if len(sample.NACA) > 1 else (0.0, 0.0, 0.0)

    gap_val = getattr(sample, "gap", None)
    stagger_val = getattr(sample, "stagger", None)

    x = torch.cat([
        sample.pos.float(),                                             # 2
        sample.saf.float(),                                             # 2
        sample.dsdf.float(),                                            # 8
        is_surface.float().unsqueeze(1),                                # 1
        torch.full((n, 1), math.log(float(sample.flowState["Re"]))),    # 1
        torch.full((n, 1), aoa0 * math.pi / 180.0),                    # 1
        torch.tensor(naca0, dtype=torch.float32).expand(n, 3),          # 3
        torch.full((n, 1), aoa1 * math.pi / 180.0),                    # 1
        torch.tensor(naca1, dtype=torch.float32).expand(n, 3),          # 3
        torch.full((n, 1), float(gap_val) if gap_val is not None else 0.0),        # 1
        torch.full((n, 1), float(stagger_val) if stagger_val is not None else 0.0),  # 1
    ], dim=1)

    return x, sample.y.float(), is_surface


# --- Save helpers ---

def global_to_file_local(global_idx: int, file_sizes: list[int]) -> tuple[int, int]:
    offset = 0
    for fi, n in enumerate(file_sizes):
        if global_idx < offset + n:
            return fi, global_idx - offset
        offset += n
    raise ValueError(f"global_idx {global_idx} out of range")


def save_samples(
    out_dir: Path,
    split_name: str,
    global_indices: list[int],
    pickle_paths: list[Path],
    file_sizes: list[int],
    include_y: bool = True,
):
    """Preprocess raw samples and save as individual .pt files."""
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = None
    if not include_y:
        gt_dir = out_dir / f".{split_name}_gt"
        gt_dir.mkdir(parents=True, exist_ok=True)

    by_file: dict[int, list[tuple[int, int]]] = {}
    for seq_idx, gidx in enumerate(global_indices):
        fi, li = global_to_file_local(gidx, file_sizes)
        by_file.setdefault(fi, []).append((seq_idx, li))

    for fi in sorted(by_file):
        console.print(f"    {pickle_paths[fi].name} ({len(by_file[fi])} samples)")
        raw = load_pickle(pickle_paths[fi])
        for seq_idx, li in by_file[fi]:
            x, y, is_surface = preprocess(raw[li])
            fname = f"{seq_idx:06d}.pt"
            if include_y:
                torch.save({"x": x, "y": y, "is_surface": is_surface}, split_dir / fname)
            else:
                torch.save({"x": x, "is_surface": is_surface}, split_dir / fname)
                torch.save({"y": y, "is_surface": is_surface}, gt_dir / fname)
        del raw

    console.print(f"  {split_name}: {len(global_indices)} samples")


def compute_stats(train_dir: Path) -> dict:
    """Two-pass mean/std over training .pt files."""
    files = sorted(train_dir.glob("*.pt"))
    n = len(files)

    console.print(f"  Pass 1/2 (mean) — {n} samples")
    sum_x = torch.zeros(X_DIM, dtype=torch.float64)
    sum_y = torch.zeros(3, dtype=torch.float64)
    total = 0

    for i, f in enumerate(files):
        if i % 200 == 0:
            console.print(f"    {i}/{n}")
        s = torch.load(f, weights_only=True)
        sum_x += s["x"].double().sum(0)
        sum_y += s["y"].double().sum(0)
        total += s["x"].shape[0]

    mean_x = sum_x / total
    mean_y = sum_y / total

    console.print(f"  Pass 2/2 (std) — {n} samples")
    sq_x = torch.zeros(X_DIM, dtype=torch.float64)
    sq_y = torch.zeros(3, dtype=torch.float64)

    for i, f in enumerate(files):
        if i % 200 == 0:
            console.print(f"    {i}/{n}")
        s = torch.load(f, weights_only=True)
        sq_x += ((s["x"].double() - mean_x) ** 2).sum(0)
        sq_y += ((s["y"].double() - mean_y) ** 2).sum(0)

    std_x = (sq_x / (total - 1)).sqrt().clamp(min=1e-6)
    std_y = (sq_y / (total - 1)).sqrt().clamp(min=1e-6)

    return {
        "x_dim": X_DIM,
        "n_train_samples": n,
        "n_train_nodes": total,
        "x_mean": mean_x.float().tolist(),
        "x_std": std_x.float().tolist(),
        "y_mean": mean_y.float().tolist(),
        "y_std": std_y.float().tolist(),
    }


# --- Main ---

args = sp.parse(Args)
data_root = Path(args.data_root)
out_dir = Path(args.out_dir)

console.rule("Loading manifest")
with open(args.manifest) as f:
    manifest = json.load(f)

pickle_paths = [data_root / f for f in manifest["pickle_files"]]
file_sizes = manifest["file_sizes"]
splits = manifest["splits"]
val_splits = manifest["val_splits"]
test_splits = manifest["test_splits"]

console.print(f"  Manifest version: {manifest['version']}, seed: {manifest['seed']}")
for k, v in manifest["split_counts"].items():
    console.print(f"  {k:30s} {v:5d}")

console.rule("Phase 1: Materialize train + val splits")
for split_name in ["train"] + val_splits:
    save_samples(out_dir, split_name, splits[split_name], pickle_paths, file_sizes)

console.rule("Phase 2: Materialize test splits (no y)")
for test_name in test_splits:
    # Shuffle test order so sequential index doesn't reveal source file
    test_indices = splits[test_name].copy()
    np.random.default_rng(123).shuffle(test_indices)
    save_samples(out_dir, test_name, test_indices, pickle_paths, file_sizes, include_y=False)

console.rule("Phase 3: Normalization stats")
stats = compute_stats(out_dir / "train")
with open(out_dir / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
console.print(f"  Wrote stats.json ({stats['n_train_nodes']} total nodes)")

console.rule("Phase 4: Write meta.json for kagglers")
meta = {
    "x_dim": X_DIM,
    "val_splits": val_splits,
    "test_splits": test_splits,
    "split_counts": manifest["split_counts"],
    "domain_groups": manifest["domain_groups"],
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
console.print("  Wrote meta.json")

n_val = manifest["n_per_val"]
n_test = manifest["n_per_test"]
console.rule("Done")
console.print(Panel(
    f"Output: {out_dir}\n"
    f"Train: {stats['n_train_samples']} samples, {stats['n_train_nodes']} nodes\n"
    f"Val: {4 * n_val} (4 × {n_val}), Test: {4 * n_test} (4 × {n_test})\n"
    f"y_mean: {[f'{v:.2f}' for v in stats['y_mean']]}\n"
    f"y_std: {[f'{v:.2f}' for v in stats['y_std']]}",
    title="Summary",
))
