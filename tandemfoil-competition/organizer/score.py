"""Score predictions against hidden test ground truth.

Organizer-only. Scores submissions, logs to W&B, writes leaderboard to PVC.

Predictions layout (per agent/commit):
  /mnt/new-pvc/predictions/<tag>/<agent>/<commit>/
  ├── test_single_in_dist.pt    list of [N_i, 3] tensors
  ├── test_geom_camber_rc.pt
  ├── test_geom_camber_cruise.pt
  └── test_re_rand.pt

Run:
  python score.py --score_all
  python score.py --predictions /mnt/new-pvc/predictions/<tag>/frieren/abc1234
"""

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import wandb

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits_v2")
PREDICTIONS_ROOT = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SCORES_FILE = PREDICTIONS_ROOT / "scores.json"

TEST_SPLITS = [
    "test_single_in_dist",
    "test_geom_camber_rc",
    "test_geom_camber_cruise",
    "test_re_rand",
]


@dataclass
class Config:
    """Score predictions against hidden test ground truth."""
    predictions: str = ""  # path to a submission directory (containing per-split .pt files)
    score_all: bool = False
    splits_dir: str = str(SPLITS_DIR)


def load_ground_truth(splits_dir: Path) -> dict[str, list[dict]]:
    """Load all ground truth files for all test splits into memory."""
    gt = {}
    for split in TEST_SPLITS:
        gt_dir = splits_dir / f".{split}_gt"
        gt_files = sorted(gt_dir.glob("*.pt"))
        print(f"  {split}: {len(gt_files)} samples")
        gt[split] = [torch.load(f, map_location="cpu", weights_only=False) for f in gt_files]
    return gt


def score_split(preds: list[torch.Tensor], gt: list[dict]) -> dict[str, float]:
    """Score one test split. Returns per-channel surface and volume MAE."""
    assert len(preds) == len(gt), f"Count mismatch: {len(preds)} vs {len(gt)}"

    mae_surf = torch.zeros(3, dtype=torch.float64)
    mae_vol = torch.zeros(3, dtype=torch.float64)
    n_surf = n_vol = 0

    for i in range(len(preds)):
        pred_y, true_y = preds[i], gt[i]["y"]
        is_surface = gt[i]["is_surface"]
        assert pred_y.shape == true_y.shape, f"Sample {i}: {pred_y.shape} vs {true_y.shape}"

        # Skip samples with non-finite ground truth (data artifact)
        if not torch.isfinite(true_y).all():
            continue

        err = (pred_y.double() - true_y.double()).abs()
        mae_surf += (err * is_surface.unsqueeze(-1)).sum(0)
        n_surf += is_surface.sum().item()
        mae_vol += (err * (~is_surface).unsqueeze(-1)).sum(0)
        n_vol += (~is_surface).sum().item()

    s = mae_surf / max(n_surf, 1)
    v = mae_vol / max(n_vol, 1)
    return {
        "mae_surf_Ux": s[0].item(), "mae_surf_Uy": s[1].item(), "mae_surf_p": s[2].item(),
        "mae_vol_Ux": v[0].item(), "mae_vol_Uy": v[1].item(), "mae_vol_p": v[2].item(),
    }


def score_submission(pred_dir: Path, gt: dict[str, list[dict]]) -> dict[str, float] | None:
    """Score a full submission (all 4 test splits). Returns None if incomplete."""
    missing = [s for s in TEST_SPLITS if not (pred_dir / f"{s}.pt").exists()]
    if missing:
        print(f"    INCOMPLETE — missing: {', '.join(missing)}")
        return None

    results = {}
    split_surf_p = []

    for split in TEST_SPLITS:
        pred_path = pred_dir / f"{split}.pt"

        preds = torch.load(pred_path, map_location="cpu", weights_only=True)
        split_results = score_split(preds, gt[split])

        for k, v in split_results.items():
            results[f"{split}/{k}"] = v

        split_surf_p.append(split_results["mae_surf_p"])

    # Equal-weight average across splits
    if split_surf_p:
        results["avg/mae_surf_p"] = sum(split_surf_p) / len(split_surf_p)

        surf_ux = [results[f"{s}/mae_surf_Ux"] for s in TEST_SPLITS if f"{s}/mae_surf_Ux" in results]
        surf_uy = [results[f"{s}/mae_surf_Uy"] for s in TEST_SPLITS if f"{s}/mae_surf_Uy" in results]
        vol_p = [results[f"{s}/mae_vol_p"] for s in TEST_SPLITS if f"{s}/mae_vol_p" in results]
        vol_ux = [results[f"{s}/mae_vol_Ux"] for s in TEST_SPLITS if f"{s}/mae_vol_Ux" in results]
        vol_uy = [results[f"{s}/mae_vol_Uy"] for s in TEST_SPLITS if f"{s}/mae_vol_Uy" in results]

        results["avg/mae_surf_Ux"] = sum(surf_ux) / len(surf_ux)
        results["avg/mae_surf_Uy"] = sum(surf_uy) / len(surf_uy)
        results["avg/mae_vol_p"] = sum(vol_p) / len(vol_p)
        results["avg/mae_vol_Ux"] = sum(vol_ux) / len(vol_ux)
        results["avg/mae_vol_Uy"] = sum(vol_uy) / len(vol_uy)

    return results


def log_to_wandb(results: dict, agent: str, commit: str):
    """Log scores as a W&B run."""
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v2"),
        name=f"score/{agent}/{commit}",
        tags=["score", agent, RESEARCH_TAG],
        config={"agent": agent, "commit": commit},
        job_type="scoring",
    )
    wandb.log({f"test/{k}": v for k, v in results.items()})
    wandb.summary.update({f"test/{k}": v for k, v in results.items()})
    wandb.finish()


def load_scores() -> dict:
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text())
    return {}


def save_scores(scores: dict):
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORES_FILE.write_text(json.dumps(scores, indent=2))


def update_leaderboard(scores: dict):
    """Write leaderboard to PVC (readable by all pods)."""
    if not scores:
        return

    best_per_agent: dict[str, tuple[str, dict]] = {}
    for key, results in scores.items():
        if not isinstance(results, dict):
            continue
        agent, commit = key.split("/", 1)
        surf_p = results.get("avg/mae_surf_p", float("inf"))
        if agent not in best_per_agent or surf_p < best_per_agent[agent][1].get("avg/mae_surf_p", float("inf")):
            best_per_agent[agent] = (commit, results)

    ranked = sorted(best_per_agent.items(), key=lambda x: x[1][1].get("avg/mae_surf_p", float("inf")))

    lines = [
        f"# Leaderboard ({RESEARCH_TAG})",
        "",
        "Ranked by **avg surface pressure MAE** across 4 test splits (lower is better).",
        "",
        "| Rank | Agent | Commit | avg_surf_p | single_in_dist | geom_rc | geom_cruise | re_rand |",
        "|------|-------|--------|-----------|----------------|---------|-------------|---------|",
    ]

    for rank, (agent, (commit, r)) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {agent} | `{commit[:7]}` "
            f"| {r.get('avg/mae_surf_p', 0):.2f} "
            f"| {r.get('test_single_in_dist/mae_surf_p', 0):.2f} "
            f"| {r.get('test_geom_camber_rc/mae_surf_p', 0):.2f} "
            f"| {r.get('test_geom_camber_cruise/mae_surf_p', 0):.2f} "
            f"| {r.get('test_re_rand/mae_surf_p', 0):.2f} |"
        )

    lines.extend(["", f"*Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*", ""])

    leaderboard_path = PREDICTIONS_ROOT / "leaderboard.md"
    leaderboard_path.write_text("\n".join(lines))
    print(f"  Leaderboard updated ({len(ranked)} agents) → {leaderboard_path}")


cfg = sp.parse(Config)
splits_dir = Path(cfg.splits_dir)

if cfg.score_all:
    scores = load_scores()

    # Find pending submissions — glob for any test split .pt file
    pending = []
    seen = set()
    for pred_file in sorted(PREDICTIONS_ROOT.glob(f"*/*/{TEST_SPLITS[0]}.pt")):
        commit_dir = pred_file.parent
        key = f"{commit_dir.parent.name}/{commit_dir.name}"
        if key not in scores and key not in seen:
            seen.add(key)
            pending.append((key, commit_dir))

    if not pending:
        print(f"All {len(scores)} submissions already scored")
    else:
        print(f"{len(pending)} new submissions to score ({len(scores)} already done)")
        gt = load_ground_truth(splits_dir)

        for i, (key, pred_dir) in enumerate(pending):
            agent, commit = key.split("/", 1)
            print(f"  [{i+1}/{len(pending)}] {key}")
            results = score_submission(pred_dir, gt)
            if results is None:
                scores[key] = "incomplete"
                continue
            log_to_wandb(results, agent, commit)
            scores[key] = results
            if (i + 1) % 10 == 0:
                save_scores(scores)

        save_scores(scores)
        update_leaderboard(scores)

elif cfg.predictions:
    pred_dir = Path(cfg.predictions)
    relative = pred_dir.relative_to(PREDICTIONS_ROOT)
    agent = relative.parts[0]
    commit = relative.parts[1]
    print(f"Scoring: {agent} @ {commit}")
    gt = load_ground_truth(splits_dir)
    results = score_submission(pred_dir, gt)
    if results is None:
        print("  Incomplete submission — not scored")
    else:
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")
        log_to_wandb(results, agent, commit)
        scores = load_scores()
        scores[f"{agent}/{commit}"] = results
        save_scores(scores)
        print(f"Saved to {SCORES_FILE}")
else:
    print("Specify --predictions <path> or --score_all")
