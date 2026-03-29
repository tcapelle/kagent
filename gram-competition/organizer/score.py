"""Score predictions against hidden test ground truth.

Organizer-only. Scores submissions, logs to W&B, writes leaderboard to PVC.

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
SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")
PREDICTIONS_ROOT = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")
SCORES_FILE = PREDICTIONS_ROOT / "scores.json"

TEST_SPLITS = ["test"]


@dataclass
class Config:
    """Score predictions against hidden test ground truth."""
    predictions: str = ""  # path to a submission directory
    score_all: bool = False
    splits_dir: str = str(SPLITS_DIR)


def load_ground_truth(splits_dir: Path) -> dict[str, list[dict]]:
    """Load all ground truth files for test splits into memory."""
    gt = {}
    for split in TEST_SPLITS:
        gt_dir = splits_dir / f".{split}_gt"
        gt_files = sorted(gt_dir.glob("*.pt"))
        print(f"  {split}: {len(gt_files)} samples")
        gt[split] = [torch.load(f, map_location="cpu", weights_only=True) for f in gt_files]
    return gt


def score_split(preds: list[torch.Tensor], gt: list[dict]) -> dict[str, float]:
    """Score one test split. Returns L2 error and per-component MAE."""
    assert len(preds) == len(gt), f"Count mismatch: {len(preds)} vs {len(gt)}"

    total_l2 = 0.0
    total_mae = torch.zeros(3, dtype=torch.float64)
    n = 0

    for i in range(len(preds)):
        pred = preds[i]  # [5, N, 3]
        true = gt[i]["velocity_out"]  # [5, N, 3]
        assert pred.shape == true.shape, f"Sample {i}: {pred.shape} vs {true.shape}"

        if not torch.isfinite(true).all():
            continue

        l2 = (pred.double() - true.double()).norm(dim=2).mean(dim=(0, 1))  # scalar
        total_l2 += l2.item()

        mae = (pred.double() - true.double()).abs().mean(dim=(0, 1))  # [3]
        total_mae += mae
        n += 1

    return {
        "l2_error": total_l2 / max(n, 1),
        "mae_Ux": (total_mae[0] / max(n, 1)).item(),
        "mae_Uy": (total_mae[1] / max(n, 1)).item(),
        "mae_Uz": (total_mae[2] / max(n, 1)).item(),
    }


def score_submission(pred_dir: Path, gt: dict[str, list[dict]]) -> dict[str, float] | None:
    """Score a full submission. Returns None if incomplete."""
    missing = [s for s in TEST_SPLITS if not (pred_dir / f"{s}.pt").exists()]
    if missing:
        print(f"    INCOMPLETE — missing: {', '.join(missing)}")
        return None

    results = {}
    for split in TEST_SPLITS:
        preds = torch.load(pred_dir / f"{split}.pt", map_location="cpu", weights_only=True)
        split_results = score_split(preds, gt[split])
        for k, v in split_results.items():
            results[f"{split}/{k}"] = v

    # Overall (just test for now, single split)
    results["avg/l2_error"] = results.get("test/l2_error", float("inf"))
    results["avg/mae_Ux"] = results.get("test/mae_Ux", 0)
    results["avg/mae_Uy"] = results.get("test/mae_Uy", 0)
    results["avg/mae_Uz"] = results.get("test/mae_Uz", 0)
    return results


def log_to_wandb(results: dict, agent: str, commit: str):
    """Log scores as a W&B run."""
    wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-gram"),
        group=RESEARCH_TAG,
        name=f"score/{agent}/{commit}",
        tags=["score", agent, RESEARCH_TAG],
        config={"agent": agent, "commit": commit, "research_tag": RESEARCH_TAG},
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
    """Write leaderboard to PVC."""
    if not scores:
        return

    best_per_agent: dict[str, tuple[str, dict]] = {}
    for key, results in scores.items():
        if not isinstance(results, dict):
            continue
        agent, commit = key.split("/", 1)
        l2 = results.get("avg/l2_error", float("inf"))
        if agent not in best_per_agent or l2 < best_per_agent[agent][1].get("avg/l2_error", float("inf")):
            best_per_agent[agent] = (commit, results)

    ranked = sorted(best_per_agent.items(), key=lambda x: x[1][1].get("avg/l2_error", float("inf")))

    lines = [
        f"# Leaderboard ({RESEARCH_TAG})",
        "",
        "Ranked by **mean L2 velocity error** (lower is better).",
        "",
        "| Rank | Agent | Commit | l2_error | mae_Ux | mae_Uy | mae_Uz |",
        "|------|-------|--------|---------|--------|--------|--------|",
    ]

    for rank, (agent, (commit, r)) in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | {agent} | `{commit[:7]}` "
            f"| {r.get('avg/l2_error', 0):.4f} "
            f"| {r.get('avg/mae_Ux', 0):.4f} "
            f"| {r.get('avg/mae_Uy', 0):.4f} "
            f"| {r.get('avg/mae_Uz', 0):.4f} |"
        )

    lines.extend(["", f"*Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*", ""])

    leaderboard_path = PREDICTIONS_ROOT / "leaderboard.md"
    leaderboard_path.write_text("\n".join(lines))
    print(f"  Leaderboard updated ({len(ranked)} agents) -> {leaderboard_path}")


cfg = sp.parse(Config)
splits_dir = Path(cfg.splits_dir)

if cfg.score_all:
    scores = load_scores()

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
