"""Score predictions against hidden test ground truth.

Organizer-only. Scores submissions and logs results to W&B.

Run:
  python score.py --predictions /mnt/new-pvc/predictions/frieren/abc1234/predictions.pt
  python score.py --score_all
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch
import wandb

SPLITS_DIR = Path("/mnt/new-pvc/datasets/tandemfoil/splits")
PREDICTIONS_ROOT = Path("/mnt/new-pvc/predictions")
SCORES_FILE = PREDICTIONS_ROOT / "scores.json"
CHANNELS = ["Ux", "Uy", "p"]


@dataclass
class Config:
    """Score predictions against hidden test ground truth."""
    predictions: str = ""  # path to a single predictions.pt
    score_all: bool = False  # score all unscored submissions in predictions dir
    gt_dir: str = str(SPLITS_DIR / ".test_gt")


def load_ground_truth(gt_dir: Path) -> list[dict]:
    """Load all ground truth files into memory once."""
    gt_files = sorted(gt_dir.glob("*.pt"))
    print(f"Loading {len(gt_files)} ground truth files into memory...")
    gt = [torch.load(f, map_location="cpu", weights_only=False) for f in gt_files]
    print(f"Ground truth cached ({len(gt)} samples)")
    return gt


def score_predictions(predictions_path: Path, gt: list[dict]) -> dict:
    """Score a predictions.pt file against cached ground truth."""
    preds = torch.load(predictions_path, map_location="cpu", weights_only=True)
    assert len(preds) == len(gt), f"Count mismatch: {len(preds)} vs {len(gt)}"

    domains: dict[str, dict] = {}
    for i in range(len(preds)):
        pred_y, true_y = preds[i], gt[i]["y"]
        is_surface, domain = gt[i]["is_surface"], gt[i]["domain"]
        assert pred_y.shape == true_y.shape, f"Sample {i}: {pred_y.shape} vs {true_y.shape}"

        err = (pred_y - true_y).abs()
        if domain not in domains:
            domains[domain] = {"mae_surf": torch.zeros(3), "n_surf": 0,
                               "mae_vol": torch.zeros(3), "n_vol": 0}
        d = domains[domain]
        d["mae_surf"] += (err * is_surface.unsqueeze(-1)).sum(0)
        d["n_surf"] += is_surface.sum().item()
        d["mae_vol"] += (err * (~is_surface).unsqueeze(-1)).sum(0)
        d["n_vol"] += (~is_surface).sum().item()

    results = {}
    total_surf, total_vol = torch.zeros(3), torch.zeros(3)
    total_n_surf = total_n_vol = 0

    for domain in sorted(domains):
        d = domains[domain]
        s = d["mae_surf"] / max(d["n_surf"], 1)
        v = d["mae_vol"] / max(d["n_vol"], 1)
        results[f"{domain}/mae_surf_Ux"] = s[0].item()
        results[f"{domain}/mae_surf_Uy"] = s[1].item()
        results[f"{domain}/mae_surf_p"] = s[2].item()
        results[f"{domain}/mae_vol_Ux"] = v[0].item()
        results[f"{domain}/mae_vol_Uy"] = v[1].item()
        results[f"{domain}/mae_vol_p"] = v[2].item()
        total_surf += d["mae_surf"]
        total_vol += d["mae_vol"]
        total_n_surf += d["n_surf"]
        total_n_vol += d["n_vol"]

    s = total_surf / max(total_n_surf, 1)
    v = total_vol / max(total_n_vol, 1)
    results["overall/mae_surf_Ux"] = s[0].item()
    results["overall/mae_surf_Uy"] = s[1].item()
    results["overall/mae_surf_p"] = s[2].item()
    results["overall/mae_vol_Ux"] = v[0].item()
    results["overall/mae_vol_Uy"] = v[1].item()
    results["overall/mae_vol_p"] = v[2].item()
    return results


def log_to_wandb(results: dict, agent: str, commit: str):
    """Log scores as a W&B run."""
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-v1"),
        name=f"score/{agent}/{commit}",
        tags=["score", agent],
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


def update_leaderboard(scores: dict, repo_dir: str = "/workspace/kagent"):
    """Update leaderboard.md in the repo and push to main."""
    if not scores:
        return

    best_per_agent: dict[str, tuple[str, dict]] = {}
    for key, results in scores.items():
        agent, commit = key.split("/", 1)
        surf_p = results.get("overall/mae_surf_p", float("inf"))
        if agent not in best_per_agent or surf_p < best_per_agent[agent][1].get("overall/mae_surf_p", float("inf")):
            best_per_agent[agent] = (commit, results)

    ranked = sorted(best_per_agent.items(), key=lambda x: x[1][1].get("overall/mae_surf_p", float("inf")))

    CODEX_AGENTS = {"luffy", "zoro", "nami", "sanji"}

    lines = [
        "# Leaderboard",
        "",
        "Ranked by **overall surface pressure MAE** (lower is better).",
        "",
        "| Rank | Agent | Runtime | Commit | mae_surf_p | mae_surf_Ux | mae_surf_Uy | mae_vol_p | mae_vol_Ux | mae_vol_Uy |",
        "|------|-------|---------|--------|-----------|-------------|-------------|----------|-----------|-----------|",
    ]

    for rank, (agent, (commit, r)) in enumerate(ranked, 1):
        runtime = "codex" if agent in CODEX_AGENTS else "claude"
        lines.append(
            f"| {rank} | {agent} | {runtime} | `{commit[:7]}` "
            f"| {r.get('overall/mae_surf_p', 0):.2f} "
            f"| {r.get('overall/mae_surf_Ux', 0):.2f} "
            f"| {r.get('overall/mae_surf_Uy', 0):.2f} "
            f"| {r.get('overall/mae_vol_p', 0):.2f} "
            f"| {r.get('overall/mae_vol_Ux', 0):.2f} "
            f"| {r.get('overall/mae_vol_Uy', 0):.2f} |"
        )

    lines.extend(["", f"*Last updated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*", ""])

    leaderboard_path = Path(repo_dir) / "leaderboard.md"
    leaderboard_path.write_text("\n".join(lines))

    import subprocess
    git = lambda *args: subprocess.run(["git", "-C", repo_dir] + list(args), capture_output=True, text=True)
    git("config", "user.name", "kagent-organizer")
    git("config", "user.email", "kagent-organizer@kagent")
    git("checkout", "main")
    git("pull", "origin", "main")
    git("add", "leaderboard.md")
    result = git("diff", "--cached", "--quiet")
    if result.returncode != 0:
        git("commit", "-m", "Update leaderboard")
        push = git("push", "origin", "main")
        if push.returncode == 0:
            print(f"  Leaderboard pushed to main ({len(ranked)} agents)")
        else:
            print(f"  Leaderboard push failed: {push.stderr.strip()}")
    else:
        print("  Leaderboard unchanged")


cfg = sp.parse(Config)
gt_dir = Path(cfg.gt_dir)

if cfg.score_all:
    scores = load_scores()

    # Find pending submissions
    pending = []
    for pred_file in sorted(PREDICTIONS_ROOT.glob("*/*/predictions.pt")):
        parts = pred_file.parts
        pred_idx = parts.index("predictions")
        key = f"{parts[pred_idx + 1]}/{parts[pred_idx + 2]}"
        if key not in scores:
            pending.append((key, pred_file))

    if not pending:
        print(f"All {len(scores)} submissions already scored")
    else:
        print(f"{len(pending)} new submissions to score ({len(scores)} already done)")

        # Load ground truth ONCE
        gt = load_ground_truth(gt_dir)

        for i, (key, pred_file) in enumerate(pending):
            agent, commit = key.split("/", 1)
            print(f"  [{i+1}/{len(pending)}] {key}")
            results = score_predictions(pred_file, gt)
            log_to_wandb(results, agent, commit)
            scores[key] = results
            # Save periodically (every 10 submissions)
            if (i + 1) % 10 == 0:
                save_scores(scores)

        save_scores(scores)
        update_leaderboard(scores)

elif cfg.predictions:
    gt = load_ground_truth(gt_dir)
    pred_path = Path(cfg.predictions)
    parts = pred_path.parts
    pred_idx = parts.index("predictions")
    agent = parts[pred_idx + 1]
    commit = parts[pred_idx + 2]
    print(f"Scoring: {agent} @ {commit}")
    results = score_predictions(pred_path, gt)
    log_to_wandb(results, agent, commit)
    scores = load_scores()
    scores[f"{agent}/{commit}"] = results
    save_scores(scores)
    print(f"Saved to {SCORES_FILE}")
else:
    print("Specify --predictions <path> or --score_all")
