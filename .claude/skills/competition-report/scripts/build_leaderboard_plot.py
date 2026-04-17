"""Build a leaderboard-evolution plot from a running kagent competition.

Pulls scores.json from the organiser pod, resolves each scored commit's
timestamp from the kaggler pod that produced it (because discarded-then-reset
commits are only reachable in the local reflog of that pod), then renders a
per-agent running-best staircase with a black envelope for the current leader.

Usage:
  uv run --with matplotlib --with numpy --no-project \\
      .claude/skills/competition-report/scripts/build_leaderboard_plot.py \\
      --tag apr16 --competition gram-competition \\
      --out gram-competition/leaderboard_evolution.png
"""

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import simple_parsing as sp
import yaml


PALETTE = [
    "#E45756", "#4C78A8", "#54A24B", "#F58518",
    "#B279A2", "#EECA3B", "#72B7B2", "#9C9C9C",
    "#D62728", "#7F7F7F", "#BCBD22", "#17BECF",
]


@dataclass
class Args:
    tag: str = ""           # research tag (default: read from config.yaml)
    competition: str = ""   # competition dir (default: read from config.yaml)
    out: str = ""           # output PNG path
    title: str = ""         # plot title (default: auto)


def load_config_defaults() -> dict:
    cfg_path = Path(__file__).resolve().parents[4] / "config.yaml"
    if cfg_path.exists():
        return yaml.safe_load(cfg_path.read_text())
    return {}


def kubectl_exec(deployment: str, script: str) -> str:
    result = subprocess.run(
        ["kubectl", "exec", f"deployment/{deployment}", "--", "bash", "-c", script],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"  kubectl exec {deployment} failed: {result.stderr[:400]}", file=sys.stderr)
    return result.stdout


def fetch_scores(tag: str) -> dict:
    """Pull scores.json from the organiser pod."""
    out = kubectl_exec(
        f"kagent-{tag}-organizer",
        f"cat /mnt/new-pvc/predictions/{tag}/scores.json",
    )
    if not out.strip():
        sys.exit("could not fetch scores.json — is the organizer running?")
    return json.loads(out)


def resolve_timestamps(tag: str, scores: dict) -> list[tuple[int, str, str, float]]:
    """Resolve (commit_ts, agent, short, l2_error) tuples, asking each agent's pod."""
    per_agent: dict[str, list[tuple[str, float]]] = {}
    for key, rec in scores.items():
        if "/" not in key:
            continue
        agent, short = key.split("/", 1)
        if agent == "unknown":
            continue
        per_agent.setdefault(agent, []).append((short, rec["val/l2_error"]))

    out: list[tuple[int, str, str, float]] = []
    for agent, entries in sorted(per_agent.items()):
        commits_json = json.dumps([[c, v] for c, v in entries])
        script = f"""
cd /workspace/kagent 2>/dev/null || exit 0
python3 - <<'PY'
import json, subprocess
rows = {commits_json}
out = []
for short, v in rows:
    r = subprocess.run(['git', 'show', '-s', '--format=%ct', short],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        out.append([int(r.stdout.strip()), '{agent}', short, v])
print(json.dumps(out))
PY
"""
        stdout = kubectl_exec(f"kagent-{tag}-{agent}", script)
        try:
            rows = json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else []
        except json.JSONDecodeError:
            rows = []
        print(f"  {agent}: {len(rows)}/{len(entries)} resolved")
        out.extend(rows)
    out.sort()
    return out


def render(timeline: list[tuple[int, str, str, float]], out_path: Path, title: str) -> None:
    agents_ordered = [a for a, _ in Counter(row[1] for row in timeline).most_common()]
    colours = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(agents_ordered)}

    running: dict[str, list[tuple[int, float]]] = {}
    for ts, agent, _, l2 in timeline:
        hist = running.setdefault(agent, [])
        best = min(hist[-1][1], l2) if hist else l2
        hist.append((ts, best))

    overall, cur = [], float("inf")
    for ts, _, _, l2 in timeline:
        cur = min(cur, l2)
        overall.append((ts, cur))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)

    for agent in agents_ordered:
        pts = [(dt.datetime.fromtimestamp(r[0]), r[3]) for r in timeline if r[1] == agent]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=22, alpha=0.35, color=colours[agent], zorder=2)

    for agent in agents_ordered:
        pts = running.get(agent, [])
        if not pts:
            continue
        xs = [dt.datetime.fromtimestamp(t) for t, _ in pts]
        ys = [v for _, v in pts]
        ax.step(xs, ys, where="post", color=colours[agent],
                linewidth=2.0, alpha=0.95, label=agent, zorder=3)

    ov_xs = [dt.datetime.fromtimestamp(t) for t, _ in overall]
    ov_ys = [v for _, v in overall]
    ax.step(ov_xs, ov_ys, where="post", color="black",
            linewidth=2.8, alpha=0.9, label="leader", zorder=4)

    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

    ax.set_ylabel("val/l2 error  (lower is better)", fontsize=11)
    ax.set_xlabel("submission time  (UTC)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)

    ymin = min(v for _, v in overall) * 0.97
    ymax = np.percentile([r[3] for r in timeline], 95) * 1.05
    ax.set_ylim(ymin, ymax)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, which="major", axis="y", linewidth=0.5, alpha=0.7)
    ax.grid(True, which="major", axis="x", linewidth=0.5, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    lead_i = labels.index("leader")
    labels.insert(0, labels.pop(lead_i))
    handles.insert(0, handles.pop(lead_i))
    ax.legend(handles, labels, loc="upper right", frameon=True,
              framealpha=0.95, fontsize=9, title="agent")

    leader_ts, leader_val = overall[-1]
    ax.annotate(
        f"current best: {leader_val:.3f}",
        xy=(dt.datetime.fromtimestamp(leader_ts), leader_val),
        xytext=(10, -18), textcoords="offset points",
        fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2, alpha=0.8),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"wrote {out_path}  ({len(timeline)} submissions, {len(agents_ordered)} agents)")


def main() -> None:
    defaults = load_config_defaults()
    args = sp.parse(Args)
    tag = args.tag or defaults.get("tag") or sys.exit("missing --tag")
    competition = args.competition or defaults.get("competition", "")
    out = Path(args.out) if args.out else Path(competition) / "leaderboard_evolution.png"
    title = args.title or f"kagent {tag} leaderboard — evolution"

    print(f"tag={tag}  competition={competition}  out={out}")
    scores = fetch_scores(tag)
    print(f"scores.json: {len(scores)} entries")
    timeline = resolve_timestamps(tag, scores)
    if not timeline:
        sys.exit("no commits resolved — are the kaggler pods still running?")
    render(timeline, out, title)


if __name__ == "__main__":
    main()
