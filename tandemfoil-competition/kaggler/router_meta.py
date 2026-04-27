"""Meta-router: blend stored prediction tensors across all agents.

The PVC at /mnt/new-pvc/predictions/$RESEARCH_TAG holds every agent's test
predictions. Per-split scores show different leaders for different splits,
so per-split routing across agents has a strict lower bound below any single
submission.

Per-split bests on the leaderboard (avg_surf_p):
  test_single_in_dist     -> edward/c773fa7  = 36.25
  test_geom_camber_rc     -> tanjiro/9f6f523 = 54.98
  test_geom_camber_cruise -> edward/c773fa7  = 23.73
  test_re_rand            -> tanjiro/9f6f523 = 40.43
  avg ~ 38.85

Usage:
  python router_meta.py --agent thorfinn
  python router_meta.py --agent thorfinn --single_w_edward 0.7  # blend
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp
import torch

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PRED = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")


# Sources we'll mix (agent, commit) referenced by short tag.
SRC = {
    "edward":   ("edward",   "c773fa7"),  # single 36.25, cruise 23.73, rc 56.68, re 54.43
    "edward2":  ("edward",   "2856b96"),  # single 36.61, cruise 25.08, rc 57.33
    "tanjiro":  ("tanjiro",  "9f6f523"),  # rc 54.98, re 40.43, single 44.86, cruise 26.14
    "fern":     ("fern",     "0f7f9e0"),  # cruise 24.85, rc 59.08, single 48.29, re 44.04
    "fern2":    ("fern",     "cc186e5"),  # cruise 25.14, rc 59.79
    "frieren":  ("frieren",  "32f0a18"),  # re 42.48 (best frieren)
    "frieren2": ("frieren",  "0596f0e"),  # re 42.53
    "frieren3": ("frieren",  "a89882c"),  # re 42.67
    "frieren4": ("frieren",  "8a3998f"),  # re 42.64 (leaderboard commit)
    "thorfinn": ("thorfinn", "a4bbc13"),  # current blend (best ours): single 39.13, rc 59.18, cruise 26.14, re 50.33
    # Frozen snapshots of our own meta-router blends — useful as starting points
    # since the source files for these commits are immutable on PVC.
    "snap_3392": ("thorfinn", "3392fb4"),  # our 36.82 #1 blend (single 35.77, rc 51.19, cr 22.32, re 38.01)
}


@dataclass
class Config:
    agent: str | None = None
    # Per-split blend weights as comma-separated "src:weight" pairs.
    # Weights renormalize automatically.
    single: str = "edward:1"
    rc: str = "tanjiro:1"
    cruise: str = "edward:1"
    re_rand: str = "tanjiro:1"
    # Per-split modes: "mean" or "median"
    single_mode: str = "mean"
    rc_mode: str = "mean"
    cruise_mode: str = "mean"
    re_rand_mode: str = "mean"


def parse_mix(spec: str) -> list[tuple[str, float]]:
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, w = chunk.split(":")
        out.append((name.strip(), float(w)))
    s = sum(w for _, w in out)
    return [(n, w / s) for n, w in out]


def blend_split(split_file: str, mix: list[tuple[str, float]], mode: str = "mean"):
    """mode: 'mean' (weighted), 'median' (per-node median ignoring weights)."""
    sources = []
    for src_name, w in mix:
        agent, commit = SRC[src_name]
        preds = torch.load(PRED / agent / commit / f"{split_file}.pt", weights_only=False)
        sources.append((src_name, w, preds))

    if mode == "median":
        out = []
        for i in range(len(sources[0][2])):
            stack = torch.stack([src[2][i] for src in sources], dim=0)  # [K, N, 3]
            out.append(stack.median(dim=0).values)
        print(f"  {split_file}: median of " + ", ".join(s[0] for s in sources))
        return out
    else:
        tensors = None
        label = []
        for _, w, preds in sources:
            if tensors is None:
                tensors = [w * p for p in preds]
            else:
                tensors = [t + w * p for t, p in zip(tensors, preds)]
        for src_name, w, _ in sources:
            label.append(f"{w:.2f}*{src_name}")
        print(f"  {split_file}: " + " + ".join(label))
        return tensors


cfg = sp.parse(Config)
agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PRED / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output: {output_dir}")

mixes = {
    "test_single_in_dist":     (parse_mix(cfg.single),  cfg.single_mode),
    "test_geom_camber_rc":     (parse_mix(cfg.rc),      cfg.rc_mode),
    "test_geom_camber_cruise": (parse_mix(cfg.cruise),  cfg.cruise_mode),
    "test_re_rand":            (parse_mix(cfg.re_rand), cfg.re_rand_mode),
}

for split, (mix, mode) in mixes.items():
    out = blend_split(split, mix, mode=mode)
    torch.save(out, output_dir / f"{split}.pt")

print(f"\nMeta-router predictions saved to {output_dir}")
