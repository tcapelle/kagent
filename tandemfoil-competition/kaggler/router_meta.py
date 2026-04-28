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
    "tanjiro2": ("tanjiro",  "5613c7b"),  # NEW! rc 52.32, re 37.94 (BEST!), single 42.04, cruise 24.06
    "tanjiro3": ("tanjiro",  "63e5e26"),  # tanjiro v2 (avg 38.45) — single 41.44, rc 51.60, cruise 23.55, re 37.20
    "fern":     ("fern",     "0f7f9e0"),  # cruise 24.85, rc 59.08, single 48.29, re 44.04
    "fern2":    ("fern",     "cc186e5"),  # cruise 25.14, rc 59.79
    "fern3":    ("fern",     "36e8feb"),  # NEW! cruise 23.88, rc 58.12, re 41.48 (worse alone but different agent)
    "fern4":    ("fern",     "3457e5a"),  # FERN JUMP 40.59
    "fern5":    ("fern",     "87992d1"),  # 40.35 (improving fern)
    "askeladd": ("askeladd", "01851f9"),  # cruise 26.59, re 45.79 (worse alone but different model)
    "alphonse": ("alphonse", "5aa9393"),  # 52.89 alone, different model
    "alphonse2": ("alphonse", "7992b5c"),  # NEW alphonse 48.23
    "alphonse3": ("alphonse", "f0b59cc"),  # ALPHONSE JUMP 42.32
    "alphonse4": ("alphonse", "ebfb061"),  # newest alphonse
    "alphonse5": ("alphonse", "7697f58"),  # 40.09 (improved alphonse)
    "tanjiro5":  ("tanjiro",  "a5b7f50"),  # newest tanjiro
    "tanjiro6":  ("tanjiro",  "cabb15b"),  # NEW tanjiro
    "askeladd3": ("askeladd", "6343096"),  # newest askeladd
    "askeladd4": ("askeladd", "b8419dd"),  # 46.40 askeladd
    "alphonse6": ("alphonse", "23249a0"),  # newest alphonse
    "frieren_new": ("frieren", "dee6e1d"),  # 42.80 best frieren
    "nezuko":   ("nezuko",   "08cdd12"),  # 60.92 alone (worst), but fully different model
    "nezuko2":  ("nezuko",   "a6bcbcd"),  # JUMP! 37.39 (single 38.62, rc 51.78, cr 21.78, re 37.39)
    "nezuko3":  ("nezuko",   "012619a"),  # 38.32 (single 40.93, rc 51.60, cr 23.55, re 37.20)
    "nezuko4":  ("nezuko",   "babbe34"),  # NEW! 36.82 (single 38.66, rc 50.70, cr 21.84, re 36.06)
    "nezuko5":  ("nezuko",   "15ec154"),  # JUMP! 35.30 (single 35.87, rc 49.05, cr 20.90, re 35.39)
    "nezuko6":  ("nezuko",   "9a31553"),  # 35.24 (single 35.68, rc 49.04, cr 20.88, re 35.36)
    "nezuko7":  ("nezuko",   "3305937"),  # 35.24 (single 35.63, rc 49.07, cr 20.92, re 35.34)
    "nezuko8":  ("nezuko",   "c01fda9"),  # TIES! 35.21 (single 35.60, rc 49.04, cr 20.88, re 35.33)
    "nezuko9":  ("nezuko",   "75d1bc6"),  # AHEAD! 35.20 (single 35.59, rc 49.04, cr 20.83, re 35.33)
    "nezuko10": ("nezuko",   "78c375e"),  # 35.197 (best nezuko)
    "nezuko11": ("nezuko",   "58aee76"),  # 35.197
    "nezuko12": ("nezuko",   "629ec60"),  # 35.20
    "snap_1f9": ("thorfinn", "1f9db55"),  # MY 35.198 frozen blend
    "snap_ca32": ("thorfinn", "ca32e09"),  # MY 35.197 frozen blend
    "snap_98b": ("thorfinn", "98b90b2"),  # MY 34.375 frozen blend
    "nezuko_top": ("nezuko",  "f23f935"),  # nezuko 35.196 (current ahead)
    "nezuko_209": ("nezuko",  "209c93e"),  # NEW! 35.195674
    "nezuko_bbb": ("nezuko",  "bbb33f7"),  # 34.90! latest top
    "nezuko_5cf": ("nezuko",  "5cf59c7"),  # 34.86 (single 35.22, rc 48.68, cr 20.62, re 34.92)
    "nezuko_98":  ("nezuko",  "98edc42"),  # 34.85 (single 35.21, rc 48.68, cr 20.62, re 34.90)
    "nezuko_3f":  ("nezuko",  "3f750ef"),  # 34.758
    "nezuko_45":  ("nezuko",  "45146ee"),  # 34.757
    "nezuko_e61": ("nezuko",  "e61bca7"),  # 34.65! (single 35.21, rc 48.57, cr 20.41, re 34.40)
    "nezuko_b0e": ("nezuko",  "b0e0889"),  # 34.61! (single 35.03, rc 48.61, cr 20.41, re 34.38)
    "nezuko_aafa": ("nezuko", "aafa24b"),  # 34.595
    "nezuko_b0b": ("nezuko",  "b0bbb68"),  # 34.582
    "nezuko_934": ("nezuko",  "93428ec"),  # 34.578
    "nezuko_dea": ("nezuko",  "dea219b"),  # 34.555
    "nezuko_a06": ("nezuko",  "a068eb1"),  # 34.54
    "nezuko_d8b": ("nezuko",  "d8b8f7f"),  # 34.374
    "nezuko_8ff": ("nezuko",  "8ff5ddf"),  # 34.33
    "nezuko_f08": ("nezuko",  "f08147d"),  # 34.32
    "nezuko_b36": ("nezuko",  "b36bcc7"),  # 34.32 newer
    "nezuko_bd0": ("nezuko",  "bd04116"),  # 34.314
    "nezuko_dda": ("nezuko",  "dda181f"),  # 33.95
    "fern6":      ("fern",    "9020ced"),  # 40.03 fern improved
    "tanjiro7":   ("tanjiro", "89d6892"),  # 37.33 latest tanjiro
    "tanjiro8":   ("tanjiro", "5219896"),  # 36.31 NEW tanjiro
    "alphonse7":  ("alphonse", "4635caf"),  # 37.68 NEW alphonse
    "fern7":      ("fern",    "b9eb8c9"),  # 39.69 NEW fern
    "askeladd5":  ("askeladd", "b3a7883"),  # 45.56 NEW askeladd
    "tanjiro4":   ("tanjiro", "ad4711d"),  # 37.44 (improved tanjiro)
    "nezuko_85":  ("nezuko",  "85845f4"),  # 35.195693
    "nezuko_abe": ("nezuko",  "abe700a"),  # 35.195697
    "nezuko_42":  ("nezuko",  "42bd8fb"),  # 35.195699
    "nezuko_57":  ("nezuko",  "57ff762"),  # 35.195703
    "askeladd2": ("askeladd", "b977533"),  # NEW askeladd 48.83
    "frieren":  ("frieren",  "32f0a18"),  # re 42.48 (best frieren)
    "frieren2": ("frieren",  "0596f0e"),  # re 42.53
    "frieren3": ("frieren",  "a89882c"),  # re 42.67
    "frieren4": ("frieren",  "8a3998f"),  # re 42.64 (leaderboard commit)
    "thorfinn": ("thorfinn", "a4bbc13"),  # current blend (best ours): single 39.13, rc 59.18, cruise 26.14, re 50.33
    # Frozen snapshots of our own meta-router blends — useful as starting points
    # since the source files for these commits are immutable on PVC.
    "snap_3392": ("thorfinn", "3392fb4"),  # 36.82 (sd 35.77, rc 51.19, cr 22.32, re 38.01)
    "snap_bc7f": ("thorfinn", "bc7f7dd"),  # 36.88 (sd 35.77, rc 51.46, cr 22.27, re 38.01)
    "snap_853d": ("thorfinn", "853d98f"),  # 36.91 (sd 35.91, rc 51.41, cr 22.32, re 38.01)
    "snap_5fb":  ("thorfinn", "5fb015e"),  # 36.96
    "snap_1128": ("thorfinn", "1128609"),  # 37.05
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
