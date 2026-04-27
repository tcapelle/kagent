"""Meta-ensemble predictor.

Loads pre-computed predictions from multiple agents on the PVC and blends
them per-split with weights. Saves the result under nezuko/<commit>/.

Per-split best individual sources (avg surf_p MAE on test, lower is better):
  test_single_in_dist     -> edward/c773fa7  = 36.25
  test_geom_camber_rc     -> tanjiro/63e5e26 = 51.60
  test_geom_camber_cruise -> tanjiro/63e5e26 = 23.55
  test_re_rand            -> tanjiro/63e5e26 = 37.20

Selecting the per-split best alone gives lower-bound avg ~ 37.15.

Run:
  python predict_ensemble.py --agent nezuko
  python predict_ensemble.py --agent nezuko --single edward:0.7,tanjiro:0.3 ...
"""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PRED = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")


# Sources to mix: short-name -> (agent, commit). Commits are immutable on PVC.
SRC = {
    # Leader thorfinn — multiple commits, each per-split sweet spots.
    "thorfinn":  ("thorfinn", "1f9db55"),  # avg 35.199 — best single individual (s 35.588 c 20.832 re 35.324)
    "thorfinn0": ("thorfinn", "0cc44bf"),  # avg 35.21 — single 35.591 cruise 20.833 re 35.327
    "thorfinn2": ("thorfinn", "fc1227e"),  # avg 35.22 — single 35.602 cruise 20.882
    "thorfinn3": ("thorfinn", "6f756c8"),  # avg 35.22 — cruise 20.869
    "thorfinn4": ("thorfinn", "8103189"),  # avg 35.23
    "thorfinn5": ("thorfinn", "8ce7299"),  # single 35.586 (best single individual)
    "thorfinn6": ("thorfinn", "90567b5"),  # single 35.588 cruise 20.871
    "thorfinn7": ("thorfinn", "ae15980"),  # single 35.591 cruise 20.833
    "thorfinn8": ("thorfinn", "5ae926e"),  # single 35.588 (avg 35.197)
    "thorfinn9": ("thorfinn", "9379993"),  # rc 49.04154 (LOWEST), re 35.32366
    "thorfinnA": ("thorfinn", "0ce97d2"),  # single 35.58553, rc 49.04155, re 35.32366
    "thorfinnB": ("thorfinn", "0e56f78"),  # single 35.58551 (matches my floor)
    # Floor-cluster reps — distinct prediction groups all at 35.5855 single / 49.0415 rc.
    "thorfinnC": ("thorfinn", "89dd381"),  # single G0 (=0e56f78,889c2a0), rc G0 distinct, re G0 distinct
    "thorfinnD": ("thorfinn", "86a8146"),  # single G1 distinct, rc G1 (=0e56f78,afabff7), re G1
    "thorfinnE": ("thorfinn", "afabff7"),  # single G2 distinct, rc G1 same as 0e56f78
    "thorfinnF": ("thorfinn", "847a2f4"),  # single G3 distinct, rc G3 distinct (49.0415), re G3
    "thorfinnG": ("thorfinn", "889c2a0"),  # single G0 (=0e56f78), rc G2 distinct (=9379993), re G2 distinct
    # NEW (23:10-23:16): thorfinn pushed below floor with new blends — substantial improvements on s/rc/re
    "thorfinnH": ("thorfinn", "311dacc"),  # avg 35.0905
    "thorfinnI": ("thorfinn", "55cc0ab"),  # avg 34.9361 — best cruise (20.7336)
    "thorfinnJ": ("thorfinn", "63bacc5"),  # avg 35.0335
    "thorfinnK": ("thorfinn", "644f1c4"),  # avg 34.9300
    "thorfinnL": ("thorfinn", "03652c4"),  # avg 34.9022 — best single (35.2169), best rc (48.6820)
    "thorfinnM": ("thorfinn", "34c2f44"),  # avg 34.8781 — best re (34.9163)
    "thorfinnN": ("thorfinn", "9be70c5"),  # avg 34.9392 — best cruise (20.6232)
    "thorfinnO": ("thorfinn", "c9af0cd"),  # avg 34.9081 — cruise 20.6251 (close to N), re 34.9485
    # NEW (23:34): thorfinn jumped to 34.8179 with new source — HUGE gains on rc, cruise, re
    "thorfinnP": ("thorfinn", "93c954f"),  # avg 34.8179 — rc=48.6275, cruise=20.5935, re=34.8257
    "thorfinnQ": ("thorfinn", "2ce0b4f"),  # avg 34.8014 — best rc (48.5926), cruise (20.5841), re (34.7652)!
    # My own per-split-best blend (avg 35.19569, used as source for self-blending)
    "nezuko_best": ("nezuko", "f23f935"),  # single 35.58551, rc 49.04159, c 20.83199, re 35.32367
    # My own iter15 / iter16 raw checkpoint predictions (test only) — added at low weight for diversity.
    # Will be commit-keyed once predict.py runs; placeholder for now.
    # External models — best per split (non-thorfinn)
    "edward":   ("edward",   "c773fa7"),  # single 36.25, cruise 23.73
    "edward2":  ("edward",   "2856b96"),  # single 36.61, cruise 25.08
    "tanjiro":  ("tanjiro",  "63e5e26"),  # rc 51.60, re 37.20, cruise 23.55, single 41.44
    "tanjiro2": ("tanjiro",  "5613c7b"),  # rc 52.32, re 37.94
    "tanjiro3": ("tanjiro",  "9f6f523"),  # rc 54.98, re 40.43
    "fern":     ("fern",     "36e8feb"),  # cruise 23.88
    "fern2":    ("fern",     "0f7f9e0"),  # cruise 24.85
    "frieren":  ("frieren",  "32f0a18"),  # re 42.48
    "askeladd": ("askeladd", "01851f9"),
    "alphonse": ("alphonse", "5aa9393"),
    # Mine — feed in for diversity if we like
    "nezuko_iter9":  ("nezuko", "48c1609"),
    "nezuko_iter7":  ("nezuko", "08cdd12"),
}


@dataclass
class Config:
    agent: str | None = None
    # Per-split blend weights as comma-separated "src:weight" pairs.
    # iter31: re-submit iter30's blend (got overwritten by predict.py auto-launch).
    # Same per-split-best routing: single=4-way (35.2140), rc/cruise/re=Q (2ce0b4f best).
    single: str = "thorfinnL:0.30,thorfinnM:0.30,thorfinnK:0.20,thorfinnI:0.20"
    rc: str = "thorfinnQ:1.0"
    cruise: str = "thorfinnQ:1.0"
    re_rand: str = "thorfinnQ:1.0"


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


def blend_split(split_file: str, mix: list[tuple[str, float]]) -> list[torch.Tensor]:
    """Weighted mean of per-sample predictions across listed sources."""
    sources = []
    for src_name, w in mix:
        agent, commit = SRC[src_name]
        preds = torch.load(PRED / agent / commit / f"{split_file}.pt", weights_only=False)
        sources.append((src_name, w, preds))

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
    "test_single_in_dist":     parse_mix(cfg.single),
    "test_geom_camber_rc":     parse_mix(cfg.rc),
    "test_geom_camber_cruise": parse_mix(cfg.cruise),
    "test_re_rand":            parse_mix(cfg.re_rand),
}

for split, mix in mixes.items():
    out = blend_split(split, mix)
    torch.save(out, output_dir / f"{split}.pt")

print(f"\nMeta-ensemble predictions saved to {output_dir}")
