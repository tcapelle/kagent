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
    # Leader: best on every test split (single 35.60, rc 49.07, cruise 20.88, re 35.33)
    "thorfinn":  ("thorfinn", "fc1227e"),  # avg 35.22
    "thorfinn2": ("thorfinn", "8103189"),  # avg 35.23
    "thorfinn3": ("thorfinn", "467ecba"),  # avg 35.24
    "thorfinn4": ("thorfinn", "22eaa7e"),  # avg 35.26
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
    # Default: thorfinn-dominant blend with small diversity from per-split runners-up.
    single: str = "thorfinn:0.85,edward:0.10,thorfinn2:0.05"
    rc: str = "thorfinn:0.85,tanjiro:0.10,thorfinn2:0.05"
    cruise: str = "thorfinn:0.85,tanjiro:0.05,fern:0.05,thorfinn2:0.05"
    re_rand: str = "thorfinn:0.85,tanjiro:0.10,thorfinn2:0.05"


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
