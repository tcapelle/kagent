"""Blend stored prediction files at the tensor level (no model inference).

Useful when we want to combine predictions from different commits without
having to reproduce the original model — just average their stored .pt outputs.

Usage:
  python router_blend.py --agent thorfinn --single_blend_w 0.5
"""

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
import torch

RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")
PREDICTIONS_DIR = Path(f"/mnt/new-pvc/predictions/{RESEARCH_TAG}")

GOLD = "4318185"   # warm-start original (rc=61.70, re=51.05)
ITER2 = "649c01d"  # iter2 chain (single=40.28, cruise=27.95)


@dataclass
class Config:
    agent: str | None = None
    single_blend_w: float = 0.0   # 0=ITER2 only, 1=GOLD only, 0.5=avg
    cruise_blend_w: float = 0.0   # same


cfg = sp.parse(Config)

agent_name = cfg.agent or "unknown"
commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip() or "unknown"
output_dir = PREDICTIONS_DIR / agent_name / commit
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Output: {output_dir}")
print(f"single_blend_w={cfg.single_blend_w}, cruise_blend_w={cfg.cruise_blend_w}")


def blend(split: str, w: float):
    a = torch.load(PREDICTIONS_DIR / agent_name / GOLD / f"{split}.pt", weights_only=False)
    b = torch.load(PREDICTIONS_DIR / agent_name / ITER2 / f"{split}.pt", weights_only=False)
    out = [w * ai + (1 - w) * bi for ai, bi in zip(a, b)]
    torch.save(out, output_dir / f"{split}.pt")
    print(f"  blend {split}: {w:.2f}*{GOLD} + {1-w:.2f}*{ITER2}")


# rc and re_rand: copy gold (warm-start original)
for split in ["test_geom_camber_rc", "test_re_rand"]:
    src = PREDICTIONS_DIR / agent_name / GOLD / f"{split}.pt"
    dst = output_dir / f"{split}.pt"
    shutil.copy(src, dst)
    print(f"  copied {GOLD}/{split}.pt")

# single and cruise: blend
blend("test_single_in_dist", cfg.single_blend_w)
blend("test_geom_camber_cruise", cfg.cruise_blend_w)

print(f"\nBlend predictions saved to {output_dir}")
