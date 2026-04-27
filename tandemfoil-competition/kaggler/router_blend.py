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
ITER1 = "790ca24"  # iter1 (single=40.73, cruise=28.39)
ITER3 = "165f5b0"  # iter3 (= iter2 numbers, same predictions)


@dataclass
class Config:
    agent: str | None = None
    single_blend_w: float = 0.0   # 0=ITER2 only, 1=GOLD only, 0.5=avg
    cruise_blend_w: float = 0.0   # same
    extra_w_single: float = 0.0   # weight on ITER1 for single (subtracts from main pair)
    extra_w_cruise: float = 0.0   # weight on ITER1 for cruise
    rc_blend_w: float = 1.0       # 1=gold only (default)
    re_blend_w: float = 1.0       # 1=gold only
    extra_w_rc: float = 0.0       # weight on ITER1 for rc
    extra_w_re: float = 0.0       # weight on ITER1 for re_rand
    use_iter1_main: bool = False  # if true, ITER2 -> ITER1 in blend


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


def blend(split: str, w_gold: float, w_extra_iter1: float = 0.0):
    """Blend = w_gold * GOLD + w_extra * ITER1 + (1 - w_gold - w_extra) * ITER2.

    If --use_iter1_main, ITER1 takes the role of ITER2 (so the main blend partner
    of GOLD is iter1 instead of iter2).
    """
    a = torch.load(PREDICTIONS_DIR / agent_name / GOLD / f"{split}.pt", weights_only=False)
    main = ITER1 if cfg.use_iter1_main else ITER2
    b = torch.load(PREDICTIONS_DIR / agent_name / main / f"{split}.pt", weights_only=False)
    if w_extra_iter1 > 0:
        third = ITER2 if cfg.use_iter1_main else ITER1
        c = torch.load(PREDICTIONS_DIR / agent_name / third / f"{split}.pt", weights_only=False)
        w_b = 1.0 - w_gold - w_extra_iter1
        out = [w_gold * ai + w_b * bi + w_extra_iter1 * ci
               for ai, bi, ci in zip(a, b, c)]
        print(f"  blend {split}: {w_gold:.2f}*{GOLD} + {w_b:.2f}*{main} + {w_extra_iter1:.2f}*{third}")
    else:
        out = [w_gold * ai + (1 - w_gold) * bi for ai, bi in zip(a, b)]
        print(f"  blend {split}: {w_gold:.2f}*{GOLD} + {1-w_gold:.2f}*{main}")
    torch.save(out, output_dir / f"{split}.pt")


# rc and re_rand: blend with iter2 too (default w=1.0 = gold only)
blend("test_geom_camber_rc", cfg.rc_blend_w, cfg.extra_w_rc)
blend("test_re_rand", cfg.re_blend_w, cfg.extra_w_re)

# single and cruise: blend
blend("test_single_in_dist", cfg.single_blend_w, cfg.extra_w_single)
blend("test_geom_camber_cruise", cfg.cruise_blend_w, cfg.extra_w_cruise)

print(f"\nBlend predictions saved to {output_dir}")
