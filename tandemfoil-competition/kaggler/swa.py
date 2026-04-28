"""Average model weights across multiple checkpoints (SWA-style)."""

import shutil
from pathlib import Path

import torch
import yaml


import simple_parsing as sp
from dataclasses import dataclass, field

@dataclass
class Args:
    out: str = "models/model-swa"
    ckpts: str = "models/model-3fsnww54/checkpoint.pt,models/model-zvhh3d5w/checkpoint.pt,models/model-lhl3kzpw/checkpoint.pt,models/model-ti5dv6wo/checkpoint.pt,models/model-cxipgis3/checkpoint.pt"

args = sp.parse(Args)
CKPTS = [c.strip() for c in args.ckpts.split(",") if c.strip()]

states = []
configs = []
for c in CKPTS:
    if not Path(c).exists():
        print(f"skip {c}")
        continue
    s = torch.load(c, map_location="cpu", weights_only=True)
    states.append(s)
    cfg_path = Path(c).parent / "config.yaml"
    with open(cfg_path) as f:
        configs.append(yaml.safe_load(f))
    print(f"loaded {c}")

# all configs should match
for cfg in configs[1:]:
    assert cfg == configs[0], "configs must match"

avg_state = {}
for k in states[0]:
    avg_state[k] = sum(s[k].float() for s in states) / len(states)
    avg_state[k] = avg_state[k].to(states[0][k].dtype)

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(avg_state, out_dir / "checkpoint.pt")
shutil.copy(Path(CKPTS[0]).parent / "config.yaml", out_dir / "config.yaml")
print(f"SWA saved to {out_dir / 'checkpoint.pt'}")
