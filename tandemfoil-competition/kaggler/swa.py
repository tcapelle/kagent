"""Average model weights across multiple checkpoints (SWA-style)."""

import shutil
from pathlib import Path

import torch
import yaml


CKPTS = [
    "models/model-3fsnww54/checkpoint.pt",  # iter16 (vol=50k)
    "models/model-zvhh3d5w/checkpoint.pt",  # iter17 (vol=80k)
    "models/model-lhl3kzpw/checkpoint.pt",  # iter18 (full mesh)
    "models/model-ti5dv6wo/checkpoint.pt",  # iter19 (full mesh fine-tune)
    "models/model-jx7l45ya/checkpoint.pt",  # iter20 (placeholder; replaced below)
]

# discover the latest model dir for iter20 (placeholder above may not exist)
import sys, os
candidates = sorted(Path("models").glob("model-*/checkpoint.pt"), key=lambda p: p.stat().st_mtime)
# take the most recently created checkpoint as iter20
latest = candidates[-1]
CKPTS[-1] = str(latest)
print(f"iter20 latest: {latest}")

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

out_dir = Path("models/model-swa")
out_dir.mkdir(parents=True, exist_ok=True)
torch.save(avg_state, out_dir / "checkpoint.pt")
shutil.copy(Path(CKPTS[0]).parent / "config.yaml", out_dir / "config.yaml")
print(f"SWA saved to {out_dir / 'checkpoint.pt'}")
