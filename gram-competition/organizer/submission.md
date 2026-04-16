# Exporting the top kaggler to the GRaM ICLR-2026 competition

Recipe for turning the current leaderboard leader into a valid pull request
against [`gram-competition/iclr-2026`](https://github.com/gram-competition/iclr-2026).

The competition expects, under `models/<name>/`:
- `model.py` with a class whose `__init__` takes no arguments and loads its
  own state dict from `models/<name>/state_dict.pt`
- `state_dict.pt` with the trained weights
- `__init__.py` that re-exports the class
- An entry appended to the repo-level `models/__init__.py`

The call signature is
`model(t, pos, idcs_airfoil, velocity_in) -> velocity_out` — different arg
order from the kaggler training signature.

## 1. Pick the leader

```bash
kubectl exec deployment/kagent-$TAG-organizer -- \
  cat /mnt/new-pvc/predictions/$TAG/leaderboard.md
```

Grab `<agent>` and `<commit>` from rank 1. Both are needed: the commit pins
the code *and* the checkpoint that was scored.

## 2. Extract the code and checkpoint at that commit

The kaggler pushes its training code and (via `checkpoints/best.pt`) its
weights to `apr??/kaggler/<agent>` on origin. The pod's working tree may
have drifted past the scored commit, so always use `git show`:

```bash
POD=$(kubectl get pod -l kaggler=<agent> -o jsonpath='{.items[0].metadata.name}')
mkdir -p /tmp/submission/models/<name>

kubectl exec deployment/kagent-$TAG-<agent> -- bash -c "
  cd /workspace/kagent
  git show <commit>:gram-competition/kaggler/train.py
" > /tmp/submission/train.py

kubectl exec deployment/kagent-$TAG-<agent> -- bash -c "
  cd /workspace/kagent
  git show <commit>:gram-competition/kaggler/checkpoints/best.pt
" > /tmp/submission/models/<name>/state_dict.pt
```

Sanity-check that the state dict matches the code by comparing `proj_in`'s
input dim to `in_dim` in the source:

```bash
uv run --with torch --no-project python - <<'PY'
import torch
sd = torch.load("/tmp/submission/models/<name>/state_dict.pt",
                map_location="cpu", weights_only=True)
print({k: tuple(v.shape) for k, v in sd.items() if "proj_in" in k})
PY
```

If the shapes disagree, the kaggler has uncommitted code — the training
branch was ahead of what got scored. Either pick an earlier commit whose
`best.pt` matches the committed code, or re-run training on the pushed
commit to regenerate the checkpoint.

## 3. Package the model

Create `models/<name>/model.py`. Structure:

- Keep every helper (`ResBlock`, `UNet3D`, spatial mixers, etc.) the
  training code defined.
- Rename the core module to match the checkpoint keys. If the training
  module was `VoxelResidualModel`, keep that class name so `load_state_dict`
  matches without `strict=False` hacks.
- Replace constructor arguments that came from `stats.json` (e.g.
  `vel_mean`, `vel_std`) with a `register_buffer` of the right shape
  initialised to zeros/ones. The real values ride along in the state dict.
- Add the competition-facing wrapper:

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = VoxelResidualModel(hidden=..., voxel_res=..., ...)
        sd = torch.load(os.path.join("models", "<name>", "state_dict.pt"),
                        map_location="cpu")
        self.net.load_state_dict(sd)

    def forward(self, t, pos, idcs_airfoil, velocity_in):
        # Compute any features the kaggler pipeline precomputed on disk.
        B = pos.shape[0]
        sdf = torch.stack([_compute_sdf(pos[b], idcs_airfoil[b]) for b in range(B)])
        return self.net(velocity_in, pos, idcs_airfoil, sdf)
```

`__init__.py` just re-exports it:

```python
from .model import Model as <Name>
```

## 4. Smoke-test locally

```bash
cd /tmp/submission && uv run --with torch --no-project python - <<'PY'
import torch
from models.<name> import <Name> as Model
m = Model().eval()
B, N = 2, 5_000  # full-size (95 × 100k) is slow without a GPU
t = torch.rand(B, 10)
pos = torch.rand(B, N, 3)
idcs = [torch.randint(N, (1234,)), torch.randint(N, (2000,))]
v_in = torch.rand(B, 5, N, 3)
with torch.no_grad():
    out = m(t, pos, idcs, v_in)
assert out.shape == (B, 5, N, 3)
print("OK", out.shape, "vel_mean:", m.net.vel_mean.flatten().tolist())
PY
```

The `vel_mean` should match the training stats (~`[35.6, 0.5, 1.9]` on
warped-ifw). If it's zeros, the state dict didn't load properly.

## 5. Fork, add, and PR

```bash
cd /tmp && gh repo fork gram-competition/iclr-2026 --clone --remote=false
cd iclr-2026 && git checkout -b <name>-submission
cp -r /tmp/submission/models/<name> models/<name>

# Append the import — leave the reference MLP line in place.
printf "from .<name> import <Name>\n" >> models/__init__.py

git add models/<name> models/__init__.py
git commit -m "Add <name> submission"
git push -u origin <name>-submission

gh pr create \
  --repo gram-competition/iclr-2026 \
  --base main --head tcapelle:<name>-submission \
  --title "Add <name> submission" \
  --body "..."
```

## Gotchas worth remembering

- **The checkpoint travels with the commit.** Kagglers run the training
  loop in the background, so the working tree can race ahead of the
  checkpoint committed to git. Always pull via `git show <commit>:...`.
- **Features not in the competition signature must be computed in
  `forward`.** SDF, k-NN graphs, augmentation statistics — none of that is
  available at evaluation time.
- **Hyperparameters must be hard-coded** in the submission constructor.
  Grep the kaggler's `Config` dataclass for the values used in the
  top-scoring run (W&B config is a cross-check).
- **File size < 100 MB** per the GitHub limit. The current Voxel-UNet
  checkpoint is 31 MB; larger models need Git LFS or an external download
  link.
- **Don't move `main.py` or the reference `mlp` submission** — the repo
  ships them as organiser-side tooling and they should stay untouched.
