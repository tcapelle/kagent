# Experiment Journal

Record every meaningful experiment here. Read this before starting a new
iteration so you don't repeat work or make the same mistake twice.

## Format

One entry per experiment, newest at the top:

```
### YYYY-MM-DD — short-title
- **Hypothesis:** what you expected to improve and why
- **Change:** what you actually modified (file + 1-line summary)
- **Result:** val/l2_error (best), train loss, epoch count, VRAM peak
- **Verdict:** kept / discarded — one sentence on why
- **Notes:** surprises, failure modes, ideas to try next
```

Keep entries short. Link W&B run URLs when useful.

---

## Entries

### Baselines (no training)

| Baseline | val/l2_error |
|---|---|
| Persistence (copy `v_in[-1]`) | **1.75** |
| Time-mean of inputs | **1.42** |
| Linear extrapolation `v[-1]+k*(v[-1]-v[-2])` | 5.26 (unstable, oscillatory flow) |

So *just predicting the mean of the input frames* beats a vanilla "copy last frame" prior. The time-mean is the bar to beat with trained models. dt between consecutive frames is ~0.001 — the future is close to the past.

### 2026-04-16 — v2: spatial KNN-GNN with time-mean prior

- **Hypothesis:** v1 is pointwise-only so it can't capture spatial structure; adding KNN-graph message passing should let the model see local neighborhoods (vortices, wake structure). Also switch the residual prior from `v_in[-1]` to `mean(v_in)` since the mean-of-inputs baseline (L2 1.42) already beats persistence (1.75).
- **Change:** `model.py::FlowGNN` — pre/post pointwise ResBlocks with 3 KNN EdgeConv layers (k=16, max-agg, relative-pos encoded messages) in the middle. Added chunked cdist KNN (`torch-cluster` not installed) and a per-geometry in-process KNN cache keyed by a pos fingerprint (162 unique geometries → recompute once per run instead of every forward). hidden=256.
- **Result:** 6 epochs (~31 min). Per-epoch val/l2: 1.4748 → 1.4741 → **1.2926** → 1.3125 → 1.2376 → **1.2140**. Epoch 1 357s (KNN cache warm), epochs 2–6 ~309s each. Train loss 4.42 → 3.17 (still descending). Peak VRAM 49 GB. Predictions saved to `/mnt/new-pvc/predictions/apr16/askeladd/686baad/val.pt`.
- **Verdict:** kept. Huge win over v1 (1.4694) and time-mean baseline (1.42). Spatial message-passing matters a lot — the first 2 epochs looked like a plateau but the model sharply improved once it "understood" the graph.
- **Notes:** clearly not converged — train & val both descending linearly at timeout. Next: run the same config with a longer time budget (v3) to ride out the descent.

### 2026-04-16 — v1: residual + normalize + no-slip BC, wider ResMLP (384/8)

- **Hypothesis:** the biggest baseline wins are physical priors — normalize velocity with stats, predict a delta off `v_in[-1]` (persistence residual), and zero predictions at airfoil surface (no-slip BC). Still pointwise but with a much wider net.
- **Change:** `model.py` adds `FlowMLP` (hidden=384, 8 ResBlocks, expand×4, zero-init output so training starts at persistence). `train.py` wires through the new model; `predict.py` refactored to import from `model.py`.
- **Result:** see W&B run `askeladd/v1-residual-noslip` (in progress).
- **Notes:** FlowMLP is still 100%-pointwise — no spatial interactions. Next step is a KNN-based GNN (v2 scaffolded in `model.py::FlowGNN`, KNN via chunked cdist since torch-cluster is unavailable).

