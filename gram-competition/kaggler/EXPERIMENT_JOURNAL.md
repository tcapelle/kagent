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

### 2026-04-16 — v1: residual + normalize + no-slip BC, wider ResMLP (384/8)

- **Hypothesis:** the biggest baseline wins are physical priors — normalize velocity with stats, predict a delta off `v_in[-1]` (persistence residual), and zero predictions at airfoil surface (no-slip BC). Still pointwise but with a much wider net.
- **Change:** `model.py` adds `FlowMLP` (hidden=384, 8 ResBlocks, expand×4, zero-init output so training starts at persistence). `train.py` wires through the new model; `predict.py` refactored to import from `model.py`.
- **Result:** see W&B run `askeladd/v1-residual-noslip` (in progress).
- **Notes:** FlowMLP is still 100%-pointwise — no spatial interactions. Next step is a KNN-based GNN (v2 scaffolded in `model.py::FlowGNN`, KNN via chunked cdist since torch-cluster is unavailable).

