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

### 2026-04-16 — v1 residual ResMLP + no-slip + normalized loss
- **Hypothesis:** baseline predicts absolute velocity from scratch — a residual around `velocity_in[-1]` is a much stronger starting point because frame-to-frame changes are small relative to the mean flow (~35 m/s mean Ux). Hard no-slip BC guarantees zero at airfoil. Normalized MSE loss stops the ~20 m/s Ux std from dominating the gradient.
- **Change:** `train.py` — `ResidualPointMLP` (hidden=384, n_blocks=8). Input features: normalized velocity_in (15) + pos (3) + airfoil mask (1) = 19. Output: delta in normalized space; denormalize and add to last input frame. Zero-init last linear → starts at exact persistence. Post-process no-slip mask. Loss is MSE on (pred - gt)/vel_std. Grad clip 1.0.
- **Result:** val/l2 = **1.3200** at epoch 21, mae (Ux,Uy,Uz)=(0.884, 0.375, 0.641). 26 epochs in 25 min, ~55s/epoch, peak 8.1 GB. 4.75M params. W&B run `ajszccxm`. Commit `adeebc6`.
- **Verdict:** kept — clean win vs baseline ~1.76 on mar29 val; zero-init residual made training stable from epoch 1 (epoch 1 already 1.59, below baseline's final).
- **Notes:** Val oscillates 0.05 between epochs — batch_size=1 is noisy. Loss kept dropping at end, so more epochs likely helps. predict.py broke because importing train.py triggered `sp.parse(sys.argv)` on predict's args; fixed by wrapping train.py body in `main()` + `if __name__ == "__main__":`. Per-point MLP — no spatial interaction. Next (v2): voxel-UNet spatial module.

