"""Flow field visualization for model predictions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def visualize(model, val_ds, stats, device, n_samples=4, out_dir=None):
    """Generate GT vs prediction plots: velocity magnitude and pressure.

    Layout: 2 rows (velocity, pressure) x 3 cols (GT, Predicted, Error).
    """
    out_dir = Path(out_dir or "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = []

    for idx in range(min(n_samples, len(val_ds))):
        x, y_true, is_surface = val_ds[idx]

        with torch.no_grad():
            x_norm = (x.unsqueeze(0).to(device) - stats["x_mean"]) / stats["x_std"]
            pred_norm = model({"x": x_norm})["preds"]
            y_pred = (pred_norm * stats["y_std"] + stats["y_mean"]).squeeze(0).cpu()

        pos = x[:, :2].numpy()
        yt, yp = y_true.numpy(), y_pred.numpy()
        surf_pos = pos[is_surface.numpy()]

        # View bounds around the airfoil
        x_lo, x_hi = -1.0, 2.0
        y_lo = max(0.0, pos[:, 1].min())
        y_hi = min(surf_pos[:, 1].mean() + 3.0, pos[:, 1].max())
        near = (
            (pos[:, 0] >= x_lo) & (pos[:, 0] <= x_hi)
            & (pos[:, 1] >= y_lo) & (pos[:, 1] <= y_hi)
        )
        px, py = pos[near, 0], pos[near, 1]

        gt_vmag = np.sqrt(yt[near, 0]**2 + yt[near, 1]**2)
        pr_vmag = np.sqrt(yp[near, 0]**2 + yp[near, 1]**2)
        gt_p, pr_p = yt[near, 2], yp[near, 2]

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        fig.suptitle(f"Sample {idx}")

        def scatter(ax, vals, cmap="viridis", vmin=None, vmax=None):
            sc = ax.scatter(px, py, c=vals, s=0.5, cmap=cmap,
                            vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
            fig.colorbar(sc, ax=ax)
            ax.set_aspect("equal")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.plot(surf_pos[:, 0], surf_pos[:, 1], "k.", markersize=0.3)

        # Row 0: velocity magnitude
        vmin_v, vmax_v = gt_vmag.min(), gt_vmag.max()
        scatter(axes[0, 0], gt_vmag, vmin=vmin_v, vmax=vmax_v)
        axes[0, 0].set_title("|U| GT")
        scatter(axes[0, 1], pr_vmag, vmin=vmin_v, vmax=vmax_v)
        axes[0, 1].set_title("|U| Pred")
        err_v = gt_vmag - pr_vmag
        evm = max(abs(err_v.min()), abs(err_v.max()), 1e-6)
        scatter(axes[0, 2], err_v, cmap="RdBu_r", vmin=-evm, vmax=evm)
        axes[0, 2].set_title("|U| Error")

        # Row 1: pressure
        vmin_p, vmax_p = gt_p.min(), gt_p.max()
        scatter(axes[1, 0], gt_p, cmap="RdBu_r", vmin=vmin_p, vmax=vmax_p)
        axes[1, 0].set_title("p GT")
        scatter(axes[1, 1], pr_p, cmap="RdBu_r", vmin=vmin_p, vmax=vmax_p)
        axes[1, 1].set_title("p Pred")
        err_p = gt_p - pr_p
        epm = max(abs(err_p.min()), abs(err_p.max()), 1e-6)
        scatter(axes[1, 2], err_p, cmap="RdBu_r", vmin=-epm, vmax=epm)
        axes[1, 2].set_title("p Error")

        plt.tight_layout()
        path = out_dir / f"val_{idx}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved.append(path)

    return saved
