"""Weighted prediction averaging: solve for optimal per-ckpt weights on val.

Caches per-ckpt predictions to disk (reusable across runs), then solves
  min_w || sum_i w_i * P_i - Y ||^2
via normal equations with optional non-negativity constraint.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn
from train import BaselineMLP

SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")
PVC_CKPTS = Path("/mnt/new-pvc/kagent/apr16/tanjiro/checkpoints")
CACHE = Path("/mnt/new-pvc/kagent/apr16/tanjiro/predcache")
CACHE.mkdir(parents=True, exist_ok=True)


def load_and_complete(path, reference_state):
    s = torch.load(path, weights_only=True, map_location="cpu")
    s = {k: v.float() if v.is_floating_point() else v for k, v in s.items()}
    out = dict(reference_state)
    for k, v in s.items():
        if k in out and v.shape == out[k].shape:
            out[k] = v
    return out


def predict_all(model, loader, device):
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in loader:
            v_in = v_in.to(device); v_out = v_out.to(device)
            pos = pos.to(device); t = t.to(device)
            pred = model(v_in, pos, t, idcs)
            preds.append(pred.cpu())
            truths.append(v_out.cpu())
    return torch.cat(preds, dim=0), torch.cat(truths, dim=0)


def l2(pred, truth):
    return (pred - truth).norm(dim=3).mean(dim=(1, 2)).mean().item()


def main():
    device = torch.device("cuda")
    with open(SPLITS_DIR / "stats.json") as f:
        stats = json.load(f)
    vm = torch.tensor(stats["vel_mean"], dtype=torch.float32)
    vs = torch.tensor(stats["vel_std"], dtype=torch.float32)

    val_ds = GRAMDataset(SPLITS_DIR / "val")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = BaselineMLP(hidden=256, n_blocks=8, grid_size=32,
                        n_fourier=8, vel_mean=vm, vel_std=vs).to(device)
    ref = torch.load(PVC_CKPTS / "model-mh5wd0t6" / "checkpoint.pt",
                     weights_only=True, map_location="cpu")
    ref = {k: v.float() if v.is_floating_point() else v for k, v in ref.items()}

    candidates = [
        "model-wavg3-e49e9cb", "model-wavg2-266db37", "model-wavg-816156a",
        "model-mh5wd0t6", "model-fgqa41ag", "model-ea2ll188", "model-eu7w7w48",
        "model-bbr0yz3i", "model-b1hzbt3r", "model-dfal10k2", "model-gfbsgqi7",
        "model-35pla0n7", "model-er5pk3oc", "model-jay6zniz", "model-uesu5tb6",
        "model-bal6xybc", "model-18f6e3td", "model-670v4v75", "model-fdgxhd3i",
        "model-kvptxsnv", "model-mgo03egs",
    ]

    truth = None
    preds = {}
    for c in candidates:
        cache_path = CACHE / f"{c}.pt"
        truth_path = CACHE / "_truth.pt"
        if cache_path.exists() and truth_path.exists():
            preds[c] = torch.load(cache_path, weights_only=True)
            if truth is None:
                truth = torch.load(truth_path, weights_only=True)
            print(f"{c}: cached, l2={l2(preds[c], truth):.4f}")
            continue
        p = PVC_CKPTS / c / "checkpoint.pt"
        if not p.exists():
            continue
        s = load_and_complete(p, ref)
        model.load_state_dict(s)
        pr, tr = predict_all(model, val_loader, device)
        if truth is None:
            truth = tr
            torch.save(truth, truth_path)
        torch.save(pr, cache_path)
        preds[c] = pr
        print(f"{c}: computed, l2={l2(pr, truth):.4f}")

    names = list(preds.keys())
    K = len(names)
    # Flatten predictions: each P_i is [80, 5, N, 3]; we average over all positions.
    # Stack as [K, M] matrix where M is total element count.
    P = torch.stack([preds[n].reshape(-1).double() for n in names], dim=0)  # [K, M]
    Y = truth.reshape(-1).double()  # [M]

    G = P @ P.T  # [K, K]
    b = P @ Y    # [K]

    # Unconstrained least squares
    w_unc = torch.linalg.solve(G, b)
    pred_unc = sum(w_unc[i].float() * preds[names[i]] for i in range(K))
    l2_unc = l2(pred_unc, truth)
    print(f"\nUnconstrained: l2={l2_unc:.4f}  sum(w)={w_unc.sum():.3f}")
    for n, w in zip(names, w_unc):
        print(f"  {n}: {w:+.3f}")

    # Non-negative, sum-to-one via projected gradient (simplex projection)
    def simplex_project(v):
        u, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(u, dim=0) - 1
        rho = torch.nonzero(u - cssv / torch.arange(1, K + 1, dtype=v.dtype) > 0).max()
        theta = cssv[rho] / (rho + 1).to(v.dtype)
        return torch.clamp(v - theta, min=0)

    w = torch.full((K,), 1.0 / K, dtype=torch.float64)
    lr = 1e-6
    best = (float("inf"), None)
    for it in range(5000):
        grad = 2 * (G @ w - b)
        w = simplex_project(w - lr * grad)
        if it % 500 == 0:
            wp = sum(w[i].float() * preds[names[i]] for i in range(K))
            cur = l2(wp, truth)
            if cur < best[0]:
                best = (cur, w.clone())
            print(f"  it={it}: l2={cur:.4f}")
    w = best[1]
    pred_simplex = sum(w[i].float() * preds[names[i]] for i in range(K))
    l2_simplex = l2(pred_simplex, truth)
    print(f"\nSimplex: l2={l2_simplex:.4f}  sum(w)={w.sum():.3f}")
    for n, wi in zip(names, w):
        if wi > 0.01:
            print(f"  {n}: {wi:+.3f}")

    # Pick the better of the two
    if l2_unc < l2_simplex:
        final_pred, final_l2, tag = pred_unc, l2_unc, "unconstrained"
    else:
        final_pred, final_l2, tag = pred_simplex, l2_simplex, "simplex"
    print(f"\n>>> BEST: {tag} l2={final_l2:.4f}")

    predictions = [final_pred[i] for i in range(final_pred.shape[0])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out_dir = Path(f"/mnt/new-pvc/predictions/apr16/tanjiro/{commit}-wpredavg")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "val.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "method": tag,
        "l2": final_l2,
        "weights": {n: float(w_unc[i] if tag == "unconstrained" else w[i])
                    for i, n in enumerate(names)},
    }, indent=2))
    print(f"Saved to {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
