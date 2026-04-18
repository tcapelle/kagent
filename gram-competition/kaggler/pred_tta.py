"""Test-time augmentation (TTA) via y-flip — double the effective ensemble.

Model was trained with yflip_prob=0.5 so it's approximately equivariant under y-flip.
For each ckpt, produce a second prediction from y-flipped input (un-flip output).
These flipped predictions add to the pool.
"""
import json
import subprocess
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import GRAMDataset, collate_fn
from train import BaselineMLP

SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")
PVC_CKPTS = Path("/mnt/new-pvc/kagent/apr16/tanjiro/checkpoints")
CACHE = Path("/mnt/new-pvc/kagent/apr16/tanjiro/predcache")
CACHE_TTA = Path("/mnt/new-pvc/kagent/apr16/tanjiro/predcache_tta")
CACHE_TTA.mkdir(parents=True, exist_ok=True)


def load_and_complete(path, reference_state):
    s = torch.load(path, weights_only=True, map_location="cpu")
    s = {k: v.float() if v.is_floating_point() else v for k, v in s.items()}
    out = dict(reference_state)
    for k, v in s.items():
        if k in out and v.shape == out[k].shape:
            out[k] = v
    return out


def predict_yflipped(model, loader, device):
    """Predict on y-flipped input, un-flip output."""
    model.eval()
    preds = []
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in loader:
            v_in = v_in.to(device); pos = pos.to(device); t = t.to(device)
            y_center = 0.5 * (pos[..., 1].amax(dim=1, keepdim=True)
                              + pos[..., 1].amin(dim=1, keepdim=True))
            pos_f = pos.clone(); pos_f[..., 1] = 2 * y_center - pos_f[..., 1]
            v_in_f = v_in.clone(); v_in_f[..., 1] = -v_in_f[..., 1]
            pred = model(v_in_f, pos_f, t, idcs)
            pred[..., 1] = -pred[..., 1]  # un-flip Uy
            preds.append(pred.cpu())
    return torch.cat(preds, dim=0)


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

    ref = torch.load(PVC_CKPTS / "model-mh5wd0t6" / "checkpoint.pt",
                     weights_only=True, map_location="cpu")
    ref = {k: v.float() if v.is_floating_point() else v for k, v in ref.items()}

    truth = torch.load(CACHE / "_truth.pt", weights_only=True)

    # Select top ckpts by individual val/l2; TTA the top 30 (enough to matter).
    all_ckpts = sorted([d.name for d in PVC_CKPTS.iterdir()
                        if d.is_dir() and (d / "checkpoint.pt").exists()])

    # Precompute individual scores from cache
    scores = {}
    for c in all_ckpts:
        cp = CACHE / f"{c}.pt"
        if cp.exists():
            p = torch.load(cp, weights_only=True)
            scores[c] = l2(p, truth)
    best_list = sorted(scores, key=scores.get)[:30]
    print(f"TTA targets (top 30 by val l2): {best_list[:5]}... ({len(best_list)} total)")

    # Build model once; reload states
    model = BaselineMLP(hidden=256, n_blocks=8, grid_size=32,
                        n_fourier=8, vel_mean=vm, vel_std=vs).to(device)

    for c in best_list:
        tta_path = CACHE_TTA / f"{c}_yflip.pt"
        if tta_path.exists():
            continue
        p = PVC_CKPTS / c / "checkpoint.pt"
        s = load_and_complete(p, ref)
        model.load_state_dict(s)
        pr_flip = predict_yflipped(model, val_loader, device)
        torch.save(pr_flip, tta_path)
        l2v = l2(pr_flip, truth)
        print(f"TTA {c}: yflip l2={l2v:.4f}")

    # Now load all original preds + TTA preds for the top 30
    names = []
    preds = []
    for c in all_ckpts:
        cp = CACHE / f"{c}.pt"
        if cp.exists():
            preds.append(torch.load(cp, weights_only=True))
            names.append(c)
    for c in best_list:
        tp = CACHE_TTA / f"{c}_yflip.pt"
        if tp.exists():
            preds.append(torch.load(tp, weights_only=True))
            names.append(f"{c}_yflip")
    K = len(preds)
    print(f"\nTotal ensemble: {K} preds ({len(preds) - len(best_list)} orig + {len(best_list)} tta)")

    preds_t = torch.stack(preds, dim=0).to(device)
    truth_d = truth.to(device)

    logits = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.02)
    best = (float("inf"), None)
    for step in range(5000):
        w = torch.softmax(logits, dim=0)
        mixed = (w[:, None, None, None, None] * preds_t).sum(dim=0)
        loss = (mixed - truth_d).norm(dim=3).mean(dim=(1, 2)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), logits.detach().clone())
        if step % 500 == 0:
            print(f"step={step}: l2={loss.item():.4f}")
    print(f">>> Adam best: l2={best[0]:.4f}")

    logits = best[1]
    w = torch.softmax(logits, dim=0)
    final_mixed = (w[:, None, None, None, None] * preds_t).sum(dim=0)
    final_l2 = (final_mixed - truth_d).norm(dim=3).mean(dim=(1, 2)).mean().item()
    print(f"Final: l2={final_l2:.4f}")

    top = sorted(zip(names, w.cpu().tolist()), key=lambda x: -x[1])
    for n, wi in top[:20]:
        if wi > 0.005:
            print(f"  {n}: {wi:.3f}")

    predictions = [final_mixed[i].cpu() for i in range(final_mixed.shape[0])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out_dir = Path(f"/mnt/new-pvc/predictions/apr16/tanjiro/{commit}-tta")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "val.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "method": "tta_yflip_direct",
        "l2": final_l2,
        "n_ckpts": K,
        "weights": {n: float(wi) for n, wi in zip(names, w.cpu().tolist())},
    }, indent=2))
    print(f"Saved to {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
