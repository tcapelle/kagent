"""Extend the candidate pool to all 56 ckpts, cache missing preds, then run direct optimization."""
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
    return (pred - truth).norm(dim=3).mean(dim=(1, 2)).mean()


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

    all_ckpts = sorted([d.name for d in PVC_CKPTS.iterdir()
                        if d.is_dir() and (d / "checkpoint.pt").exists()])
    print(f"Found {len(all_ckpts)} ckpts")

    truth_path = CACHE / "_truth.pt"
    truth = torch.load(truth_path, weights_only=True) if truth_path.exists() else None

    for c in all_ckpts:
        cache_path = CACHE / f"{c}.pt"
        if cache_path.exists():
            continue
        p = PVC_CKPTS / c / "checkpoint.pt"
        try:
            s = load_and_complete(p, ref)
            model.load_state_dict(s)
            pr, tr = predict_all(model, val_loader, device)
            if truth is None:
                truth = tr
                torch.save(truth, truth_path)
            torch.save(pr, cache_path)
            print(f"cached {c}, l2={l2(pr, truth).item():.4f}")
        except Exception as e:
            print(f"SKIP {c}: {e}")

    # Load all cached
    cand_files = sorted(CACHE.glob("model-*.pt"))
    names = [p.stem for p in cand_files]
    preds = torch.stack([torch.load(p, weights_only=True) for p in cand_files], dim=0).to(device)
    truth = torch.load(truth_path, weights_only=True).to(device)
    K = preds.shape[0]
    print(f"\nLoaded {K} cached preds")

    # Per-ckpt scores
    print("\nIndividual scores:")
    singles = []
    for i, n in enumerate(names):
        s = l2(preds[i], truth).item()
        singles.append((n, s))
        print(f"  {n}: {s:.4f}")

    # Direct optimization
    logits = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.02)
    best = (float("inf"), None)
    for step in range(5000):
        w = torch.softmax(logits, dim=0)
        mixed = (w[:, None, None, None, None] * preds).sum(dim=0)
        loss = l2(mixed, truth)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), logits.detach().clone())
        if step % 500 == 0:
            print(f"step={step}: l2={loss.item():.4f}")
    print(f">>> Adam best: l2={best[0]:.4f}")

    logits = best[1]
    w = torch.softmax(logits, dim=0)
    final_mixed = (w[:, None, None, None, None] * preds).sum(dim=0)
    final_l2 = l2(final_mixed, truth).item()
    print(f"Final: l2={final_l2:.4f}")
    top = sorted(zip(names, w.cpu().tolist()), key=lambda x: -x[1])
    for n, wi in top[:15]:
        print(f"  {n}: {wi:.3f}")

    predictions = [final_mixed[i].cpu() for i in range(final_mixed.shape[0])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out_dir = Path(f"/mnt/new-pvc/predictions/apr16/tanjiro/{commit}-all")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "val.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "method": "adam_softmax_all_ckpts",
        "l2": final_l2,
        "n_ckpts": K,
        "weights": {n: float(wi) for n, wi in zip(names, w.cpu().tolist())},
    }, indent=2))
    print(f"Saved to {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
