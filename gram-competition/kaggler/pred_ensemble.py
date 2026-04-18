"""Prediction-averaging ensemble: average predictions from multiple ckpts rather than weights.

Unlike weight averaging, this works across any architecture versions since we only combine
the final outputs. Typically stronger because ckpts in different loss basins can complement.
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


def l2_score(pred, truth):
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

    # All reasonable candidates — we'll pick by val score then greedy-combine
    candidates = [
        # wavg meta-ckpts
        "model-wavg3-e49e9cb",  # exp59, 0.8774
        "model-wavg2-266db37",  # exp58, 0.8786
        "model-wavg-816156a",   # exp57, 0.8789
        # 243-key (exp44-56 era)
        "model-mh5wd0t6",   # exp45
        "model-fgqa41ag",   # exp56 E1
        "model-ea2ll188",
        "model-eu7w7w48",
        "model-bbr0yz3i",
        "model-b1hzbt3r",
        "model-dfal10k2",
        "model-gfbsgqi7",
        "model-35pla0n7",
        # 241-key (pre-pos-offset)
        "model-er5pk3oc",
        "model-jay6zniz",
        "model-uesu5tb6",
        # 237-key (pre-knn)
        "model-bal6xybc",
        "model-18f6e3td",
        "model-670v4v75",
        "model-fdgxhd3i",
        "model-kvptxsnv",
        "model-mgo03egs",
    ]

    truth = None
    preds = {}
    scores = {}
    for c in candidates:
        p = PVC_CKPTS / c / "checkpoint.pt"
        if not p.exists():
            continue
        s = load_and_complete(p, ref)
        model.load_state_dict(s)
        pr, tr = predict_all(model, val_loader, device)
        if truth is None:
            truth = tr
        preds[c] = pr
        scores[c] = l2_score(pr, truth)
        print(f"{c}: l2={scores[c]:.4f}")

    sorted_names = sorted(scores, key=scores.get)
    print(f"\nBest single: {sorted_names[0]} ({scores[sorted_names[0]]:.4f})")

    # Greedy: start with best single, add ckpt whose mean-pred most reduces val l2
    print("\n--- greedy prediction ensemble (uniform) ---")
    chosen = [sorted_names[0]]
    remaining = [n for n in sorted_names if n not in chosen]
    best_l2 = scores[sorted_names[0]]
    while remaining:
        next_cand, next_l2 = None, best_l2
        for cand in remaining:
            avg = sum(preds[n] for n in chosen + [cand]) / (len(chosen) + 1)
            l2 = l2_score(avg, truth)
            if l2 < next_l2:
                next_l2 = l2
                next_cand = cand
        if next_cand is None:
            break
        chosen.append(next_cand)
        remaining.remove(next_cand)
        best_l2 = next_l2
        print(f"  + {next_cand} → {best_l2:.4f}  (size {len(chosen)})")

    print(f"\n>>> BEST uniform: l2={best_l2:.4f}  {chosen}")

    # Save ensemble predictions under new commit dir
    avg = sum(preds[n] for n in chosen) / len(chosen)
    predictions = [avg[i] for i in range(avg.shape[0])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out_dir = Path(f"/mnt/new-pvc/predictions/apr16/tanjiro/{commit}-predavg")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "val.pt")
    print(f"Saved ensemble preds to {out_dir / 'val.pt'}")

    # Save chosen list + scores
    (out_dir / "meta.json").write_text(json.dumps({
        "chosen": chosen,
        "best_l2": best_l2,
        "all_scores": scores,
    }, indent=2))


if __name__ == "__main__":
    main()
