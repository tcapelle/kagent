"""Find best weight-average combination of candidate checkpoints."""
import json
from itertools import combinations
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import GRAMDataset, collate_fn
from train import BaselineMLP

SPLITS_DIR = Path("/mnt/new-pvc/datasets/gram/splits")
PVC_CKPTS = Path("/mnt/new-pvc/kagent/apr16/tanjiro/checkpoints")


def load_state(path):
    s = torch.load(path, weights_only=True, map_location="cpu")
    return {k: v.float() if v.is_floating_point() else v for k, v in s.items()}


def avg_states(states):
    n = len(states)
    out = {}
    for k in states[0].keys():
        if not states[0][k].is_floating_point():
            out[k] = states[0][k]
        else:
            out[k] = sum(s[k] for s in states) / n
    return out


def validate(model, loader, device):
    model.eval()
    total_l2 = 0.0
    total_mae = torch.zeros(3, device=device, dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in loader:
            v_in = v_in.to(device); v_out = v_out.to(device)
            pos = pos.to(device); t = t.to(device)
            pred = model(v_in, pos, t, idcs)
            total_l2 += (pred - v_out).norm(dim=3).mean(dim=(1, 2)).sum().item()
            total_mae += (pred - v_out).abs().mean(dim=(1, 2)).double().sum(dim=0)
            n += v_in.shape[0]
    return total_l2 / n, (total_mae / n).tolist()


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

    # All 243-key checkpoints on PVC
    candidates = [
        "model-mh5wd0t6",  # exp45, 0.8801
        "model-fgqa41ag",  # exp56 E1, 0.8802
        "model-ea2ll188",  # 0.8806
        "model-eu7w7w48",  # exp44, 0.8805
        "model-bbr0yz3i",
        "model-b1hzbt3r",
        "model-dfal10k2",
        "model-gfbsgqi7",
        "model-35pla0n7",
    ]
    scores = {}
    states = {}
    for c in candidates:
        p = PVC_CKPTS / c / "checkpoint.pt"
        if not p.exists(): continue
        s = load_state(p)
        model.load_state_dict(s)
        l2, _ = validate(model, val_loader, device)
        scores[c] = l2
        states[c] = s
        print(f"{c}: l2={l2:.4f}")

    # Sort by score
    sorted_names = sorted(scores, key=scores.get)
    print(f"\nBest single: {sorted_names[0]} ({scores[sorted_names[0]]:.4f})")

    # Try best-K for K=2..6
    best_avg = (scores[sorted_names[0]], [sorted_names[0]])
    for K in range(2, min(7, len(sorted_names) + 1)):
        top_k = sorted_names[:K]
        avg = avg_states([states[n] for n in top_k])
        model.load_state_dict(avg)
        l2, _ = validate(model, val_loader, device)
        print(f"AVG(top-{K}): {l2:.4f}  {top_k}")
        if l2 < best_avg[0]:
            best_avg = (l2, top_k)

    # Also try greedy: start with best, add ckpt that most improves
    print("\n--- greedy ensemble ---")
    chosen = [sorted_names[0]]
    remaining = sorted_names[1:]
    current_l2 = scores[sorted_names[0]]
    while remaining:
        best_next = None
        best_next_l2 = current_l2
        for cand in remaining:
            test = chosen + [cand]
            avg = avg_states([states[n] for n in test])
            model.load_state_dict(avg)
            l2, _ = validate(model, val_loader, device)
            if l2 < best_next_l2:
                best_next_l2 = l2
                best_next = cand
        if best_next is None:
            break
        chosen.append(best_next)
        remaining.remove(best_next)
        current_l2 = best_next_l2
        print(f"  + {best_next} → {current_l2:.4f}  {chosen}")
    if current_l2 < best_avg[0]:
        best_avg = (current_l2, chosen)

    print(f"\n>>> BEST: l2={best_avg[0]:.4f}  {best_avg[1]}")

    # Save the best average as the new best
    avg = avg_states([states[n] for n in best_avg[1]])
    model.load_state_dict(avg)
    out_path = Path("checkpoints/best.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    # Also save to PVC under a new run id
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    pvc = PVC_CKPTS / f"model-wavg-{commit}"
    pvc.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pvc / "checkpoint.pt")
    print(f"Saved to {out_path} and {pvc / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
