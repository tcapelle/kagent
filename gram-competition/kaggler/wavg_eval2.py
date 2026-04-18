"""Extended wavg: include 241-key checkpoints (pre pos-offset) with zero-init pos_proj."""
import json
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
    for k, ref in reference_state.items():
        if k not in s:
            s[k] = torch.zeros_like(ref) if ref.is_floating_point() else ref.clone()
    return s


def avg_states(states):
    out = {}
    for k in states[0].keys():
        if not states[0][k].is_floating_point():
            out[k] = states[0][k]
        else:
            out[k] = sum(s[k] for s in states) / len(states)
    return out


def validate(model, loader, device):
    model.eval()
    total_l2 = 0.0
    n = 0
    with torch.no_grad():
        for v_in, v_out, pos, t, idcs in loader:
            v_in = v_in.to(device); v_out = v_out.to(device)
            pos = pos.to(device); t = t.to(device)
            pred = model(v_in, pos, t, idcs)
            total_l2 += (pred - v_out).norm(dim=3).mean(dim=(1, 2)).sum().item()
            n += v_in.shape[0]
    return total_l2 / n


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

    # All compatible 241+243 key checkpoints
    candidates = [
        "model-mh5wd0t6",   # exp45, 243
        "model-fgqa41ag",   # exp56, 243
        "model-ea2ll188",   # 243
        "model-eu7w7w48",   # exp44, 243
        "model-bbr0yz3i",   # 243
        "model-b1hzbt3r",   # 243
        "model-dfal10k2",   # 243
        "model-gfbsgqi7",   # 243
        "model-35pla0n7",   # 243
        "model-er5pk3oc",   # exp41, 241 (no pos_proj)
        "model-jay6zniz",   # exp42, 241
        "model-uesu5tb6",   # exp43 era, 241
        "model-wavg-816156a",  # the wavg result
    ]
    scores = {}
    states = {}
    for c in candidates:
        p = PVC_CKPTS / c / "checkpoint.pt"
        if not p.exists(): continue
        s = load_and_complete(p, ref)
        model.load_state_dict(s)
        l2 = validate(model, val_loader, device)
        scores[c] = l2
        states[c] = s
        print(f"{c}: l2={l2:.4f}")

    sorted_names = sorted(scores, key=scores.get)
    print(f"\nBest single: {sorted_names[0]} ({scores[sorted_names[0]]:.4f})")

    # Greedy search starting from best single
    print("\n--- greedy ensemble ---")
    chosen = [sorted_names[0]]
    current_l2 = scores[sorted_names[0]]
    remaining = [n for n in sorted_names if n not in chosen]
    while remaining:
        best_next = None
        best_next_l2 = current_l2
        for cand in remaining:
            test = chosen + [cand]
            avg = avg_states([states[n] for n in test])
            model.load_state_dict(avg)
            l2 = validate(model, val_loader, device)
            if l2 < best_next_l2:
                best_next_l2 = l2
                best_next = cand
        if best_next is None:
            break
        chosen.append(best_next)
        remaining.remove(best_next)
        current_l2 = best_next_l2
        print(f"  + {best_next} → {current_l2:.4f}  {chosen}")

    print(f"\n>>> BEST: l2={current_l2:.4f}  {chosen}")

    # Save the best average
    avg = avg_states([states[n] for n in chosen])
    model.load_state_dict(avg)
    out_path = Path("checkpoints/best.pt")
    torch.save(model.state_dict(), out_path)
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    pvc = PVC_CKPTS / f"model-wavg2-{commit}"
    pvc.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pvc / "checkpoint.pt")
    print(f"Saved to {out_path} and {pvc / 'checkpoint.pt'}")


if __name__ == "__main__":
    main()
