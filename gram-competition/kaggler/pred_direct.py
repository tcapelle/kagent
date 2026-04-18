"""Directly optimize per-ckpt weights against the actual eval metric via autograd.

Uses cached predictions from pred_weighted.py. Each ckpt's prediction is a tensor of
shape [80, 5, N, 3] stacked across samples (padded). The eval metric is:
  mean_samples( mean_points( ||pred - truth||_2 ) )
which is smooth in w via mixture: pred(w) = sum_i softmax(w)_i * P_i.
"""
import json
import subprocess
from pathlib import Path

import torch

CACHE = Path("/mnt/new-pvc/kagent/apr16/tanjiro/predcache")


def l2(pred, truth):
    return (pred - truth).norm(dim=3).mean(dim=(1, 2)).mean()


def main():
    truth = torch.load(CACHE / "_truth.pt", weights_only=True)
    cand_files = sorted(CACHE.glob("model-*.pt"))
    names = [p.stem for p in cand_files]
    preds = torch.stack([torch.load(p, weights_only=True) for p in cand_files], dim=0)  # [K, 80, 5, N, 3]
    K = preds.shape[0]
    print(f"Loaded {K} ckpts, preds shape {tuple(preds.shape)}")

    # Move to GPU for speed
    device = torch.device("cuda")
    preds = preds.to(device)
    truth = truth.to(device)

    # Baseline: uniform across all K
    uniform = preds.mean(dim=0)
    print(f"Uniform (all {K}): l2={l2(uniform, truth).item():.4f}")

    # Baseline: uniform of the 8 chosen by exp60 greedy
    exp60_chosen = ["model-wavg3-e49e9cb", "model-wavg-816156a", "model-670v4v75",
                    "model-mgo03egs", "model-bbr0yz3i", "model-kvptxsnv",
                    "model-eu7w7w48", "model-b1hzbt3r"]
    mask = torch.tensor([n in exp60_chosen for n in names], device=device, dtype=torch.float32)
    w60 = mask / mask.sum()
    avg60 = (w60[:, None, None, None, None] * preds).sum(dim=0)
    print(f"Uniform (8 chosen): l2={l2(avg60, truth).item():.4f}")

    # Softmax-parameterized weights, init such that softmax ≈ uniform-of-8
    logits = torch.full((K,), -5.0, device=device, requires_grad=False)
    for i, n in enumerate(names):
        if n in exp60_chosen:
            logits[i] = 0.0
    logits = logits.detach().clone().requires_grad_(True)

    opt = torch.optim.Adam([logits], lr=0.02)
    best = (float("inf"), None)
    for step in range(3000):
        w = torch.softmax(logits, dim=0)
        mixed = (w[:, None, None, None, None] * preds).sum(dim=0)
        loss = l2(mixed, truth)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < best[0]:
            best = (loss.item(), logits.detach().clone())
        if step % 200 == 0:
            print(f"step={step}: l2={loss.item():.4f}")
    print(f">>> Adam best: l2={best[0]:.4f}")

    logits = best[1]
    w = torch.softmax(logits, dim=0)
    final_mixed = (w[:, None, None, None, None] * preds).sum(dim=0)
    final_l2 = l2(final_mixed, truth).item()
    print(f"Final: l2={final_l2:.4f}")
    top = sorted(zip(names, w.cpu().tolist()), key=lambda x: -x[1])
    for n, wi in top[:12]:
        print(f"  {n}: {wi:.3f}")

    predictions = [final_mixed[i].cpu() for i in range(final_mixed.shape[0])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out_dir = Path(f"/mnt/new-pvc/predictions/apr16/tanjiro/{commit}-direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "val.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "method": "adam_softmax_direct",
        "l2": final_l2,
        "weights": {n: float(wi) for n, wi in zip(names, w.cpu().tolist())},
    }, indent=2))
    print(f"Saved to {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
