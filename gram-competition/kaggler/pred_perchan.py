"""Per-timestep, per-channel mixture weights across the full 90-pred pool.

Instead of a single scalar weight per ckpt, learn [K, 5, 3] weights (softmax over K
per timestep+channel). Each channel/timestep combination picks its own mix.
"""
import json
import subprocess
from pathlib import Path

import torch

CACHE = Path("/mnt/new-pvc/kagent/apr16/tanjiro/predcache")
CACHE_TTA = Path("/mnt/new-pvc/kagent/apr16/tanjiro/predcache_tta")


def l2(pred, truth):
    return (pred - truth).norm(dim=3).mean(dim=(1, 2)).mean()


def main():
    device = torch.device("cuda")
    truth = torch.load(CACHE / "_truth.pt", weights_only=True).to(device)

    orig_files = sorted(CACHE.glob("model-*.pt"))
    tta_files = sorted(CACHE_TTA.glob("model-*_yflip.pt"))
    names = [p.stem for p in orig_files] + [p.stem for p in tta_files]
    preds = torch.stack(
        [torch.load(p, weights_only=True) for p in orig_files] +
        [torch.load(p, weights_only=True) for p in tta_files],
        dim=0,
    ).to(device)  # [K, 80, 5, N, 3] fp32 on GPU (~43 GB)
    truth = truth.to(device)
    K, S, T, N, C = preds.shape
    print(f"Loaded K={K} preds, shape {tuple(preds.shape)} fp32 on GPU")

    CHUNK = 16
    chunks = [(i, min(i + CHUNK, S)) for i in range(0, S, CHUNK)]

    def chunked_loss(get_w):
        total = torch.zeros((), device=device)
        for a, b in chunks:
            w = get_w()  # broadcast tensor
            mixed = (w * preds[:, a:b]).sum(dim=0)
            total = total + (mixed - truth[a:b]).norm(dim=-1).mean(dim=(-1, -2)).sum()
        return total / S

    # Baseline: scalar weights [K]
    logits_s = torch.zeros(K, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits_s], lr=0.02)
    best_s = (float("inf"), None)
    for step in range(2000):
        loss = chunked_loss(lambda: torch.softmax(logits_s, dim=0)[:, None, None, None, None])
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_s[0]:
            best_s = (loss.item(), logits_s.detach().clone())
        if step % 200 == 0:
            print(f"[scalar] step={step}: l2={loss.item():.4f}")
    print(f">>> scalar best: l2={best_s[0]:.4f}")

    # Per-timestep, per-channel: logits [K, T, C]
    logits_tc = torch.zeros(K, T, C, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits_tc], lr=0.02)
    best_tc = (float("inf"), None)
    for step in range(3000):
        loss = chunked_loss(lambda: torch.softmax(logits_tc, dim=0)[:, None, :, None, :])
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < best_tc[0]:
            best_tc = (loss.item(), logits_tc.detach().clone())
        if step % 200 == 0:
            print(f"[t,c] step={step}: l2={loss.item():.4f}")
    print(f">>> [t,c] best: l2={best_tc[0]:.4f}")

    logits_tc = best_tc[1]
    w_tc = torch.softmax(logits_tc, dim=0)  # [K, T, C]
    with torch.no_grad():
        final_mixed = torch.zeros(S, T, N, C, device=device)
        for a, b in chunks:
            final_mixed[a:b] = (w_tc[:, None, :, None, :] * preds[:, a:b]).sum(dim=0)
    final_l2 = l2(final_mixed, truth).item()
    print(f"Final per-(t,c): l2={final_l2:.4f}")

    # Per-channel summary of weights (mean across timesteps)
    w_c = w_tc.mean(dim=1)  # [K, 3]
    print("\nPer-channel top-10 (mean over t):")
    for c_idx, c_name in enumerate(["Ux", "Uy", "Uz"]):
        ranked = sorted(zip(names, w_c[:, c_idx].cpu().tolist()), key=lambda x: -x[1])
        print(f"  {c_name}:")
        for n, wi in ranked[:10]:
            print(f"    {n}: {wi:.3f}")

    predictions = [final_mixed[i].cpu() for i in range(final_mixed.shape[0])]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    out_dir = Path(f"/mnt/new-pvc/predictions/apr16/tanjiro/{commit}-perchan")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, out_dir / "val.pt")
    (out_dir / "meta.json").write_text(json.dumps({
        "method": "per_timestep_per_channel_softmax",
        "l2": final_l2,
        "l2_scalar_baseline": best_s[0],
        "n_ckpts": K,
        "weights_per_channel_mean_over_t": {
            n: {"Ux": float(w_c[i, 0]), "Uy": float(w_c[i, 1]), "Uz": float(w_c[i, 2])}
            for i, n in enumerate(names)
        },
    }, indent=2))
    print(f"Saved to {out_dir / 'val.pt'}")


if __name__ == "__main__":
    main()
