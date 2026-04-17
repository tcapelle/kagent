"""Train a 3D airflow velocity predictor.

Template — fill in your model architecture.
The training loop, loss, validation, and W&B logging are provided.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import simple_parsing as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, VAL_SPLIT_NAMES, collate_fn, load_data


# ---------------------------------------------------------------------------
# Model — residual prediction + voxel-grid spatial mixer
#
# Contract: model(velocity_in [B,5,N,3], pos [B,N,3], t [B,10], idcs_airfoil list[Tensor]) -> [B,5,N,3]
#
# Key design:
#   - Predict delta from last input frame (residual), in normalized space.
#   - Spatial context via voxel-grid pooling + 3D conv + trilinear gather.
#   - Per-point ResMLP blocks alternate with VoxelMixer blocks.
#   - Hard no-slip: zero velocity on airfoil indices.
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, pos=None):
        return x + self.net(x)


class VoxelMixer(nn.Module):
    """Pool point features onto a 3D voxel grid, mix via 3D conv, gather back.

    Uses per-sample bbox normalization so the grid covers each geometry tightly.
    Scatter-average pooling, then 2-layer 3D conv with residual, then trilinear
    grid_sample to recover per-point features.
    """

    def __init__(self, dim, grid_size=32):
        super().__init__()
        self.G = grid_size
        self.G_coarse = grid_size // 2
        self.norm = nn.LayerNorm(dim)
        # Fine branch (G³): mean+max aggregation, identity-init on mean half.
        self.proj_agg = nn.Conv3d(2 * dim, dim, 1)
        nn.init.zeros_(self.proj_agg.weight)
        nn.init.zeros_(self.proj_agg.bias)
        with torch.no_grad():
            for i in range(dim):
                self.proj_agg.weight[i, i, 0, 0, 0] = 1.0
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )
        # Coarse branch ((G/2)³): zero-init end-to-end → starts at 0 contribution (warm-start safe).
        self.proj_agg_coarse = nn.Conv3d(2 * dim, dim, 1)
        nn.init.zeros_(self.proj_agg_coarse.weight)
        nn.init.zeros_(self.proj_agg_coarse.bias)
        self.conv_coarse = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(dim, dim, 3, padding=1),
        )
        nn.init.zeros_(self.conv_coarse[-1].weight)
        nn.init.zeros_(self.conv_coarse[-1].bias)

    def _voxel_mm(self, h, p_norm, G):
        B, N, D = h.shape
        v_idx = (p_norm * G).long().clamp(0, G - 1)
        flat_idx = v_idx[..., 0] * G * G + v_idx[..., 1] * G + v_idx[..., 2]
        idx_D = flat_idx.unsqueeze(-1).expand(-1, -1, D)
        voxel_mean = torch.zeros(B, G ** 3, D, device=h.device, dtype=h.dtype)
        count = torch.zeros(B, G ** 3, device=h.device, dtype=h.dtype)
        voxel_mean.scatter_add_(1, idx_D, h)
        count.scatter_add_(1, flat_idx, torch.ones_like(flat_idx, dtype=h.dtype))
        voxel_mean = voxel_mean / count.unsqueeze(-1).clamp(min=1.0)
        voxel_max = torch.zeros(B, G ** 3, D, device=h.device, dtype=h.dtype)
        voxel_max.scatter_reduce_(1, idx_D, h, reduce="amax", include_self=False)
        vm = voxel_mean.view(B, G, G, G, D).permute(0, 4, 1, 2, 3).contiguous()
        vx = voxel_max.view(B, G, G, G, D).permute(0, 4, 1, 2, 3).contiguous()
        return vm, vx

    def _gather(self, vf, p_norm, B, N):
        # grid_sample: 5D input [B, C, D, H, W], grid last dim = (x, y, z) → (W, H, D).
        # Our voxel axes are (pos_x=D, pos_y=H, pos_z=W), so grid = (p_z, p_y, p_x).
        grid = (p_norm * 2 - 1)[:, :, [2, 1, 0]].view(B, 1, 1, N, 3)
        sampled = F.grid_sample(vf, grid, mode="bilinear",
                                align_corners=True, padding_mode="border")
        return sampled.squeeze(2).squeeze(2).permute(0, 2, 1)

    def forward(self, x, pos):
        B, N, D = x.shape
        h = self.norm(x)

        p_min = pos.amin(dim=1, keepdim=True)
        p_max = pos.amax(dim=1, keepdim=True)
        p_norm = (pos - p_min) / (p_max - p_min).clamp(min=1e-6)  # [B, N, 3] in [0,1]

        vm, vx = self._voxel_mm(h, p_norm, self.G)
        vf = self.proj_agg(torch.cat([vm, vx], dim=1))
        vf = vf + self.conv(vf)
        sampled = self._gather(vf, p_norm, B, N)

        vm_c, vx_c = self._voxel_mm(h, p_norm, self.G_coarse)
        vf_c = self.proj_agg_coarse(torch.cat([vm_c, vx_c], dim=1))
        vf_c = vf_c + self.conv_coarse(vf_c)
        sampled_c = self._gather(vf_c, p_norm, B, N)

        return x + sampled + sampled_c


class BaselineMLP(nn.Module):
    """Residual MLP + voxel-grid spatial mixer, predicts delta in normalized space."""

    def __init__(self, hidden=256, n_blocks=4, grid_size=32,
                 vel_mean=None, vel_std=None, n_fourier=8, sdf_samples=1024):
        super().__init__()
        self.n_fourier = n_fourier
        self.sdf_samples = sdf_samples
        fourier_dim = 3 * n_fourier * 2
        # +2 feats: normalized SDF + is_airfoil indicator
        in_dim = 3 + fourier_dim + T_IN * 3 + 2
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(hidden))
            blocks.append(VoxelMixer(hidden, grid_size=grid_size))
        blocks.append(ResBlock(hidden))
        self.blocks = nn.ModuleList(blocks)

        self.proj_out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))
        vel_mean = torch.zeros(3) if vel_mean is None else vel_mean
        vel_std = torch.ones(3) if vel_std is None else vel_std
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

        freqs = 2.0 ** torch.arange(n_fourier) * torch.pi
        self.register_buffer("fourier_freqs", freqs)

        nn.init.zeros_(self.proj_out[-1].weight)
        nn.init.zeros_(self.proj_out[-1].bias)

    def _fourier(self, pos):
        p = pos.unsqueeze(-1) * self.fourier_freqs  # [B, N, 3, K]
        return torch.cat([p.sin(), p.cos()], dim=-1).reshape(pos.shape[0], pos.shape[1], -1)

    @torch.no_grad()
    def _geom_features(self, pos, idcs_airfoil):
        """SDF-to-airfoil (normalized by bbox diag) + is_airfoil binary."""
        B, N, _ = pos.shape
        p_min = pos.amin(dim=1, keepdim=True)
        p_max = pos.amax(dim=1, keepdim=True)
        diag = (p_max - p_min).pow(2).sum(-1).sqrt().clamp(min=1e-6)  # [B, 1]

        sdf = pos.new_zeros(B, N)
        is_af = pos.new_zeros(B, N)
        for b in range(B):
            af_idx = idcs_airfoil[b]
            af = pos[b, af_idx]  # [M, 3]
            if af.shape[0] > self.sdf_samples:
                sel = torch.randperm(af.shape[0], device=af.device)[:self.sdf_samples]
                af = af[sel]
            # Chunked cdist to avoid ~100k * 1k memory spikes
            d_list = []
            for i in range(0, N, 20000):
                d = torch.cdist(pos[b, i:i + 20000], af).min(dim=-1).values
                d_list.append(d)
            sdf[b] = torch.cat(d_list)
            is_af[b, af_idx] = 1.0
        sdf_norm = sdf / diag  # [B, N]
        return sdf_norm.unsqueeze(-1), is_af.unsqueeze(-1)

    def forward(self, velocity_in, pos, t, idcs_airfoil):
        B, T, N, C = velocity_in.shape
        v_in_norm = (velocity_in - self.vel_mean) / self.vel_std  # [B, 5, N, 3]
        v_feat = v_in_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)  # [B, N, 15]
        fpos = self._fourier(pos)
        sdf, is_af = self._geom_features(pos, idcs_airfoil)
        x = torch.cat([pos, fpos, v_feat, sdf, is_af], dim=-1)
        x = self.proj_in(x)

        for blk in self.blocks:
            x = blk(x, pos)

        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std
        pred = velocity_in[:, -1:] + delta
        for b in range(B):
            pred[b, :, idcs_airfoil[b]] = 0.0
        return pred


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loaders, device, global_step):
    model.eval()
    val_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        total_l2 = 0.0
        total_mae = torch.zeros(3, device=device, dtype=torch.float64)
        n_samples = 0

        with torch.no_grad():
            for v_in, v_out, pos, t, idcs in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs)

                l2_err = (pred - v_out).norm(dim=3).mean(dim=(1, 2))
                total_l2 += l2_err.sum().item()

                mae = (pred - v_out).abs().mean(dim=(1, 2))
                total_mae += mae.double().sum(dim=0)
                n_samples += v_in.shape[0]

        mean_l2 = total_l2 / max(n_samples, 1)
        mean_mae = total_mae / max(n_samples, 1)

        val_metrics[split_name] = {
            f"{split_name}/l2_error": mean_l2,
            f"{split_name}/mae_Ux": mean_mae[0].item(),
            f"{split_name}/mae_Uy": mean_mae[1].item(),
            f"{split_name}/mae_Uz": mean_mae[2].item(),
        }

    mean_val = sum(m[f"{k}/l2_error"] for k, m in val_metrics.items()) / len(val_metrics)

    metrics = {"val/l2_error": mean_val, "global_step": global_step}
    for sm in val_metrics.values():
        metrics.update(sm)
    wandb.log(metrics)

    return mean_val, val_metrics


# ---------------------------------------------------------------------------
# Config + training entrypoint
# ---------------------------------------------------------------------------

MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))  # minutes


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 50
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False
    hidden: int = 256
    n_blocks: int = 8
    grid_size: int = 32
    n_fourier: int = 8
    grad_clip: float = 1.0
    bf16: bool = True
    subsample_train: int = 50000
    warm_start: str | None = None


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    loader_kwargs = dict(collate_fn=collate_fn, num_workers=2, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = BaselineMLP(
        hidden=cfg.hidden,
        n_blocks=cfg.n_blocks,
        grid_size=cfg.grid_size,
        n_fourier=cfg.n_fourier,
        vel_mean=stats["vel_mean"],
        vel_std=stats["vel_std"],
    ).to(device)

    if cfg.warm_start:
        state = torch.load(cfg.warm_start, map_location=device, weights_only=True)
        state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Warm-started from {cfg.warm_start} (missing={len(missing)}, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    RESEARCH_TAG = os.environ.get("RESEARCH_TAG", "default")

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team"),
        project=os.environ.get("WANDB_PROJECT", "kagent-gram"),
        group=cfg.wandb_group or RESEARCH_TAG,
        name=cfg.wandb_name,
        tags=[t for t in [cfg.agent, RESEARCH_TAG] if t],
        config={**asdict(cfg), "n_params": n_params,
                "train_samples": len(train_ds),
                "val_samples": {k: len(v) for k, v in val_splits.items()}},
        mode=os.environ.get("WANDB_MODE", "online"),
    )

    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("val/*", step_metric="global_step")

    KAGGLER_NAME = os.environ.get("KAGGLER_NAME", cfg.agent or "local")
    pvc_dir = Path(f"/mnt/new-pvc/kagent/{RESEARCH_TAG}/{KAGGLER_NAME}/checkpoints/model-{run.id}")
    pvc_dir.mkdir(parents=True, exist_ok=True)
    model_path = pvc_dir / "checkpoint.pt"

    git_ckpt_path = Path("checkpoints/best.pt")
    git_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    best_metrics: dict = {}
    global_step = 0
    train_start = time.time()

    for epoch in range(MAX_EPOCHS):
        if (time.time() - train_start) / 60.0 >= MAX_TIMEOUT:
            print(f"Timeout ({MAX_TIMEOUT} min). Stopping.")
            break

        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        amp_enabled = cfg.bf16 and device.type == "cuda"

        for v_in, v_out, pos, t, idcs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)

            if 0 < cfg.subsample_train < v_in.shape[2]:
                N = v_in.shape[2]
                K = cfg.subsample_train
                sel = torch.randperm(N, device=device)[:K]
                v_in = v_in[:, :, sel]
                v_out = v_out[:, :, sel]
                pos = pos[:, sel]
                new_idcs = []
                mask_full = torch.zeros(N, dtype=torch.bool, device=device)
                inv = torch.empty(N, dtype=torch.long, device=device).fill_(-1)
                inv[sel] = torch.arange(K, device=device)
                for af in idcs:
                    af = af.to(device, non_blocking=True)
                    new_idcs.append(inv[af][inv[af] >= 0])
                idcs = new_idcs

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                pred = model(v_in, pos, t, idcs)
                vel_std = model.vel_std
                loss = ((pred - v_out) / vel_std).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            global_step += 1
            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)
        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": scheduler.get_last_lr()[0],
                   "epoch_time_s": dt, "global_step": global_step})

        tag = ""
        if mean_val < best_val:
            best_val = mean_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": mean_val}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(model.state_dict(), model_path)
            shutil.copyfile(model_path, git_ckpt_path)
            tag = " *"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
            f"train={epoch_loss:.4f}  val/l2={mean_val:.4f}{tag}"
        )

    total_time = (time.time() - train_start) / 60.0
    print(f"\nDone ({total_time:.1f} min)")

    if best_metrics:
        print(f"Best: epoch {best_metrics['epoch']}, val/l2_error={best_metrics['val_l2_error']:.4f}")
        wandb.summary.update({"best_" + k: v for k, v in best_metrics.items()})

    if best_metrics and not cfg.debug:
        import subprocess
        print("\nGenerating test predictions...")
        pred_cmd = ["python", "predict.py", "--checkpoint", str(model_path)]
        if cfg.agent:
            pred_cmd += ["--agent", cfg.agent]
        result = subprocess.run(pred_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"predict.py failed:\n{result.stderr[-500:]}")

    wandb.finish()


if __name__ == "__main__":
    main()
