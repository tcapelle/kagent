"""v5 — v2 + signed-distance-to-airfoil feature.

For each point, precompute the Euclidean distance to the nearest airfoil
point. That scalar is passed as an input feature so the model has an explicit
wall-distance prior — critical for boundary-layer physics.

SDF is computed once per sample on the GPU (chunked `cdist`), cached in
RAM, and re-used across all epochs.

Run:
  python train.py --agent <your-name> --wandb_name "<your-name>/<description>"
"""

import copy
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import N_POINTS, T_IN, T_OUT, load_data


def compute_sdf(pos, airfoil_idcs, device, chunk=2048):
    """Per-point Euclidean distance to the nearest airfoil point."""
    pos_g = pos.to(device)
    a = pos_g[airfoil_idcs.to(device)]
    sdf = torch.full((pos.shape[0],), float("inf"), device=device)
    for s in range(0, a.shape[0], chunk):
        d = torch.cdist(pos_g, a[s:s + chunk]).min(dim=-1).values
        sdf = torch.minimum(sdf, d)
    return sdf.cpu()


class SDFDataset(Dataset):
    def __init__(self, base, sdfs):
        self.base = base
        self.sdfs = sdfs

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        v_in, v_out, pos, t, idcs = self.base[idx]
        return v_in, v_out, pos, t, idcs, self.sdfs[idx]


def collate_sdf(batch):
    v_in, v_out, pos, t, idcs, sdf = zip(*batch)
    return (
        torch.stack(v_in),
        torch.stack(v_out),
        torch.stack(pos),
        torch.stack(t),
        list(idcs),
        torch.stack(sdf),
    )


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ConvBlock3D(nn.Module):
    def __init__(self, c_in, c_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
            nn.Conv3d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    """Tiny 3D UNet with 2 pooling levels."""

    def __init__(self, c_in, c_mid=64, c_out=None, groups=8):
        super().__init__()
        c_out = c_out or c_in
        self.enc1 = ConvBlock3D(c_in, c_mid, groups)
        self.enc2 = ConvBlock3D(c_mid, c_mid * 2, groups)
        self.enc3 = ConvBlock3D(c_mid * 2, c_mid * 4, groups)
        self.dec2 = ConvBlock3D(c_mid * 2 + c_mid * 4, c_mid * 2, groups)
        self.dec1 = ConvBlock3D(c_mid + c_mid * 2, c_mid, groups)
        self.out = nn.Conv3d(c_mid, c_out, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool3d(e1, 2))
        e3 = self.enc3(F.avg_pool3d(e2, 2))
        d2 = self.dec2(torch.cat(
            [F.interpolate(e3, scale_factor=2, mode="trilinear", align_corners=False), e2], dim=1
        ))
        d1 = self.dec1(torch.cat(
            [F.interpolate(d2, scale_factor=2, mode="trilinear", align_corners=False), e1], dim=1
        ))
        return self.out(d1)


class VoxelSpatial(nn.Module):
    """Scatter-mean points into voxel grid, UNet3D, trilinear back to points."""

    def __init__(self, dim, res=64, unet_mid=64, pad=0.05):
        super().__init__()
        self.res = res
        self.pad = pad
        self.unet = UNet3D(c_in=dim, c_mid=unet_mid, c_out=dim)
        # zero-init last conv → residual starts as identity
        nn.init.zeros_(self.unet.out.weight)
        nn.init.zeros_(self.unet.out.bias)

    def forward(self, feats, pos):
        B, N, D = feats.shape
        R = self.res
        lo = pos.amin(dim=1, keepdim=True) - self.pad
        hi = pos.amax(dim=1, keepdim=True) + self.pad
        p01 = (pos - lo) / (hi - lo).clamp(min=1e-6)
        idx = (p01 * R).long().clamp(0, R - 1)
        flat = idx[..., 0] * R * R + idx[..., 1] * R + idx[..., 2]

        vox = feats.new_zeros(B, D, R * R * R)
        cnt = feats.new_zeros(B, 1, R * R * R)
        vox.scatter_add_(2, flat.unsqueeze(1).expand(-1, D, -1), feats.transpose(1, 2))
        cnt.scatter_add_(2, flat.unsqueeze(1),
                         torch.ones_like(flat, dtype=feats.dtype).unsqueeze(1))
        vox = vox / cnt.clamp(min=1.0)
        vox = vox.view(B, D, R, R, R)

        vox = self.unet(vox)

        # grid_sample expects (x,y,z) in (W,H,D) order which maps to our (Z,Y,X) voxel axes
        grid = (p01 * 2 - 1)[:, None, None, :, [2, 1, 0]]
        sampled = F.grid_sample(vox, grid, mode="bilinear",
                                align_corners=False, padding_mode="border")
        sampled = sampled.squeeze(2).squeeze(2).transpose(1, 2)
        return feats + sampled


class VoxelResidualModel(nn.Module):
    """Residual-from-last-frame + per-point ResMLP + voxel-UNet spatial context + SDF."""

    def __init__(
        self,
        vel_mean,
        vel_std,
        hidden=256,
        n_blocks_pre=2,
        n_blocks_post=4,
        voxel_res=64,
        voxel_mid=64,
    ):
        super().__init__()
        in_dim = T_IN * 3 + 3 + 1 + 2  # velocities + pos + airfoil-mask + (sdf, log1p(sdf))
        out_dim = T_OUT * 3
        self.proj_in = nn.Linear(in_dim, hidden)
        self.blocks_pre = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks_pre)])
        self.spatial = VoxelSpatial(dim=hidden, res=voxel_res, unet_mid=voxel_mid)
        self.blocks_post = nn.Sequential(*[ResBlock(hidden) for _ in range(n_blocks_post)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        self.register_buffer("vel_mean", vel_mean.view(1, 1, 1, 3))
        self.register_buffer("vel_std", vel_std.view(1, 1, 1, 3))

    def forward(self, velocity_in, pos, t, idcs_airfoil, sdf):
        B, T, N, C = velocity_in.shape
        v_norm = (velocity_in - self.vel_mean) / self.vel_std
        v_feat = v_norm.permute(0, 2, 1, 3).reshape(B, N, T * C)

        mask = torch.zeros(B, N, 1, device=velocity_in.device, dtype=velocity_in.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            mask[b, idcs.to(mask.device), 0] = 1.0

        sdf_raw = (sdf / 5.0).unsqueeze(-1)  # rough ~unit scale
        sdf_log = torch.log1p(sdf).unsqueeze(-1)

        x = torch.cat([v_feat, pos, mask, sdf_raw, sdf_log], dim=-1)
        x = self.proj_in(x)
        x = self.blocks_pre(x)
        x = self.spatial(x, pos)
        x = self.blocks_post(x)
        x = self.norm_out(x)
        delta_norm = self.proj_out(x).reshape(B, N, T_OUT, 3).permute(0, 2, 1, 3)
        delta = delta_norm * self.vel_std

        last_frame = velocity_in[:, -1:].expand(-1, T_OUT, -1, -1)
        pred = last_frame + delta

        no_slip = torch.ones(B, 1, N, 1, device=pred.device, dtype=pred.dtype)
        for b, idcs in enumerate(idcs_airfoil):
            no_slip[b, 0, idcs.to(no_slip.device), 0] = 0.0
        return pred * no_slip


def infer_arch_from_state_dict(sd):
    """Given a model state_dict, recover the arch kwargs for VoxelResidualModel."""
    hidden = sd["proj_in.weight"].shape[0]
    voxel_mid = sd["spatial.unet.enc1.block.0.weight"].shape[0]
    return {"hidden": hidden, "voxel_res": 64, "voxel_mid": voxel_mid}


def validate(model, val_loaders, device, global_step):
    model.eval()
    val_metrics: dict[str, dict] = {}

    for split_name, vloader in val_loaders.items():
        total_l2 = 0.0
        total_mae = torch.zeros(3, device=device, dtype=torch.float64)
        n_samples = 0

        with torch.no_grad():
            for v_in, v_out, pos, t, idcs, sdf in vloader:
                v_in = v_in.to(device, non_blocking=True)
                v_out = v_out.to(device, non_blocking=True)
                pos = pos.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                sdf = sdf.to(device, non_blocking=True)

                pred = model(v_in, pos, t, idcs, sdf)

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


MAX_TIMEOUT = float(os.environ.get("MAX_TIMEOUT_MIN", "30"))


@dataclass
class Config:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    epochs: int = 90
    hidden: int = 256
    voxel_res: int = 64
    voxel_mid: int = 64
    ema_beta: float = 0.0  # 0 = off; typical: 0.999 (short window) to 0.9999 (long)
    loss_type: str = "mse"  # "mse" (current) or "l2" (norm-L2 per point, matches eval metric)
    splits_dir: str = "/mnt/new-pvc/datasets/gram/splits"
    wandb_group: str | None = None
    wandb_name: str | None = None
    agent: str | None = None
    debug: bool = False


def main():
    cfg = sp.parse(Config)
    MAX_EPOCHS = 3 if cfg.debug else cfg.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (" [DEBUG]" if cfg.debug else ""))

    train_ds, val_splits, stats = load_data(cfg.splits_dir, debug=cfg.debug)

    print("Precomputing SDFs...")
    t0 = time.time()
    train_sdfs = [compute_sdf(train_ds[i][2], train_ds[i][4], device) for i in tqdm(range(len(train_ds)), desc="train SDF")]
    val_sdfs = {name: [compute_sdf(ds[i][2], ds[i][4], device) for i in tqdm(range(len(ds)), desc=f"{name} SDF")]
                for name, ds in val_splits.items()}
    print(f"  done ({time.time()-t0:.1f}s)")

    train_ds = SDFDataset(train_ds, train_sdfs)
    val_splits = {name: SDFDataset(ds, val_sdfs[name]) for name, ds in val_splits.items()}

    loader_kwargs = dict(collate_fn=collate_sdf, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, **loader_kwargs)
    val_loaders = {
        name: DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, ds in val_splits.items()
    }

    model = VoxelResidualModel(
        vel_mean=stats["vel_mean"],
        vel_std=stats["vel_std"],
        hidden=cfg.hidden,
        voxel_res=cfg.voxel_res,
        voxel_mid=cfg.voxel_mid,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.2f} M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # EMA shadow model (optional). Updated every step; validated alongside model.
    ema_model = None
    if cfg.ema_beta > 0:
        ema_model = copy.deepcopy(model).to(device)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"EMA enabled, beta={cfg.ema_beta}")

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

        for v_in, v_out, pos, t, idcs, sdf in tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS}", leave=False):
            v_in = v_in.to(device, non_blocking=True)
            v_out = v_out.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            sdf = sdf.to(device, non_blocking=True)

            pred = model(v_in, pos, t, idcs, sdf)
            vel_std = stats["vel_std"].to(device).view(1, 1, 1, 3)
            diff = (pred - v_out) / vel_std
            if cfg.loss_type == "l2":
                loss = diff.norm(dim=-1).mean()
            else:
                loss = diff.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if ema_model is not None:
                with torch.no_grad():
                    beta = cfg.ema_beta
                    for ep, p in zip(ema_model.parameters(), model.parameters()):
                        ep.data.mul_(beta).add_(p.data, alpha=1 - beta)
                    for eb, b in zip(ema_model.buffers(), model.buffers()):
                        eb.data.copy_(b.data)

            wandb.log({"train/loss": loss.item(), "global_step": global_step})

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        mean_val, split_metrics = validate(model, val_loaders, device, global_step)

        mean_val_ema = None
        if ema_model is not None:
            mean_val_ema, _ = validate(ema_model, val_loaders, device, global_step)
            wandb.log({"val/l2_error_ema": mean_val_ema, "global_step": global_step})

        dt = time.time() - t0

        wandb.log({"train/epoch_loss": epoch_loss, "lr": scheduler.get_last_lr()[0],
                   "epoch_time_s": dt, "global_step": global_step})

        candidate_val = mean_val_ema if (mean_val_ema is not None and mean_val_ema < mean_val) else mean_val
        candidate_state = ema_model.state_dict() if (mean_val_ema is not None and mean_val_ema < mean_val) else model.state_dict()
        candidate_source = "ema" if (mean_val_ema is not None and mean_val_ema < mean_val) else "raw"

        tag = ""
        if candidate_val < best_val:
            best_val = candidate_val
            best_metrics = {"epoch": epoch + 1, "val_l2_error": candidate_val, "source": candidate_source}
            for sm in split_metrics.values():
                best_metrics.update({f"best_{k}": v for k, v in sm.items()})
            torch.save(candidate_state, model_path)
            shutil.copyfile(model_path, git_ckpt_path)
            tag = f" *({candidate_source})"

        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        ema_str = f"  ema={mean_val_ema:.4f}" if mean_val_ema is not None else ""
        print(
            f"Epoch {epoch+1:3d} ({dt:.0f}s) [{peak_gb:.1f}GB]  "
            f"train={epoch_loss:.4f}  val/l2={mean_val:.4f}{ema_str}{tag}"
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
