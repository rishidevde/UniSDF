#!/usr/bin/env python
"""Quantitative evaluation script for UniSDF.

Computes PSNR, SSIM and Chamfer Distance on the test split of a dataset.
Usage example:
    python scripts/eval_quantitative.py \
        --config configs/unisdf-colmap_sparse.yaml \
        --ckpt path/to/checkpoint.ckpt \
        --output results/quantitative.json \
        --device cuda
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np

# Optional: use torchmetrics if installed, otherwise fallback to simple implementations
try:
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    _has_torchmetrics = True
except Exception:
    _has_torchmetrics = False

# ---------------------------------------------------------------------------
# Ensure the repository root is on PYTHONPATH.
# scripts/ is one level below the repo root, so parents[1] gives us repo root.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

import datasets
import systems
from utils.misc import load_config


# ── Fallback metric implementations ──────────────────────────────────────────

def compute_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (simple fallback)."""
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return float('inf')
    return (20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)).item()


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Structural Similarity (placeholder – install torchmetrics for real SSIM)."""
    return 0.0


# ── Core helpers ─────────────────────────────────────────────────────────────

def load_model_and_data(config_path: str, ckpt_path: str, device: str):
    """Load config, build system + dataset, restore checkpoint weights."""
    config = load_config(config_path)
    config.cmd_args = {}
    dm = datasets.make(config.dataset.name, config.dataset)
    system = systems.make(config.system.name, config, load_from_checkpoint=None)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    system.load_state_dict(state_dict, strict=False)
    system.to(device)
    system.eval()
    return system, dm, config


def evaluate_image_metrics(system, datamodule, device: str):
    """Render every test view and compute per-image PSNR / SSIM."""
    loader = datamodule.test_dataloader()
    psnr_list, ssim_list = [], []

    with torch.no_grad():
        for batch in loader:
            rays = batch["rays"].to(device)
            gt_rgb = batch["rgb"].to(device)
            render_out = system(rays)
            pred_rgb = render_out["rgb"]

            if _has_torchmetrics:
                psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
                ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
                psnr_val = psnr_metric(pred_rgb, gt_rgb).item()
                ssim_val = ssim_metric(pred_rgb, gt_rgb).item()
            else:
                psnr_val = compute_psnr(pred_rgb, gt_rgb)
                ssim_val = compute_ssim(pred_rgb, gt_rgb)

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    return avg_psnr, avg_ssim


def evaluate_chamfer(system, datamodule, device: str):
    """Extract mesh via marching-cubes and compute Chamfer distance to GT."""
    try:
        import mcubes
    except ImportError:
        print("[WARNING] mcubes not installed – Chamfer will be skipped.")
        return None

    # Dense SDF grid sampling
    vol_res = 256
    xs = torch.linspace(-1, 1, vol_res)
    coords = (
        torch.stack(torch.meshgrid(xs, xs, xs, indexing='ij'), dim=-1)
        .reshape(-1, 3)
        .to(device)
    )
    with torch.no_grad():
        sdf = system.sdf_network(coords).view(vol_res, vol_res, vol_res).cpu().numpy()

    verts, faces = mcubes.marching_cubes(sdf, 0.0)

    # Collect ground-truth point cloud from dataset
    gt_pts = []
    for batch in datamodule.test_dataloader():
        if "points" in batch:
            gt_pts.append(batch["points"].cpu().numpy())
    if not gt_pts:
        print("[WARNING] No ground-truth point cloud found in dataset – Chamfer skipped.")
        return None
    gt_pts = np.concatenate(gt_pts, axis=0)

    # Chamfer Distance
    try:
        from torchmetrics.image import ChamferDistance
        cd = ChamferDistance().to(device)
        cd_val = cd(
            torch.tensor(verts, dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(gt_pts, dtype=torch.float32, device=device).unsqueeze(0),
        ).item()
        return cd_val
    except Exception:
        # Fallback: scipy kd-tree based Chamfer
        from scipy.spatial import cKDTree
        tree_pred = cKDTree(verts)
        tree_gt = cKDTree(gt_pts)
        d1, _ = tree_pred.query(gt_pts, k=1)
        d2, _ = tree_gt.query(verts, k=1)
        return float(np.mean(d1 ** 2) + np.mean(d2 ** 2))


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantitative evaluation for UniSDF (PSNR, SSIM, Chamfer)"
    )
    parser.add_argument("--config", required=True, help="YAML config used for training")
    parser.add_argument("--ckpt", required=True, help="Checkpoint (.ckpt) to evaluate")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--output", default="quantitative_results.json",
                        help="JSON file for results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    system, dm, config = load_model_and_data(args.config, args.ckpt, str(device))

    print("[INFO] Computing PSNR & SSIM …")
    avg_psnr, avg_ssim = evaluate_image_metrics(system, dm, str(device))

    print("[INFO] Computing Chamfer Distance …")
    chamfer = evaluate_chamfer(system, dm, str(device))

    results = {"psnr": avg_psnr, "ssim": avg_ssim, "chamfer": chamfer}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Evaluation complete. Results saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
