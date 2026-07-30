#!/usr/bin/env python
"""Run the hybrid smoke test from inside this architecture folder.

Usage from ``hybrid_stroke_diffusion/``:

    PYTHONPATH=. python -u scripts/run_hybrid_stroke_diffusion_smoke.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

try:
    from .model import StrokeAutoencoder
    from .geometry import sample_clothoid_params
    from .render import render_strokes
    from .vector_data import SVGClothoidDataset
except ImportError:
    from model import StrokeAutoencoder
    from geometry import sample_clothoid_params
    from render import render_strokes
    from vector_data import SVGClothoidDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ae-steps", type=int, default=200)
    ap.add_argument("--diffusion-steps", type=int, default=0)
    ap.add_argument("--out-dir", default="runs/hybrid_stroke_diffusion_smoke")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device(args.device)

    # Create dummy dataset for smoke test
    print("creating dummy SVG dataset...", flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Initialize AE
    ae = StrokeAutoencoder(latent_dim=64, hidden=128, udf_dim=16).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=3e-4)

    # Generate random strokes for smoke test
    n_strokes = 16
    params = torch.zeros(1, n_strokes, 10, device=device)
    presence = torch.zeros(1, n_strokes, device=device)

    # Fill with random valid strokes
    for i in range(n_strokes):
        if torch.rand(1).item() > 0.3:
            params[0, i, :2] = torch.rand(2, device=device) * 0.8 + 0.1
            params[0, i, 2] = torch.rand(1, device=device) * math.pi
            params[0, i, 3] = torch.rand(1, device=device) * 0.3 + 0.1
            params[0, i, 4] = torch.randn(1, device=device) * 5.0
            params[0, i, 5] = torch.randn(1, device=device) * 10.0
            params[0, i, 6] = torch.rand(1, device=device) * 0.04 + 0.005
            params[0, i, 7:10] = torch.rand(3, device=device)
            presence[0, i] = 1.0

    print(f"training AE for {args.ae_steps} steps...", flush=True)
    for step in range(args.ae_steps):
        recon, logits, z = ae(params)
        loss_dict = {}
        total_loss, loss_dict = ae.loss(params, presence)
        opt.zero_grad(); total_loss.backward(); opt.step()
        if step % 50 == 0 or step == args.ae_steps - 1:
            print(f"AE step {step}: loss={float(total_loss):.4f}", flush=True)

    # Render target vs reconstruction
    with torch.no_grad():
        recon, logits, _ = ae(params)
        pred_presence = (logits > 0).float()
        target_img = render_strokes(params, presence, size=192, curve_samples=32)[0]
        recon_img = render_strokes(recon, pred_presence, size=192, curve_samples=32)[0]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(target_img.cpu()); axes[0].set_title("target")
    axes[1].imshow(recon_img.cpu()); axes[1].set_title("AE reconstruction")
    for ax in axes: ax.axis('off')
    fig.tight_layout()
    fig.savefig(out / "preview_target_vs_reconstruction.png", dpi=140)
    plt.close(fig)
    print(f"preview -> {out / 'preview_target_vs_reconstruction.png'}", flush=True)
    print(json.dumps({"ae_final_loss": float(total_loss)}), flush=True)


if __name__ == "__main__":
    main()
