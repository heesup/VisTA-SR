"""Visualization utilities for generating paper heatmaps and comparative figures."""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np


def save_heatmap_comparison(
    rgb_tensor: torch.Tensor,
    lr_tensor: torch.Tensor,
    aligned_tensor: torch.Tensor,
    vista_sr_tensor: torch.Tensor,
    hr_tensor: torch.Tensor,
    save_path: str,
    title: str = "VisTA-SR Output Comparison"
):
    """Save qualitative visualization grid comparing Input RGB, LR Thermal, Aligned RGB2IR, VisTA-SR, and HR Thermal."""

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # RGB image
    rgb_img = rgb_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)
    axes[0].imshow(rgb_img)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    # LR Thermal
    lr_img = lr_tensor.squeeze().cpu().numpy()
    axes[1].imshow(lr_img, cmap="magma")
    axes[1].set_title("Low-Res Thermal")
    axes[1].axis("off")

    # Aligned RGB2IR
    aligned_img = aligned_tensor.squeeze().cpu().numpy()
    axes[2].imshow(aligned_img, cmap="magma")
    axes[2].set_title("Aligned RGB2IR")
    axes[2].axis("off")

    # VisTA-SR
    sr_img = vista_sr_tensor.squeeze().cpu().numpy()
    axes[3].imshow(sr_img, cmap="magma")
    axes[3].set_title("VisTA-SR (Ours)")
    axes[3].axis("off")

    # Ground Truth HR
    hr_img = hr_tensor.squeeze().cpu().numpy()
    im = axes[4].imshow(hr_img, cmap="magma")
    axes[4].set_title("Ground Truth HR")
    axes[4].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.suptitle(title, fontsize=16)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
