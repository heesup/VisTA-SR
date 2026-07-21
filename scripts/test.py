#!/usr/bin/env python3
"""Evaluation and metrics computation script for VisTA-SR."""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from vista_sr.config import Config
from vista_sr.datasets import ThermalImageDataset
from vista_sr.models import VisTASRGenerator
from vista_sr.utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VisTA-SR model")
    parser.add_argument("--data-dir", type=str, default="data/Training_T4_1_2_3", help="Dataset directory")
    parser.add_argument("--weights", type=str, default="", help="Path to generator weights (.pth)")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    config = Config(mode="valid")
    device = config.device

    val_dataset = ThermalImageDataset(os.path.join(args.data_dir, "val"), image_size=256, mode="valid")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    generator = VisTASRGenerator().to(device)
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        generator.load_state_dict(torch.load(args.weights, map_location=device))
    else:
        print("Evaluating initialized/baseline model...")

    generator.eval()

    # Metrics storage
    bilinear_metrics = {"rmse": [], "ssim": [], "psnr": []}
    vista_metrics = {"rmse": [], "ssim": [], "psnr": []}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            rgb = batch["rgb"].to(device)

            # 1. Bilinear Interpolation Baseline
            bilinear_sr = F.interpolate(lr, size=(hr.shape[-2], hr.shape[-1]), mode="bilinear", align_corners=False)
            bilinear_metrics["rmse"].append(calculate_rmse(bilinear_sr, hr))
            bilinear_metrics["ssim"].append(calculate_ssim(bilinear_sr, hr))
            bilinear_metrics["psnr"].append(calculate_psnr(bilinear_sr, hr))

            # 2. VisTA-SR Model
            vista_sr = generator(lr, rgb)
            vista_metrics["rmse"].append(calculate_rmse(vista_sr, hr))
            vista_metrics["ssim"].append(calculate_ssim(vista_sr, hr))
            vista_metrics["psnr"].append(calculate_psnr(vista_sr, hr))

    print("\n==================================================")
    print("           VisTA-SR Benchmark Results             ")
    print("==================================================")
    print(f"{'Technique':<15} | {'RMSE (°C)':<10} | {'SSIM':<8} | {'PSNR':<8}")
    print("-" * 50)
    print(f"{'Bilinear':<15} | {sum(bilinear_metrics['rmse'])/len(bilinear_metrics['rmse']):<10.2f} | {sum(bilinear_metrics['ssim'])/len(bilinear_metrics['ssim']):<8.2f} | {sum(bilinear_metrics['psnr'])/len(bilinear_metrics['psnr']):<8.2f}")
    print(f"{'VisTA-SR (Ours)':<15} | {sum(vista_metrics['rmse'])/len(vista_metrics['rmse']):<10.2f} | {sum(vista_metrics['ssim'])/len(vista_metrics['ssim']):<8.2f} | {sum(vista_metrics['psnr'])/len(vista_metrics['psnr']):<8.2f}")
    print("==================================================\n")


if __name__ == "__main__":
    evaluate()
