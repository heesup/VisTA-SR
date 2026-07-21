#!/usr/bin/env python3
"""Visualization script generating qualitative figures comparing input, alignment, VisTA-SR output, and ground truth."""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vista_sr.config import Config
from vista_sr.datasets import ThermalImageDataset
from vista_sr.models import VisTASRGenerator
from vista_sr.utils.visualization import save_heatmap_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VisTA-SR visualization figures")
    parser.add_argument("--data-dir", type=str, default="data/Training_T4_1_2_3", help="Dataset directory")
    parser.add_argument("--weights", type=str, default="", help="Path to generator weights (.pth)")
    parser.add_argument("--output-dir", type=str, default="results/visualizations", help="Output directory for figures")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of visualization samples to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(mode="valid")
    device = config.device

    val_dataset = ThermalImageDataset(os.path.join(args.data_dir, "val"), image_size=256, mode="valid")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    generator = VisTASRGenerator().to(device)
    if args.weights and os.path.exists(args.weights):
        print(f"Loading weights from {args.weights}")
        generator.load_state_dict(torch.load(args.weights, map_location=device))

    generator.eval()
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader, desc="Generating Visualizations")):
            if idx >= args.num_samples:
                break
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            rgb = batch["rgb"].to(device)
            filename = batch["filename"][0]

            sr = generator(lr, rgb)
            aligned_rgb2ir = generator.out_rgb2ir_aligned

            save_path = os.path.join(args.output_dir, f"vis_{idx+1}_{filename}.png")
            save_heatmap_comparison(
                rgb_tensor=rgb,
                lr_tensor=lr,
                aligned_tensor=aligned_rgb2ir,
                vista_sr_tensor=sr,
                hr_tensor=hr,
                save_path=save_path,
                title=f"Sample {idx+1}: {filename}"
            )

    print(f"Saved {min(args.num_samples, len(val_dataset))} visualization samples to '{args.output_dir}'!")


if __name__ == "__main__":
    main()
