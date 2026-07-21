#!/usr/bin/env python3
"""Training script for VisTA-SR model."""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vista_sr.config import Config
from vista_sr.datasets import ThermalImageDataset
from vista_sr.models import VisTASRGenerator, Discriminator, ContentLoss
from vista_sr.utils.metrics import calculate_psnr, calculate_ssim, calculate_rmse


def parse_args():
    parser = argparse.ArgumentParser(description="Train VisTA-SR model")
    parser.add_argument("--data-dir", type=str, default="data/Training_T4_1_2_3", help="Dataset directory")
    parser.add_argument("--exp-name", type=str, default="vista_sr_run", help="Experiment name")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(mode="train_srgan", exp_name=args.exp_name)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.g_model_lr = args.lr
    config.d_model_lr = args.lr

    print(f"--- Training VisTA-SR: {config.exp_name} on {config.device} ---")

    # Datasets & Dataloaders
    train_dataset = ThermalImageDataset(os.path.join(args.data_dir, "train"), image_size=config.image_size, mode="train")
    val_dataset = ThermalImageDataset(os.path.join(args.data_dir, "val"), image_size=config.image_size, mode="valid")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Networks
    generator = VisTASRGenerator(stn_image_size=config.stn_image_size).to(config.device)
    discriminator = Discriminator(image_size=config.image_size).to(config.device)

    # Losses
    pixel_criterion = nn.MSELoss().to(config.device)
    content_criterion = ContentLoss().to(config.device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(config.device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=config.g_model_lr, betas=config.g_model_betas)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=config.d_model_lr, betas=config.d_model_betas)

    # Checkpoint dirs
    checkpoint_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(checkpoint_dir, "logs"))

    best_psnr = 0.0

    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()

        total_g_loss = 0.0
        total_d_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        for step, batch in enumerate(pbar):
            lr = batch["lr"].to(config.device)
            hr = batch["hr"].to(config.device)
            rgb = batch["rgb"].to(config.device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            d_optimizer.zero_grad()
            sr = generator(lr, rgb)

            real_label = torch.full((hr.size(0), 1), 1.0, device=config.device)
            fake_label = torch.full((hr.size(0), 1), 0.0, device=config.device)

            d_real_out = discriminator(hr)
            d_real_loss = adversarial_criterion(d_real_out, real_label)

            d_fake_out = discriminator(sr.detach())
            d_fake_loss = adversarial_criterion(d_fake_out, fake_label)

            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            d_optimizer.step()

            # ---------------------
            # Train Generator
            # ---------------------
            g_optimizer.zero_grad()
            
            p_loss = pixel_criterion(sr, hr)
            c_loss = content_criterion(sr, hr)
            d_out = discriminator(sr)
            a_loss = adversarial_criterion(d_out, real_label)

            g_loss = p_loss + 0.006 * c_loss + 1e-3 * a_loss
            g_loss.backward()
            g_optimizer.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            pbar.set_postfix({"G_Loss": f"{g_loss.item():.4f}", "D_Loss": f"{d_loss.item():.4f}"})

        writer.add_scalar("Loss/Generator", total_g_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Discriminator", total_d_loss / len(train_loader), epoch)

        # Validation step every 5 epochs
        if epoch % 5 == 0 or epoch == config.epochs:
            generator.eval()
            val_psnr, val_ssim, val_rmse = 0.0, 0.0, 0.0
            with torch.no_grad():
                for batch in val_loader:
                    lr = batch["lr"].to(config.device)
                    hr = batch["hr"].to(config.device)
                    rgb = batch["rgb"].to(config.device)
                    sr = generator(lr, rgb)

                    val_psnr += calculate_psnr(sr, hr)
                    val_ssim += calculate_ssim(sr, hr)
                    val_rmse += calculate_rmse(sr, hr)

            val_psnr /= len(val_loader)
            val_ssim /= len(val_loader)
            val_rmse /= len(val_loader)

            print(f"--> Epoch {epoch} Validation - PSNR: {val_psnr:.2f} dB | SSIM: {val_ssim:.4f} | RMSE: {val_rmse:.4f}")
            writer.add_scalar("Val/PSNR", val_psnr, epoch)
            writer.add_scalar("Val/SSIM", val_ssim, epoch)
            writer.add_scalar("Val/RMSE", val_rmse, epoch)

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "g-best.pth"))
                torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "d-best.pth"))

        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, "g-last.pth"))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, "d-last.pth"))

    print(f"Training completed! Best PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
