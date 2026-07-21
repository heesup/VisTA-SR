"""Dataset class for paired low-res thermal, high-res thermal, and RGB images."""

import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


class ThermalImageDataset(Dataset):
    """Dataset class for VisTA-SR dataset loading.

    Args:
        dataroot (str): Path to dataset folder containing IR_LOW, IR_HIGH, and RGB subfolders.
        image_size (int): Target high resolution image crop size.
        upscale_factor (int): Upscaling factor (default: 4).
        mode (str): Data loading mode ('train' or 'valid').
    """

    def __init__(self, dataroot: str, image_size: int = 256, upscale_factor: int = 4, mode: str = "train") -> None:
        super(ThermalImageDataset, self).__init__()
        self.dataroot = dataroot
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        self.mode = mode

        low_dataroot = os.path.join(dataroot, "IR_LOW")
        high_dataroot = os.path.join(dataroot, "IR_HIGH")
        rgb_dataroot = os.path.join(dataroot, "RGB")

        valid_exts = ("jpg", "jpeg", "png", "bmp", "tiff")

        self.low_filenames = sorted([
            os.path.join(low_dataroot, x) for x in os.listdir(low_dataroot)
            if x.lower().endswith(valid_exts)
        ])
        self.high_filenames = sorted([
            os.path.join(high_dataroot, x) for x in os.listdir(high_dataroot)
            if x.lower().endswith(valid_exts)
        ])
        self.rgb_filenames = sorted([
            os.path.join(rgb_dataroot, x) for x in os.listdir(rgb_dataroot)
            if x.lower().endswith(valid_exts)
        ])

        assert len(self.low_filenames) == len(self.high_filenames), \
            f"Mismatched dataset lengths: {len(self.low_filenames)} LR vs {len(self.high_filenames)} HR"

    def __len__(self) -> int:
        return len(self.low_filenames)

    def __getitem__(self, idx: int):
        # Open Low-Res Thermal (1 channel / Grayscale)
        lr_img = Image.open(self.low_filenames[idx]).convert("L")
        # Open High-Res Thermal (1 channel / Grayscale)
        hr_img = Image.open(self.high_filenames[idx]).convert("L")
        # Open RGB Image (3 channels)
        if idx < len(self.rgb_filenames):
            rgb_img = Image.open(self.rgb_filenames[idx]).convert("RGB")
        else:
            rgb_img = Image.new("RGB", hr_img.size)

        if self.mode == "train":
            # Random Crop for training
            w, h = hr_img.size
            if w >= self.image_size and h >= self.image_size:
                top = random.randint(0, h - self.image_size)
                left = random.randint(0, w - self.image_size)
                hr_img = TF.crop(hr_img, top, left, self.image_size, self.image_size)
                rgb_img = TF.crop(rgb_img, top, left, self.image_size, self.image_size)
                
                lr_size = self.image_size // self.upscale_factor
                lr_top = top // self.upscale_factor
                lr_left = left // self.upscale_factor
                lr_img = TF.crop(lr_img, lr_top, lr_left, lr_size, lr_size)
            else:
                hr_img = TF.resize(hr_img, (self.image_size, self.image_size))
                rgb_img = TF.resize(rgb_img, (self.image_size, self.image_size))
                lr_img = TF.resize(lr_img, (self.image_size // self.upscale_factor, self.image_size // self.upscale_factor))

            # Random horizontal flip
            if random.random() > 0.5:
                lr_img = TF.hflip(lr_img)
                hr_img = TF.hflip(hr_img)
                rgb_img = TF.hflip(rgb_img)

        # Convert images to PyTorch Tensors normalized to [0, 1]
        lr_tensor = TF.to_tensor(lr_img)
        hr_tensor = TF.to_tensor(hr_img)
        rgb_tensor = TF.to_tensor(rgb_img)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "rgb": rgb_tensor,
            "filename": os.path.basename(self.high_filenames[idx])
        }
