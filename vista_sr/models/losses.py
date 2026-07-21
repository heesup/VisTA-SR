"""Loss functions for training VisTA-SR model (Content, MSE, Cycle, Adversarial)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor


class ContentLoss(nn.Module):
    """VGG19 Feature Content Loss for Perceptual Quality."""

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).eval()
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        if sr.shape[1] == 1:
            sr = sr.repeat(1, 3, 1, 1)
        if hr.shape[1] == 1:
            hr = hr.repeat(1, 3, 1, 1)

        sr_norm = (sr - self.mean) / self.std
        hr_norm = (hr - self.mean) / self.std

        return F.l1_loss(self.feature_extractor(sr_norm), self.feature_extractor(hr_norm))
