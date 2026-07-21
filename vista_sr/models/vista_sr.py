"""VisTA-SR integrated network architecture combining Stage 1 alignment and Stage 2 SR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .cyclegan import ResnetGenerator


class ResidualConvBlock(nn.Module):
    """Residual Convolutional Block."""

    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.add(self.rcb(x), x)


@torch.jit.script
def matchTemplateTorchCore(img_tensor, template_tensor):
    result1 = F.conv2d(img_tensor, template_tensor, bias=None, stride=1, padding=0)
    result2 = torch.sqrt(
        torch.sum(template_tensor**2) * F.conv2d(img_tensor**2, torch.ones_like(template_tensor), bias=None, stride=1, padding=0)
    )
    return (result1 / (result2 + 1e-8)).squeeze(0).squeeze(0)


def matchTemplateThetaBatch(background, template):
    batch_size = background.shape[0]
    theta = torch.zeros(batch_size, 6, device=background.device)

    for i in range(batch_size):
        template_i = template[i].unsqueeze(0)
        min_val, max_val = torch.min(template_i), torch.max(template_i)
        if max_val > min_val:
            template_i = (template_i - min_val) / (max_val - min_val)

        bg_i = F.interpolate(
            background[i].unsqueeze(0),
            size=(template_i.shape[-2], template_i.shape[-1]),
            mode='bilinear',
            align_corners=False
        )

        x_offset = background.shape[-2] // 4
        y_offset = background.shape[-1] // 4
        bg_i = F.pad(bg_i, (x_offset, x_offset, y_offset, y_offset), mode='replicate')

        res = matchTemplateTorchCore(bg_i, template_i)
        result_max_loc = torch.argmax(res)
        result_max_loc_x = result_max_loc % res.shape[0]
        result_max_loc_y = result_max_loc // res.shape[1]

        x_t = -2 * (result_max_loc_x - x_offset) / template.shape[-2]
        y_t = -2 * (result_max_loc_y - y_offset) / template_i.shape[-1]

        if abs(x_t) + abs(y_t) > 0.8:
            x_t = y_t = 0.0

        t_matrix = torch.tensor([1.0, 0.0, x_t, 0.0, 1.0, y_t], device=background.device)
        theta[i] = t_matrix

    return theta


class VisTASRGenerator(nn.Module):
    """VisTA-SR Network Generator combining Stage 1 (Alignment) and Stage 2 (Super-Resolution)."""

    def __init__(self, stn_image_size: int = 96, debug: bool = False) -> None:
        super(VisTASRGenerator, self).__init__()
        self.stn_image_size = stn_image_size
        self.debug = debug

        # Domain translation networks
        self.rgb2ir = ResnetGenerator(3, 1, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)
        self.ir2rgb = ResnetGenerator(1, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False)

        # Super Resolution feature extraction
        self.conv_block1_cycleGAN = nn.Sequential(
            nn.Conv2d(5, 64, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
        )

        trunk = [ResidualConvBlock(64) for _ in range(16)]
        self.trunk = nn.Sequential(*trunk)

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )

        self.upsampling_img = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(),
            nn.Conv2d(32, 1, (1, 1), (1, 1), (0, 0))
        )

    def forward(self, x_ir: Tensor, y_rgb: Tensor) -> Tensor:
        # Domain translation RGB -> IR
        out_rgb2ir = self.rgb2ir(y_rgb)
        self.out_rgb2ir = out_rgb2ir
        self.out_rgb2ir2rgb = self.ir2rgb(out_rgb2ir)

        # Domain translation IR -> RGB
        out_ir2rgb = self.ir2rgb(x_ir)
        self.out_ir2rgb = out_ir2rgb
        self.out_ir2rgb2ir = self.rgb2ir(out_ir2rgb)

        # Template Matching Alignment
        x_stn = F.interpolate(x_ir, size=(self.stn_image_size, self.stn_image_size), mode='bilinear', align_corners=False)
        out_rgb2ir_stn = F.interpolate(out_rgb2ir, size=(self.stn_image_size, self.stn_image_size), mode='bilinear', align_corners=False)
        theta = matchTemplateThetaBatch(x_stn, out_rgb2ir_stn)

        resampling_grid_ir = F.affine_grid(theta.view(-1, 2, 3), out_rgb2ir.size(), align_corners=False)
        out_rgb2ir_aligned = F.grid_sample(out_rgb2ir, resampling_grid_ir, mode='bilinear', padding_mode='border', align_corners=False)
        self.out_rgb2ir_aligned = out_rgb2ir_aligned

        resampling_grid_rgb = F.affine_grid(theta.view(-1, 2, 3), y_rgb.size(), align_corners=False)
        y_aligned = F.grid_sample(y_rgb, resampling_grid_rgb, mode='bilinear', padding_mode='border', align_corners=False)
        self.y_aligned = y_aligned

        # Feature Concatenation & SR Generation
        out_ir_upsampled = self.upsampling_img(x_ir)
        out_concat = torch.cat((out_ir_upsampled, out_rgb2ir_aligned.detach(), y_aligned.detach()), dim=1)

        f1 = self.conv_block1_cycleGAN(out_concat)
        f_trunk = self.trunk(f1)
        f2 = self.conv_block2(f_trunk)
        f_res = torch.add(f1, f2)

        sr_output = self.conv_block3(f_res)
        return sr_output
