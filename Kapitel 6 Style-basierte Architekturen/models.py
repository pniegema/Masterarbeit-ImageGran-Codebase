from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class HeatmapEncoder64(nn.Module):
    def __init__(self, in_channels=68, latent_dim=128, first_layer_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(in_channels, first_layer_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(first_layer_channels),
            nn.ReLU(True),

            # 32x32 -> 16x16
            nn.Conv2d(first_layer_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),

            # 8x8 -> 4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),

            # 4x4 -> 2x2
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.ReLU(True)

        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, latent_dim)
        self.scaling = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)         # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)           # (B, latent_dim)
        x = self.scaling(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(out_channels)  # ArcFace often uses PReLU
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class IdentityEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512):
        super(IdentityEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.PReLU(32)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 256, num_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, latent_dim)
        self.scaling = nn.Linear(latent_dim, latent_dim)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (B, 256)
        x = self.fc(x)             # (B, latent_dim)
        x = self.scaling(x)
        return x

class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(style_dim, channels)
        self.style_shift_transform = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        normalized = self.norm(x)
        style_scale = self.style_scale_transform(style).unsqueeze(2).unsqueeze(3)
        style_shift = self.style_shift_transform(style).unsqueeze(2).unsqueeze(3)
        return style_scale * normalized + style_shift



class StyleConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, style_dim, w_dim, upsample):
        super(StyleConvBlock, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.adain1 = AdaIN(out_channels, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.adain2 = AdaIN(out_channels, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

        #Network to Map the fused latent of size input_dimension to style space of size style_dim
        self.w_dim = w_dim
        self.style_MLP1 = nn.Sequential(nn.Linear(w_dim, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(512, style_dim))

        self.style_MLP2 = nn.Sequential(nn.Linear(w_dim, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, style_dim))


    def forward(self, x, w):
        if self.upsample:
            x = self.upsample_layer(x)

        style1 = self.style_MLP1(w)
        style2 = self.style_MLP2(w)

        x = self.conv1(x)
        x = self.adain1(x, style1)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.adain2(x, style2)
        x = self.lrelu2(x)

        return x

class StyleGenerator(nn.Module):
    """
    Style-based Generator Network with adjustable input dimension and output resolution.

    """
    def __init__(self, input_dimension=1152):
        super(StyleGenerator, self).__init__()

        self.input_dimension = input_dimension

        self.MappingNetwork = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        self.starting_constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        self.channels = {
            4: 512,
            8: 256,
            16: 256,
            32: 128,
            64: 64,
            128: 64
        }

        self.blocks = nn.ModuleList()
        resolutions = [4, 8, 16, 32, 64, 128]
        in_ch = 512

        for res in resolutions[1:]:
            if res == resolutions[0]: # Dont upsample

                out_ch = self.channels[res]
                self.blocks.append(StyleConvBlock(in_ch, out_ch, style_dim=512, w_dim=512, upsample=False))
                in_ch = out_ch

            else:
                out_ch = self.channels[res]
                self.blocks.append(StyleConvBlock(in_ch, out_ch, style_dim=512, w_dim=512, upsample=True))
                in_ch = out_ch

        self.to_rgb = nn.Conv2d(in_ch, 3, kernel_size=1)

    def forward(self, z):
        w = self.MappingNetwork(z)
        B = z.size(0)
        x = self.starting_constant.repeat(B, 1, 1, 1)
        for block in self.blocks:
            x = block(x, w)
        rgb = self.to_rgb(x)
        rgb = torch.tanh(rgb)
        return rgb



