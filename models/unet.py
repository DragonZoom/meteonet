# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:28:32 2020

@author: Vince
https://1.bp.blogspot.com/-Mz4K8FlBjbE/Xh0CKBF8wOI/AAAAAAAAFMs/7r3_QnAhN9A0Ervr8plf7qVORnmFkh-qgCLcBGAsYHQ/s1600/image4.png
https://arxiv.org/pdf/1608.04117.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels=out_channels // 2, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, model_size = 8, bilinear=True):
        super(UNet, self).__init__()

        n = model_size

        self.inc = DoubleConv(n_channels, 2*n)
        self.down1 = Down(2*n, 4*n)
        self.down2 = Down(4*n, 8*n)
        self.down3 = Down(8*n, 16*n)
        factor = 2 if bilinear else 1
        self.down4 = Down(16*n, 32*n // factor)
        self.up1 = Up(32*n, 16*n, bilinear)
        self.up2 = Up(16*n, 8*n, bilinear)
        self.up3 = Up(8*n, 4*n, bilinear)
        self.up4 = Up(4*n, 2*n*factor, bilinear)
        self.outc = OutConv(2*n, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def get_nb_params(self):
        pp=0
        for p in list(self.parameters()):
            n = 1
            for s in list(p.size()):
                n = n*s
            pp += n
        return pp
