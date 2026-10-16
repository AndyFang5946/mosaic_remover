import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False 
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): 
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, base_channels=32):
        super().__init__() 

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.bottlenext = DoubleConv(base_channels * 8, base_channels * 16)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.output = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _merge(up, skip):
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(
                up,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return torch.cat([up, skip], dim=1)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        b = self.bottlenext(self.pool(e4))

        d4 = self.dec4(self._merge(self.up4(b), e4))
        d3 = self.dec3(self._merge(self.up3(d4), e3))
        d2 = self.dec2(self._merge(self.up2(d3), e2))
        d1 = self.dec1(self._merge(self.up1(d2), e1))

        return self.output(d1)

        