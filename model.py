import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class CNF_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetBlock(2, 32)
        self.down2 = UNetBlock(32, 64)
        self.mid = UNetBlock(64, 64)
        self.up1 = UNetBlock(64 + 64, 32)
        self.up2 = UNetBlock(32 + 32, 16)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        xt = torch.cat([x, t], dim=1)

        d1 = self.down1(xt)
        d2 = self.down2(F.avg_pool2d(d1, 2))
        m = self.mid(F.avg_pool2d(d2, 2))

        u1 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, d2], dim=1))

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, d1], dim=1))

        return self.out(u2)
