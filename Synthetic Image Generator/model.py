import torch
import torch.nn as nn
import torch.nn.functional as F

# === UNet Block (Generator part) ===
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

# === CNF-UNet Model (Generator) ===
class CNF_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input to down1 is 2 channels: image + time
        self.down1 = UNetBlock(2, 64)
        self.down2 = UNetBlock(64, 128)
        # Using a deeper mid block might help with capturing more complex features
        self.mid = UNetBlock(128, 256) # Increased channels in mid block
        self.up1 = UNetBlock(256 + 128, 128) # Adjusted for mid block change
        self.up2 = UNetBlock(128 + 64, 64) # Adjusted for up1 change
        self.out = nn.Conv2d(64, 1, kernel_size=1) # Adjusted for up2 change

    def forward(self, x, t):
        t_embedded = t.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        xt = torch.cat([x, t_embedded], dim=1)

        d1 = self.down1(xt)
        d2 = self.down2(F.avg_pool2d(d1, 2))
        m = self.mid(F.avg_pool2d(d2, 2))

        u1 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, d2], dim=1))

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, d1], dim=1))

        return self.out(u2)

    def _positional_encoding(self, t, dim): # This function is not currently used in forward.
        inv_freq = 1. / (10000**(torch.arange(0, dim, 2).float() / dim)).to(t.device)
        sin_term = torch.sin(t.unsqueeze(1) * inv_freq)
        cos_term = torch.cos(t.unsqueeze(1) * inv_freq)
        pos_embed = torch.cat([sin_term, cos_term], dim=-1).view(t.size(0), -1)
        if pos_embed.size(1) < dim:
            pos_embed = F.pad(pos_embed, (0, dim - pos_embed.size(1)))
        elif pos_embed.size(1) > dim:
            pos_embed = pos_embed[:, :dim]
        return pos_embed

# === Discriminator Model ===
class Discriminator(nn.Module):
    def __init__(self, img_channels=1, features_d=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2),
            # Final layer, output single logit
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x):
        return self.disc(x)