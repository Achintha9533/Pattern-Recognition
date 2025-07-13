# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # For get_sinusoidal_embedding

# === Helper function for sinusoidal positional embedding ===
def get_sinusoidal_embedding(t, embed_dim):
    inv_freq = 1. / (10000**(torch.arange(0, embed_dim, 2).float() / embed_dim)).to(t.device)
    sin_term = torch.sin(t.unsqueeze(1) * inv_freq)
    cos_term = torch.cos(t.unsqueeze(1) * inv_freq)
    emb = torch.cat([sin_term, cos_term], dim=-1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# === Time Embedding Module ===
class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.embed_dim = embed_dim

    def forward(self, t):
        emb = get_sinusoidal_embedding(t, self.embed_dim)
        return self.fc(emb)

# === Residual Block for UNet ===
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, t_emb):
        B, C, H, W = x.shape
        time_scaled = self.time_proj(t_emb).view(B, -1, 1, 1)

        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x + time_scaled) # Additive conditioning
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + self.shortcut(residual)) # Additive conditioning with shortcut
        return x

# === Self-Attention Block ===
class SelfAttention2d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1) # B, HW, C/8
        proj_key = self.key(x).view(B, -1, H * W) # B, C/8, HW
        energy = torch.bmm(proj_query, proj_key) # B, HW, HW
        attention = torch.softmax(energy, dim=-1) # B, HW, HW
        proj_value = self.value(x).view(B, -1, H * W) # B, C, HW

        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, HW
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# === UNet Block (Generator part with Residual Blocks) ===
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embed_dim=256, num_res_blocks=2):
        super().__init__()
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, time_embed_dim))
        for _ in range(num_res_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, time_embed_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x, t_emb):
        for layer in self.block:
            x = layer(x, t_emb)
        return x

# === CNF-UNet Model (Generator) ===
class CNF_UNet(nn.Module):
    def __init__(self, time_embed_dim=256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)

        self.inc = nn.Conv2d(1, 64, kernel_size=3, padding=1) # Input: 1 channel image

        self.down1 = UNetBlock(64, 128, time_embed_dim)
        self.down2 = UNetBlock(128, 256, time_embed_dim)

        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256, time_embed_dim),
            SelfAttention2d(256),
            ResidualBlock(256, 256, time_embed_dim)
        )

        self.up1 = UNetBlock(256 + 256, 128, time_embed_dim) # Concatenate with skip connection
        self.up2 = UNetBlock(128 + 128, 64, time_embed_dim) # Concatenate with skip connection
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, t):
        t_emb = self.time_embedding(t) # Shape: (batch_size, time_embed_dim)

        x_in = self.inc(x) # initial convolution to expand channels

        d1 = self.down1(x_in, t_emb)
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb)

        m = self.bottleneck[0](F.avg_pool2d(d2, 2), t_emb) # First Residual Block in bottleneck
        m = self.bottleneck[1](m) # Self-Attention
        m = self.bottleneck[2](m, t_emb) # Second Residual Block in bottleneck

        u1 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, d2], dim=1), t_emb)

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, d1], dim=1), t_emb)

        return self.outc(u2)