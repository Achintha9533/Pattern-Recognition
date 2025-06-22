import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# === UNet Block with GroupNorm (for Generator) ===
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, out_channels), out_channels), # Adjust group count for small channels
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(out_channels)

    def forward(self, x):
        x = self.block(x)
        if self.use_attention:
            x = self.attention(x)
        return x

# === Self-Attention Block ===
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        num_heads = max(1, channels // 64) if channels > 64 else 4 # Heuristic
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True) 
        self.ln = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(), # GELU is common in Transformers
            nn.Linear(channels * 4, channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1) # B, HW, C

        # Multi-head attention
        attn_output, _ = self.mha(x_flat, x_flat, x_flat)
        attn_output = attn_output + x_flat # Residual connection
        attn_output = self.ln(attn_output)

        # Feed-forward network
        ffn_output = self.ffn(attn_output)
        ffn_output = ffn_output + attn_output # Residual connection
        ffn_output = self.ln(ffn_output)

        return ffn_output.permute(0, 2, 1).view(B, C, H, W) # Back to B, C, H, W

# === Timestep Embedding (adapted from Diffusion Model for CNF) ===
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * (-np.log(10000) / (half_dim - 1)))
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# === CNF-UNet Model (Generator with Attention and Timestep Embedding) ===
class CNF_UNet(nn.Module):
    def __init__(self, time_emb_dim=32, image_size=(64, 64)):
        super().__init__()
        self.image_size = image_size
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4), # Increased MLP capacity
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Initial convolution to increase channels
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.initial_gn = nn.GroupNorm(8, 64)

        # Downsampling blocks with increased channels and potential attention
        self.down1 = UNetBlock(64, 128) # 64x64 -> 32x32
        self.down2 = UNetBlock(128, 256) # 32x32 -> 16x16
        self.down3 = UNetBlock(256, 512, use_attention=True) # 16x16 -> 8x8 (Bottleneck)

        # Upsampling blocks
        self.up1 = UNetBlock(512 + 256, 256) # From d3 (512) + d2 (256)
        self.up2 = UNetBlock(256 + 128, 128) # From u1 (256) + d1 (128)
        self.up3 = UNetBlock(128 + 64, 64) # From u2 (128) + x0 (64)

        self.out = nn.Conv2d(64, 1, kernel_size=1) # Output 1 channel

        # Project time embedding to match feature map channels at different scales
        self.time_emb_proj_initial = nn.Linear(time_emb_dim, 64)
        self.time_emb_proj_down1 = nn.Linear(time_emb_dim, 128)
        self.time_emb_proj_down2 = nn.Linear(time_emb_dim, 256)
        self.time_emb_proj_down3 = nn.Linear(time_emb_dim, 512)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        def tile_time_emb(emb, spatial_size):
            return emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_size, spatial_size)

        x0_feat = F.relu(self.initial_gn(self.initial_conv(x)))
        x0_feat_timed = x0_feat + tile_time_emb(self.time_emb_proj_initial(t_emb), self.image_size[0])

        d1_feat = self.down1(x0_feat_timed)
        d1_feat_timed = d1_feat + tile_time_emb(self.time_emb_proj_down1(t_emb), self.image_size[0])
        
        d2_feat = self.down2(F.avg_pool2d(d1_feat_timed, 2))
        d2_feat_timed = d2_feat + tile_time_emb(self.time_emb_proj_down2(t_emb), self.image_size[0] // 2)
        
        d3_feat = self.down3(F.avg_pool2d(d2_feat_timed, 2))
        d3_feat_timed = d3_feat + tile_time_emb(self.time_emb_proj_down3(t_emb), self.image_size[0] // 4)

        u1 = F.interpolate(d3_feat_timed, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, d2_feat_timed], dim=1))

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, d1_feat_timed], dim=1))

        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        # Ensure x0_feat_timed matches u3's spatial size if necessary before concatenation
        if x0_feat_timed.shape[2:] != u3.shape[2:]:
            x0_feat_timed = F.interpolate(x0_feat_timed, size=u3.shape[2:], mode='bilinear', align_corners=False)
        u3 = self.up3(torch.cat([u3, x0_feat_timed], dim=1))
        
        # This interpolation might not be strictly necessary if the UNet architecture
        # maintains the output size correctly, but acts as a safeguard.
        if u3.shape[2:] != self.image_size:
            u3 = F.interpolate(u3, size=self.image_size, mode='bilinear', align_corners=False)

        return self.out(u3)

# === Discriminator Network (D) ===
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, img_size=64):
        super().__init__()
        # Adjusted architecture for fixed 64x64 input
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), # 8x8 -> 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0), # 4x4 -> 1x1
            nn.Sigmoid() # Output probability
        )

    def forward(self, x):
        return self.model(x).view(-1, 1) # Flatten to (batch_size, 1)