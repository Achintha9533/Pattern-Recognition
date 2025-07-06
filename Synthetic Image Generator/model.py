# Synthetic Image Generator/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def get_sinusoidal_embedding(t, embed_dim):
    """
    Generates sinusoidal positional embeddings for a given time tensor.

    Args:
        t (torch.Tensor): A 1D tensor of shape (batch_size,) representing time steps.
        embed_dim (int): The desired dimension of the embedding.

    Returns:
        torch.Tensor: The sinusoidal embedding tensor of shape (batch_size, embed_dim).
    """
    # Inverse frequency for positional encoding
    inv_freq = 1. / (10000**(torch.arange(0, embed_dim, 2).float() / embed_dim)).to(t.device)
    # Apply sin and cos to the scaled time
    sin_term = torch.sin(t.unsqueeze(1) * inv_freq)
    cos_term = torch.cos(t.unsqueeze(1) * inv_freq)
    # Concatenate sin and cos terms
    emb = torch.cat([sin_term, cos_term], dim=-1)
    # Handle cases where embed_dim is odd by padding with a zero
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TimeEmbedding(nn.Module):
    """
    Module to process sinusoidal time embeddings through a small MLP.
    """
    def __init__(self, embed_dim):
        """
        Initializes the TimeEmbedding module.

        Args:
            embed_dim (int): The dimension of the input and output embeddings.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.embed_dim = embed_dim
        logger.debug(f"TimeEmbedding initialized with embed_dim={embed_dim}.")

    def forward(self, t):
        """
        Forward pass for time embedding.

        Args:
            t (torch.Tensor): Time tensor (batch_size,).

        Returns:
            torch.Tensor: Processed time embedding (batch_size, embed_dim).
        """
        emb = get_sinusoidal_embedding(t, self.embed_dim)
        return self.fc(emb)

class ResidualBlock(nn.Module):
    """
    A standard Residual Block for the UNet, incorporating time conditioning.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim=256):
        """
        Initializes a ResidualBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_embed_dim (int): Dimension of the time embedding.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        # Shortcut connection: either identity or a 1x1 convolution if channels change
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        logger.debug(f"ResidualBlock initialized: {in_channels} -> {out_channels} channels.")

    def forward(self, x, t_emb):
        """
        Forward pass for the ResidualBlock.

        Args:
            x (torch.Tensor): Input feature map.
            t_emb (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output feature map.
        """
        B, C, H, W = x.shape
        # Project time embedding to match output channels and broadcast for addition
        time_scaled = self.time_proj(t_emb).view(B, -1, 1, 1)

        residual = self.shortcut(x) # Apply shortcut to original input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x + time_scaled) # Additive conditioning after first conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + residual) # Add shortcut connection
        return x

class SelfAttention2d(nn.Module):
    """
    Self-Attention block for 2D feature maps.
    """
    def __init__(self, in_dim):
        """
        Initializes the SelfAttention2d module.

        Args:
            in_dim (int): Number of input channels.
        """
        super().__init__()
        # Query, Key, Value convolutions
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        # Learnable gamma parameter for scaling the attention output
        self.gamma = nn.Parameter(torch.zeros(1))
        logger.debug(f"SelfAttention2d initialized with in_dim={in_dim}.")

    def forward(self, x):
        """
        Forward pass for the SelfAttention2d block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map with attention applied.
        """
        B, C, H, W = x.size()
        # Reshape for matrix multiplication
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1) # B, HW, C/8
        proj_key = self.key(x).view(B, -1, H * W) # B, C/8, HW
        # Calculate attention energy
        energy = torch.bmm(proj_query, proj_key) # B, HW, HW
        attention = torch.softmax(energy, dim=-1) # B, HW, HW (softmax over rows)
        proj_value = self.value(x).view(B, -1, H * W) # B, C, HW

        # Apply attention to value features
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, HW
        out = out.view(B, C, H, W) # Reshape back to 2D feature map

        # Add scaled attention output to original input (residual connection)
        return self.gamma * out + x

class UNetBlock(nn.Module):
    """
    A block containing multiple ResidualBlocks for either downsampling or upsampling paths in UNet.
    """
    def __init__(self, in_channels, out_channels, time_embed_dim=256, num_res_blocks=2):
        """
        Initializes a UNetBlock.

        Args:
            in_channels (int): Number of input channels for the first ResidualBlock.
            out_channels (int): Number of output channels for all ResidualBlocks in this block.
            time_embed_dim (int): Dimension of the time embedding.
            num_res_blocks (int): Number of ResidualBlocks within this UNetBlock.
        """
        super().__init__()
        layers = []
        # First residual block might change channel dimensions
        layers.append(ResidualBlock(in_channels, out_channels, time_embed_dim))
        # Subsequent residual blocks maintain channel dimensions
        for _ in range(num_res_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, time_embed_dim))
        self.block = nn.Sequential(*layers)
        logger.debug(f"UNetBlock initialized with {num_res_blocks} residual blocks: {in_channels} -> {out_channels}.")

    def forward(self, x, t_emb):
        """
        Forward pass for the UNetBlock.

        Args:
            x (torch.Tensor): Input feature map.
            t_emb (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output feature map after passing through all residual blocks.
        """
        for layer in self.block:
            x = layer(x, t_emb)
        return x

class CNF_UNet(nn.Module):
    """
    Conditional Normalizing Flow (CNF) model with a U-Net architecture.
    This model learns to predict the velocity field for transforming
    Gaussian noise into target images over a continuous time interval.
    """
    def __init__(self, time_embed_dim=256):
        """
        Initializes the CNF_UNet model.

        Args:
            time_embed_dim (int): Dimension for the time embedding.
        """
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)

        # Initial convolution to expand channels from 1 (grayscale image)
        self.inc = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Downsampling path
        # Each down block halves the spatial dimensions
        self.down1 = UNetBlock(64, 128, time_embed_dim) # Output: 128 channels, H/2, W/2
        self.down2 = UNetBlock(128, 256, time_embed_dim) # Output: 256 channels, H/4, W/4

        # Bottleneck with residual blocks and self-attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 256, time_embed_dim),
            SelfAttention2d(256),
            ResidualBlock(256, 256, time_embed_dim)
        )

        # Upsampling path
        # Up1 concatenates with skip connection from down2 (256 + 256 channels)
        self.up1 = UNetBlock(256 + 256, 128, time_embed_dim) # Output: 128 channels, H/2, W/2
        # Up2 concatenates with skip connection from down1 (128 + 128 channels)
        self.up2 = UNetBlock(128 + 128, 64, time_embed_dim) # Output: 64 channels, H, W

        # Output convolution to reduce channels back to 1 (grayscale image)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        logger.info(f"CNF_UNet initialized with time_embed_dim={time_embed_dim}.")

    def forward(self, x, t):
        """
        Forward pass of the CNF_UNet.

        Args:
            x (torch.Tensor): Input image tensor (noisy image or intermediate state).
                              Shape: (batch_size, 1, H, W)
            t (torch.Tensor): Time tensor, typically sampled from [0, 1].
                              Shape: (batch_size,)

        Returns:
            torch.Tensor: Predicted velocity field. Shape: (batch_size, 1, H, W)
        """
        # Generate time embedding
        t_emb = self.time_embedding(t) # Shape: (batch_size, time_embed_dim)

        # Initial feature extraction
        x_in = self.inc(x) # Output: (B, 64, H, W)

        # Downsampling path
        d1 = self.down1(x_in, t_emb) # Output: (B, 128, H/2, W/2)
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb) # Output: (B, 256, H/4, W/4)

        # Bottleneck
        m = self.bottleneck[0](F.avg_pool2d(d2, 2), t_emb) # Apply pooling, then first ResBlock
        m = self.bottleneck[1](m) # Apply Self-Attention
        m = self.bottleneck[2](m, t_emb) # Apply second ResBlock

        # Upsampling path
        u1 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False) # Upsample to H/4, W/4
        u1 = self.up1(torch.cat([u1, d2], dim=1), t_emb) # Concatenate with skip connection from d2

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False) # Upsample to H/2, W/2
        u2 = self.up2(torch.cat([u2, d1], dim=1), t_emb) # Concatenate with skip connection from d1

        # Final output convolution
        return self.outc(u2)