# Synthetic Image Generator/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module defines the neural network architecture for the Conditional Normalizing Flow (CNF)
model, specifically a U-Net based generator. It includes components for time embedding,
residual connections, and self-attention, all crucial for learning complex image distributions.
"""

def get_sinusoidal_embedding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    Generates sinusoidal positional embeddings for a given time tensor.
    This function creates a periodic signal that helps the network understand
    the continuous nature of the time variable 't'.

    Args:
        t (torch.Tensor): A 1D tensor of shape (batch_size,) representing time steps,
                          typically sampled uniformly from [0, 1].
        embed_dim (int): The desired dimension of the embedding. This determines
                         the number of sinusoidal features generated.

    Returns:
        torch.Tensor: The sinusoidal embedding tensor of shape (batch_size, embed_dim).
                      Each row corresponds to the embedding for a given time step.
    """
    # Calculate inverse frequencies for positional encoding.
    # These frequencies decrease geometrically, allowing the embedding to capture
    # both fine-grained and coarse-grained temporal information.
    inv_freq: torch.Tensor = 1. / (10000**(torch.arange(0, embed_dim, 2).float() / embed_dim)).to(t.device)

    # Apply sine and cosine functions to the scaled time.
    # Unsqueezing 't' adds a dimension for broadcasting with inv_freq.
    sin_term: torch.Tensor = torch.sin(t.unsqueeze(1) * inv_freq)
    cos_term: torch.Tensor = torch.cos(t.unsqueeze(1) * inv_freq)

    # Concatenate sine and cosine terms to form the full embedding.
    emb: torch.Tensor = torch.cat([sin_term, cos_term], dim=-1)

    # Handle cases where embed_dim is odd by padding with a zero at the end.
    # This ensures the output dimension always matches `embed_dim`.
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class TimeEmbedding(nn.Module):
    """
    Module to process sinusoidal time embeddings through a small Multi-Layer Perceptron (MLP).
    This MLP transforms the raw sinusoidal embedding into a more expressive feature vector
    that can be effectively used for conditioning the U-Net's feature maps.
    """
    def __init__(self, embed_dim: int):
        """
        Initializes the TimeEmbedding module.

        Args:
            embed_dim (int): The dimension of the input sinusoidal embedding and
                             the output processed time embedding.
        """
        super().__init__()
        self.fc: nn.Sequential = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # Expand dimension
            nn.ReLU(inplace=True),               # Non-linear activation
            nn.Linear(embed_dim * 4, embed_dim)  # Project back to original dimension
        )
        self.embed_dim: int = embed_dim
        logger.debug(f"TimeEmbedding initialized with embed_dim={embed_dim}.")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for time embedding.

        Args:
            t (torch.Tensor): Time tensor of shape (batch_size,), typically
                              containing values sampled from [0, 1].

        Returns:
            torch.Tensor: Processed time embedding tensor of shape (batch_size, embed_dim).
        """
        # Generate raw sinusoidal embedding
        emb: torch.Tensor = get_sinusoidal_embedding(t, self.embed_dim)
        # Pass through the MLP
        return self.fc(emb)

class ResidualBlock(nn.Module):
    """
    A standard Residual Block for the U-Net architecture, incorporating time conditioning.
    This block helps in training deeper networks by providing a shortcut connection,
    and it integrates time information additively into the feature maps.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 256):
        """
        Initializes a ResidualBlock.

        Args:
            in_channels (int): Number of input channels to the block.
            out_channels (int): Number of output channels from the block.
            time_embed_dim (int): Dimension of the time embedding tensor that will
                                  be used for conditioning.
        """
        super().__init__()
        self.conv1: nn.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)
        self.relu: nn.ReLU = nn.ReLU(inplace=True) # Use inplace for memory efficiency
        self.conv2: nn.Conv2d = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2: nn.BatchNorm2d = nn.BatchNorm2d(out_channels)

        # Linear layer to project time embedding to the feature map's channel dimension
        self.time_proj: nn.Linear = nn.Linear(time_embed_dim, out_channels)
        
        # Shortcut connection: if input and output channels differ, use a 1x1 convolution
        # to match dimensions for the residual addition; otherwise, use identity.
        if in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        logger.debug(f"ResidualBlock initialized: {in_channels} -> {out_channels} channels.")

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResidualBlock.

        Args:
            x (torch.Tensor): Input feature map tensor of shape (B, C, H, W).
            t_emb (torch.Tensor): Time embedding tensor of shape (B, time_embed_dim).

        Returns:
            torch.Tensor: Output feature map tensor of shape (B, out_channels, H, W).
        """
        B, C, H, W = x.shape
        # Project time embedding to match output channels and reshape to (B, out_channels, 1, 1)
        # for broadcasting across spatial dimensions (H, W) during addition.
        time_scaled: torch.Tensor = self.time_proj(t_emb).view(B, -1, 1, 1)

        # Apply shortcut connection to the original input
        residual: torch.Tensor = self.shortcut(x)

        # First convolutional layer, batch normalization, and ReLU activation.
        # Time conditioning is applied here.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x + time_scaled) # Additive conditioning

        # Second convolutional layer and batch normalization.
        x = self.conv2(x)
        x = self.bn2(x)

        # Add the shortcut connection and apply final ReLU activation.
        x = self.relu(x + residual)
        return x

class SelfAttention2d(nn.Module):
    """
    Self-Attention block for 2D feature maps. This mechanism allows the model
    to weigh the importance of different spatial locations when processing features,
    capturing long-range dependencies and enhancing global context understanding.
    """
    def __init__(self, in_dim: int):
        """
        Initializes the SelfAttention2d module.

        Args:
            in_dim (int): Number of input channels (and output channels) for the attention block.
        """
        super().__init__()
        # Define 1x1 convolutions for Query, Key, and Value projections.
        # Query and Key typically project to a smaller dimension (e.g., in_dim // 8)
        # to reduce computational complexity.
        self.query: nn.Conv2d = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key: nn.Conv2d = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value: nn.Conv2d = nn.Conv2d(in_dim, in_dim, 1)
        
        # Learnable scalar parameter `gamma` to control the contribution of the
        # attention mechanism. It's initialized to zero, allowing the network
        # to initially rely on the identity mapping and gradually learn attention.
        self.gamma: nn.Parameter = nn.Parameter(torch.zeros(1))
        logger.debug(f"SelfAttention2d initialized with in_dim={in_dim}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SelfAttention2d block.

        Args:
            x (torch.Tensor): Input feature map tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output feature map tensor of shape (B, C, H, W)
                          with attention applied.
        """
        B, C, H, W = x.size()

        # Project input to query, key, and value spaces.
        # Reshape to (B, C_proj, H*W) for matrix multiplication and permute query.
        proj_query: torch.Tensor = self.query(x).view(B, -1, H * W).permute(0, 2, 1) # B, HW, C/8
        proj_key: torch.Tensor = self.key(x).view(B, -1, H * W)                     # B, C/8, HW
        proj_value: torch.Tensor = self.value(x).view(B, -1, H * W)                 # B, C, HW

        # Calculate attention energy (dot product between query and key).
        energy: torch.Tensor = torch.bmm(proj_query, proj_key) # B, HW, HW

        # Apply softmax to get attention weights.
        attention: torch.Tensor = torch.softmax(energy, dim=-1) # B, HW, HW (softmax over rows)

        # Apply attention weights to the value features.
        out: torch.Tensor = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, HW

        # Reshape the output back to a 2D feature map format.
        out = out.view(B, C, H, W)

        # Add the scaled attention output to the original input (residual connection).
        # The gamma parameter controls the strength of the attention's contribution.
        return self.gamma * out + x

class UNetBlock(nn.Module):
    """
    A composite block containing multiple ResidualBlocks for either downsampling
    or upsampling paths within the U-Net architecture. This modular design
    simplifies the construction of the U-Net's encoder and decoder.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 256, num_res_blocks: int = 2):
        """
        Initializes a UNetBlock.

        Args:
            in_channels (int): Number of input channels for the first ResidualBlock in this sequence.
            out_channels (int): Number of output channels for all ResidualBlocks within this block.
            time_embed_dim (int): Dimension of the time embedding.
            num_res_blocks (int): The number of `ResidualBlock` instances to stack within this UNetBlock.
        """
        super().__init__()
        layers: list[nn.Module] = []
        # The first residual block might change channel dimensions from `in_channels` to `out_channels`.
        layers.append(ResidualBlock(in_channels, out_channels, time_embed_dim))
        # Subsequent residual blocks (if any) maintain the `out_channels` dimension.
        for _ in range(num_res_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, time_embed_dim))
        self.block: nn.Sequential = nn.Sequential(*layers)
        logger.debug(f"UNetBlock initialized with {num_res_blocks} residual blocks: {in_channels} -> {out_channels}.")

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNetBlock.

        Args:
            x (torch.Tensor): Input feature map tensor.
            t_emb (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output feature map tensor after passing through all
                          stacked residual blocks.
        """
        for layer in self.block:
            # Each ResidualBlock processes the feature map and incorporates time conditioning.
            x = layer(x, t_emb)
        return x

class CNF_UNet(nn.Module):
    """
    Conditional Normalizing Flow (CNF) model with a U-Net architecture.
    This model is designed to learn and predict the velocity field that transforms
    a simple prior distribution (Gaussian noise) into the complex target data
    distribution (Lung CT images) over a continuous time interval.

    The U-Net structure allows it to capture multi-scale features, while time
    conditioning enables it to model the continuous flow dynamics.
    """
    def __init__(self, time_embed_dim: int = 256):
        """
        Initializes the CNF_UNet model.

        Args:
            time_embed_dim (int): The dimension for the time embedding, which will
                                  be used to condition the network's layers.
        """
        super().__init__()
        # Module to generate and process time embeddings.
        self.time_embedding: TimeEmbedding = TimeEmbedding(time_embed_dim)

        # Initial convolution layer: expands the single channel (grayscale image)
        # to the initial feature map depth of 64 channels.
        self.inc: nn.Conv2d = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # Downsampling path (Encoder) of the U-Net.
        # Each down block typically reduces spatial dimensions (e.g., by pooling)
        # and increases channel dimensions.
        self.down1: UNetBlock = UNetBlock(64, 128, time_embed_dim) # Output: 128 channels, H/2, W/2 (after pooling in forward)
        self.down2: UNetBlock = UNetBlock(128, 256, time_embed_dim) # Output: 256 channels, H/4, W/4 (after pooling in forward)

        # Bottleneck section: contains residual blocks and a self-attention layer.
        # Self-attention helps capture global dependencies in the most compressed feature space.
        self.bottleneck: nn.Sequential = nn.Sequential(
            ResidualBlock(256, 256, time_embed_dim), # First residual block in bottleneck
            SelfAttention2d(256),                     # Self-attention layer
            ResidualBlock(256, 256, time_embed_dim)  # Second residual block in bottleneck
        )

        # Upsampling path (Decoder) of the U-Net.
        # Each up block typically increases spatial dimensions (e.g., by interpolation)
        # and reduces channel dimensions, incorporating skip connections.
        # The input channels for up blocks account for concatenated skip connections.
        self.up1: UNetBlock = UNetBlock(256 + 256, 128, time_embed_dim) # Input: Bottleneck output (256) + skip from down2 (256)
        self.up2: UNetBlock = UNetBlock(128 + 128, 64, time_embed_dim)  # Input: Up1 output (128) + skip from down1 (128)

        # Output convolution layer: reduces the feature map channels back to 1,
        # representing the predicted velocity field for a grayscale image.
        self.outc: nn.Conv2d = nn.Conv2d(64, 1, kernel_size=1)
        logger.info(f"CNF_UNet initialized with time_embed_dim={time_embed_dim}.")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNF_UNet model.

        Args:
            x (torch.Tensor): Input image tensor (e.g., noisy image `z_t` or intermediate state `x_t`).
                              Expected shape: (batch_size, 1, H, W).
            t (torch.Tensor): Time tensor, typically sampled from [0, 1].
                              Expected shape: (batch_size,). This tensor is used to
                              generate time-dependent conditioning for the network.

        Returns:
            torch.Tensor: Predicted velocity field `v_pred`. The output tensor has the
                          same shape as the input image (batch_size, 1, H, W), representing
                          the estimated direction of flow from noise to data at time `t`.
        """
        # Generate time embedding from the input time tensor.
        t_emb: torch.Tensor = self.time_embedding(t) # Shape: (batch_size, time_embed_dim)

        # Initial feature extraction: apply the first convolution.
        x_in: torch.Tensor = self.inc(x) # Output: (B, 64, H, W)

        # Downsampling path:
        # Pass through the first UNetBlock (down1).
        d1: torch.Tensor = self.down1(x_in, t_emb) # Output: (B, 128, H, W)
        # Apply average pooling and then pass through the second UNetBlock (down2).
        d2: torch.Tensor = self.down2(F.avg_pool2d(d1, 2), t_emb) # Output: (B, 256, H/2, W/2)

        # Bottleneck:
        # Apply average pooling, then pass through the first residual block in the bottleneck.
        m: torch.Tensor = self.bottleneck[0](F.avg_pool2d(d2, 2), t_emb) # Output: (B, 256, H/4, W/4)
        # Apply self-attention.
        m = self.bottleneck[1](m)
        # Pass through the second residual block in the bottleneck.
        m = self.bottleneck[2](m, t_emb)

        # Upsampling path:
        # Upsample bottleneck output (m) to match spatial dimensions of d2.
        u1: torch.Tensor = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenate with skip connection from d2 and pass through up1 block.
        u1 = self.up1(torch.cat([u1, d2], dim=1), t_emb) # Output: (B, 128, H/2, W/2)

        # Upsample u1 to match spatial dimensions of d1.
        u2: torch.Tensor = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenate with skip connection from d1 and pass through up2 block.
        u2 = self.up2(torch.cat([u2, d1], dim=1), t_emb) # Output: (B, 64, H, W)

        # Final output convolution: project feature map back to 1 channel.
        return self.outc(u2)
