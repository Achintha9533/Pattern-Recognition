"""
Defines the Conditional Normalizing Flow (CNF) U-Net model architecture.

This module implements the neural network architecture used as the generator
in a synthetic image generation pipeline, typically a variant of a U-Net
conditioned on a time embedding. It includes helper functions for sinusoidal
positional embeddings and custom building blocks like `TimeEmbedding`,
`ResidualBlock`, `SelfAttention2d`, and `UNetBlock`.

Functions
---------
get_sinusoidal_embedding(t, embed_dim)
    Generates sinusoidal positional embeddings for a given time scalar.

Classes
-------
TimeEmbedding
    A module that transforms sinusoidal time embeddings through fully
    connected layers.
ResidualBlock
    A standard residual block with convolutional layers, batch normalization,
    ReLU activation, and time-conditional scaling/biasing.
SelfAttention2d
    A 2D self-attention mechanism to capture long-range dependencies in
    feature maps.
UNetBlock
    A sequential block composed of multiple `ResidualBlock` instances.
CNF_UNet
    The main U-Net architecture for the generative model, incorporating
    time embeddings, residual blocks, and self-attention.

Notes
-----
- The U-Net follows an encoder-decoder structure with skip connections.
- Time conditioning is applied additively within `ResidualBlock`s.
- `SelfAttention2d` is placed in the bottleneck for global context.
- Input image assumed to be 1 channel (grayscale), output is also 1 channel.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # For get_sinusoidal_embedding

# === Helper function for sinusoidal positional embedding ===
def get_sinusoidal_embedding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    Generates sinusoidal positional embeddings for a given time scalar `t`.

    This function creates a sinusoidal embedding vector for each value in `t`,
    which helps the model understand the "time" or "step" information.
    The embedding uses sine and cosine functions at different frequencies.

    Parameters
    ----------
    t : torch.Tensor
        A tensor representing the time steps (e.g., batch_size, or batch_size, 1).
        Typically, `t` values are scalars or 1D tensors.
    embed_dim : int
        The desired dimensionality of the output embedding vector. This must be
        an even number for perfect sin/cos pairing, but handles odd dimensions by padding.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(t.shape[0], embed_dim)` containing the sinusoidal embeddings.

    Examples
    --------
    >>> t = torch.tensor([0.0, 0.5, 1.0])
    >>> emb = get_sinusoidal_embedding(t, 256)
    >>> emb.shape
    torch.Size([3, 256])
    """
    # Calculate inverse frequencies for sinusoidal terms
    inv_freq = 1. / (10000**(torch.arange(0, embed_dim, 2).float() / embed_dim)).to(t.device)
    # Expand t to match the inverse frequencies for element-wise multiplication
    sin_term = torch.sin(t.unsqueeze(1) * inv_freq)
    cos_term = torch.cos(t.unsqueeze(1) * inv_freq)
    # Concatenate sine and cosine terms
    emb = torch.cat([sin_term, cos_term], dim=-1)
    # Pad if embed_dim is odd
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# === Time Embedding Module ===
class TimeEmbedding(nn.Module):
    """
    Module that processes sinusoidal time embeddings through fully connected layers.

    This module takes a time scalar (or batch of scalars), transforms it into
    a sinusoidal embedding, and then passes this embedding through a small
    feed-forward neural network to generate a richer time-conditional feature.

    Parameters
    ----------
    embed_dim : int
        The dimensionality of the sinusoidal time embedding and the output of
        the feed-forward network.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), # First linear layer expands dimension
            nn.ReLU(), # Non-linear activation
            nn.Linear(embed_dim * 4, embed_dim) # Second linear layer projects back to embed_dim
        )
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TimeEmbedding module.

        Parameters
        ----------
        t : torch.Tensor
            A tensor representing the time steps (e.g., batch_size, or batch_size, 1).

        Returns
        -------
        torch.Tensor
            The processed time embedding tensor of shape `(batch_size, embed_dim)`.
        """
        # Generate sinusoidal embedding
        emb = get_sinusoidal_embedding(t, self.embed_dim)
        # Pass through fully connected layers
        return self.fc(emb)

# === Residual Block for UNet ===
class ResidualBlock(nn.Module):
    """
    A standard Residual Block with time-conditional feature integration.

    This block applies two convolutional layers, each followed by Batch Normalization
    and ReLU activation. Time embedding is added additively after the first
    activation. It includes a shortcut connection for the residual path, which
    can be a simple identity or a 1x1 convolution if input and output channels differ.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the block.
    out_channels : int
        Number of output channels for the block.
    time_embed_dim : int, optional
        Dimensionality of the time embedding. Defaults to 256.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # Inplace operation saves memory
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Linear projection for time embedding to match feature map channels
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        # Shortcut connection: Identity if dimensions match, else 1x1 conv
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResidualBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map tensor of shape `(B, C, H, W)`.
        t_emb : torch.Tensor
            Time embedding tensor of shape `(B, time_embed_dim)`.

        Returns
        -------
        torch.Tensor
            Output feature map tensor of shape `(B, out_channels, H, W)`.
        """
        # Project time embedding and reshape to match feature map dimensions for broadcasting
        B, C, H, W = x.shape
        time_scaled = self.time_proj(t_emb).view(B, -1, 1, 1) # Reshape to (B, out_channels, 1, 1)

        residual = x # Store input for shortcut connection
        
        # First convolution, batch norm, and ReLU, with additive time conditioning
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x + time_scaled) # Additive conditioning
        
        # Second convolution, batch norm, and ReLU, with additive shortcut connection
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + self.shortcut(residual)) # Additive conditioning with shortcut
        return x

# === Self-Attention Block ===
class SelfAttention2d(nn.Module):
    """
    A 2D self-attention mechanism applied to feature maps.

    This block computes attention weights by using query, key, and value
    projections from the input feature map, allowing the model to capture
    long-range dependencies. The output is a weighted sum of values,
    combined with the original input via a learnable gamma parameter.

    Parameters
    ----------
    in_dim : int
        Number of input channels for the feature map.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        # 1x1 convolutions for query, key, and value projections
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        # Learnable scalar parameter to control the impact of attention output
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SelfAttention2d module.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map tensor of shape `(B, C, H, W)`.

        Returns
        -------
        torch.Tensor
            Output feature map tensor with self-attention applied,
            of the same shape as the input `(B, C, H, W)`.
        """
        B, C, H, W = x.size()
        
        # Reshape and permute query, key, and value for matrix multiplication
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1) # B, HW, C/8
        proj_key = self.key(x).view(B, -1, H * W) # B, C/8, HW
        
        # Compute attention energy (dot product between queries and keys)
        energy = torch.bmm(proj_query, proj_key) # B, HW, HW
        # Apply softmax to get attention probabilities
        attention = torch.softmax(energy, dim=-1) # B, HW, HW
        
        proj_value = self.value(x).view(B, -1, H * W) # B, C, HW

        # Compute weighted sum of values using attention probabilities
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # B, C, HW
        out = out.view(B, C, H, W) # Reshape back to feature map dimensions
        
        # Combine attention output with original input (residual connection)
        return self.gamma * out + x

# === UNet Block (Generator part with Residual Blocks) ===
class UNetBlock(nn.Module):
    """
    A sequential block composed of multiple `ResidualBlock` instances.

    This serves as a building block for the encoder and decoder paths of the U-Net,
    allowing for stacking of residual transformations at each resolution level.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the first residual block.
    out_channels : int
        Number of output channels for all residual blocks in this sequence.
    time_embed_dim : int, optional
        Dimensionality of the time embedding. Defaults to 256.
    num_res_blocks : int, optional
        The number of `ResidualBlock`s to stack within this block. Defaults to 2.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int = 256, num_res_blocks: int = 2):
        super().__init__()
        layers = []
        # First residual block might change channel dimensions
        layers.append(ResidualBlock(in_channels, out_channels, time_embed_dim))
        # Subsequent residual blocks maintain the `out_channels` dimension
        for _ in range(num_res_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, time_embed_dim))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNetBlock.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map tensor of shape `(B, C, H, W)`.
        t_emb : torch.Tensor
            Time embedding tensor of shape `(B, time_embed_dim)`.

        Returns
        -------
        torch.Tensor
            Output feature map tensor after passing through the sequence of
            residual blocks, of shape `(B, out_channels, H_out, W_out)`.
        """
        for layer in self.block:
            x = layer(x, t_emb)
        return x

# === CNF-UNet Model (Generator) ===
class CNF_UNet(nn.Module):
    """
    The main U-Net architecture for the Conditional Normalizing Flow (CNF)
    generative model.

    This U-Net is designed for image generation and is conditioned on a time
    embedding. It features an encoder-decoder structure with skip connections,
    incorporating `ResidualBlock`s and a `SelfAttention2d` layer in the bottleneck.

    Parameters
    ----------
    image_size : tuple of int
        The (height, width) dimensions of the input/output images.
        Required for the input and output convolutions if not explicitly defined
        by `input_channels` and `output_channels`.
        (Note: The provided code does not use image_size directly in __init__
         for channel definitions, so this parameter might be implicit in other parts
         of the project, or the provided snippet is simplified).
    input_channels : int, optional
        Number of channels in the input image. Defaults to 1 (grayscale).
    base_channels : int, optional
        The base number of channels used in the first layer of the U-Net.
        Channel counts typically scale up from this base. Defaults to 64.
    time_embed_dim : int, optional
        Dimensionality of the time embedding. Defaults to 256.
    """
    def __init__(self, image_size: Tuple[int, int] = (96, 96), input_channels: int = 1, base_channels: int = 64, time_embed_dim: int = 256):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)

        # Initial convolution to expand channels from input_channels to base_channels
        self.inc = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1) 

        # Downsampling path (Encoder)
        self.down1 = UNetBlock(base_channels, base_channels * 2, time_embed_dim) # From base_channels to 2*base_channels
        self.down2 = UNetBlock(base_channels * 2, base_channels * 4, time_embed_dim) # From 2*base_channels to 4*base_channels

        # Bottleneck with Residual Blocks and Self-Attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim), # Maintain channels
            SelfAttention2d(base_channels * 4), # Apply attention
            ResidualBlock(base_channels * 4, base_channels * 4, time_embed_dim) # Maintain channels
        )

        # Upsampling path (Decoder) with skip connections
        # Concatenate feature maps from downsampling path (e.g., d2 with m)
        self.up1 = UNetBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_embed_dim) 
        self.up2 = UNetBlock(base_channels * 2 + base_channels * 2, base_channels, time_embed_dim) 
        
        # Output convolution to map back to the desired number of image channels
        self.outc = nn.Conv2d(base_channels, input_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNF_UNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape `(batch_size, input_channels, H, W)`.
            This is typically an image corrupted with noise, which the model
            aims to denoise or transform.
        t : torch.Tensor
            Time tensor of shape `(batch_size,)` or `(batch_size, 1)`, representing
            the current time step in the generation process.

        Returns
        -------
        torch.Tensor
            The output tensor of shape `(batch_size, input_channels, H, W)`,
            representing the predicted change or a refined image at the current time step.
        """
        # Generate time embedding from the raw time tensor
        t_emb = self.time_embedding(t) # Shape: (batch_size, time_embed_dim)

        # Initial convolution to expand channels
        x_in = self.inc(x) 

        # Downsampling path (Encoder)
        d1 = self.down1(x_in, t_emb) # First down block
        d2 = self.down2(F.avg_pool2d(d1, 2), t_emb) # Second down block with pooling

        # Bottleneck (latent space)
        m = self.bottleneck[0](F.avg_pool2d(d2, 2), t_emb) # First Residual Block after pooling
        m = self.bottleneck[1](m) # Self-Attention applied
        m = self.bottleneck[2](m, t_emb) # Second Residual Block

        # Upsampling path (Decoder) with skip connections
        # Upsample bottleneck output and concatenate with skip connection from d2
        u1 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, d2], dim=1), t_emb) # Concatenate with d2 skip connection

        # Upsample and concatenate with skip connection from d1
        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, d1], dim=1), t_emb) # Concatenate with d1 skip connection

        # Final convolution to produce the output image (or prediction)
        return self.outc(u2)