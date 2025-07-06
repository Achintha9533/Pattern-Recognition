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
The U-Net structure allows for efficient capture of multi-scale features, while time conditioning
enables the model to learn the continuous transformation from noise to data.
"""

def get_sinusoidal_embedding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    Generates sinusoidal positional embeddings for a given time tensor.

    This function creates a periodic signal that helps the network understand
    the continuous nature of the time variable 't'. By using sine and cosine
    functions at different frequencies, it encodes temporal information into
    a high-dimensional vector, allowing the model to condition its output on time
    in a meaningful way, without assuming discrete time steps.

    Args:
        t (torch.Tensor): A 1D tensor of shape (batch_size,) representing time steps,
                          typically sampled uniformly from [0, 1].
        embed_dim (int): The desired dimension of the embedding. This determines
                         the number of sinusoidal features generated. Should be an even number.

    Returns:
        torch.Tensor: The sinusoidal embedding tensor of shape (batch_size, embed_dim).
                      Each row corresponds to the embedding for a given time step.

    Potential Exceptions Raised:
        - ValueError: If `embed_dim` is not an even number.

    Example of Usage:
    ```python
    import torch
    time_steps = torch.tensor([0.1, 0.5, 0.9])
    embedding = get_sinusoidal_embedding(time_steps, 256)
    print(f"Time embedding shape: {embedding.shape}") # Expected: torch.Size([3, 256])
    ```

    Relationships with other functions/modules:
    - Used by `TimestepEmbedder` to generate time embeddings for the U-Net.

    Explanation of the theory:
    - **Positional Encoding (Sinusoidal):** Originally introduced in Transformer networks
      for sequences, it's adapted here for continuous time. It allows models to use
      the relative or absolute position (time in this case) of inputs. For continuous
      values, `sin(t/10000^(2i/d_model))` and `cos(t/10000^(2i/d_model))` are used,
      where `i` is the dimension index and `d_model` is the embedding dimension.
      This enables the network to learn a wide range of transformations dependent on time.

    References for the theory:
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
      Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need.
      In Advances in neural information processing systems (pp. 5998-6008).
      (Original paper introducing sinusoidal positional encoding for sequences)
    """
    half_dim = embed_dim // 2
    # Calculate inverse frequencies for positional encoding.
    # These frequencies decrease geometrically, allowing the embedding to capture
    # both fine-grained and coarse-grained temporal dependencies.
    embeddings = torch.exp(
        torch.arange(half_dim, device=t.device) * -(torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1))
    )
    embeddings = t[:, None] * embeddings[None, :]
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    return embeddings

class TimestepEmbedder(nn.Module):
    """
    Applies a sinusoidal embedding to a timestep (t), followed by a multi-layer
    perceptron (MLP) to project it into a desired embedding dimension.

    This class takes a scalar time value, transforms it into a sinusoidal
    positional embedding, and then processes it through a small neural network
    to create a dense, learned time-dependent feature vector. This embedding
    is then added to the feature maps in the U-Net, allowing the model to be
    conditioned on the current time step of the generative process.
    """
    def __init__(self, embed_dim: int, hidden_dim: int):
        """
        Initializes the TimestepEmbedder.

        Args:
            embed_dim (int): The dimension of the initial sinusoidal embedding.
                             This will be the input dimension to the MLP.
            hidden_dim (int): The output dimension of the MLP, which will be the
                              final time embedding dimension used for conditioning.

        Returns:
            None: The constructor initializes the module.

        Potential Exceptions Raised:
            - None explicitly, but `nn.Linear` or `nn.SiLU` might raise if inputs are malformed.

        Example of Usage:
        ```python
        import torch
        embedder = TimestepEmbedder(embed_dim=256, hidden_dim=512)
        time_tensor = torch.tensor([0.5, 0.8])
        time_embedding = embedder(time_tensor)
        print(f"Time embedding shape: {time_embedding.shape}") # Expected: torch.Size([2, 512])
        ```

        Relationships with other functions/modules:
        - Uses `get_sinusoidal_embedding`.
        - Integrated into `UNetBlock` and `CNF_UNet` for time conditioning.

        Explanation of the theory:
        - **MLP (Multi-Layer Perceptron):** A simple feedforward neural network used here
          to further process the sinusoidal embedding. This allows the model to learn
          more complex, non-linear representations of time that are specifically tuned
          for the downstream task (predicting velocity fields).
        - **Conditioning:** In generative models, conditioning refers to making the
          generation process dependent on some input. Here, the time `t` is a condition.
          The `TimestepEmbedder` generates a vector representation of `t` that can be
          injected into various layers of the neural network.
        """
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.SiLU() # Swish activation function
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TimestepEmbedder.

        Args:
            t (torch.Tensor): A 1D tensor of shape (batch_size,) representing time steps.

        Returns:
            torch.Tensor: The processed time embedding tensor of shape (batch_size, hidden_dim).
        """
        t_emb = get_sinusoidal_embedding(t, self.linear1.in_features)
        t_emb = self.linear1(t_emb)
        t_emb = self.act(t_emb)
        t_emb = self.linear2(t_emb)
        return t_emb


class Block(nn.Module):
    """
    A basic convolutional block consisting of a Conv2d layer, Group Normalization,
    and a Swish (SiLU) activation function.

    This block is a fundamental building block for the U-Net architecture,
    ensuring standard operations for feature extraction and non-linearity.
    Group Normalization is preferred over Batch Normalization in some generative
    models due to its independence from batch size.
    """
    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        """
        Initializes a Block.

        Args:
            in_channels (int): Number of input channels to the convolutional layer.
            out_channels (int): Number of output channels from the convolutional layer.
            groups (int): Number of groups for Group Normalization. Defaults to 8.

        Returns:
            None: The constructor initializes the module.

        Potential Exceptions Raised:
            - ValueError: If `in_channels` or `out_channels` are not positive, or `groups` is invalid.

        Example of Usage:
        ```python
        import torch
        block = Block(in_channels=3, out_channels=64)
        dummy_input = torch.randn(1, 3, 32, 32)
        output = block(dummy_input)
        print(f"Block output shape: {output.shape}") # Expected: torch.Size([1, 64, 32, 32])
        ```

        Relationships with other functions/modules:
        - Used within `ResBlock` to create a sequence of convolutional operations.

        Explanation of the theory:
        - **Convolutional Layer (Conv2d):** Extracts spatial features from the input tensor.
        - **Group Normalization (GN):** Normalizes features across a fixed number of groups
          within each channel, independently for each sample in the batch. This can be more
          stable than Batch Normalization when using small batch sizes.
        - **SiLU (Swish) Activation:** A non-linear activation function defined as `x * sigmoid(x)`.
          It's known for performing well in deep learning models, offering a smooth, non-monotonic
          curve that helps gradients flow.

        References for the theory:
        - Wu, Y., & He, K. (2018). Group Normalization. In European Conference on Computer Vision.
        - Hendrycks, D., & Gimpel, K. (2017). Gaussian Error Linear Units (GELUs).
          (SiLU is a special case of Swish, similar to GELU).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Block.

        Args:
            x (torch.Tensor): Input feature map tensor of shape (batch_size, in_channels, H, W).

        Returns:
            torch.Tensor: Output feature map tensor of shape (batch_size, out_channels, H, W).
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    """
    A Residual Block that incorporates time embeddings and residual connections.

    This block is a core component of the U-Net, designed to facilitate training
    of deep networks by allowing gradients to flow directly through skip connections.
    It processes feature maps and time embeddings, summing the output with the
    original input (residual connection) to learn incremental changes.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, groups: int = 8):
        """
        Initializes a ResBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            time_embed_dim (int): Dimension of the time embedding.
            groups (int): Number of groups for Group Normalization. Defaults to 8.

        Returns:
            None: The constructor initializes the module.

        Potential Exceptions Raised:
            - ValueError: If channel dimensions are non-positive.

        Example of Usage:
        ```python
        import torch
        res_block = ResBlock(in_channels=64, out_channels=128, time_embed_dim=256)
        dummy_input = torch.randn(1, 64, 32, 32)
        dummy_time_emb = torch.randn(1, 256)
        output = res_block(dummy_input, dummy_time_emb)
        print(f"ResBlock output shape: {output.shape}") # Expected: torch.Size([1, 128, 32, 32])
        ```

        Relationships with other functions/modules:
        - Uses `Block` for its internal convolutional operations.
        - Integrated into `UNetBlock` and `CNF_UNet`.

        Explanation of the theory:
        - **Residual Connection (Skip Connection):** A fundamental concept in deep learning
          (ResNet) where the input of a block is added to its output. This helps to
          alleviate the vanishing gradient problem in very deep networks and allows
          the network to learn identity mappings, making it easier to learn complex functions.
        - **Time Conditioning (Feature-wise Linear Modulation - FiLM):** The time embedding
          is processed and then used to scale and shift the normalized features within the
          block. This allows the network to adapt its behavior dynamically based on the
          current time step, which is crucial for learning continuous flows.

        References for the theory:
        - He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
          In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
        - Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018).
          FiLM: Visual Reasoning with a General Conditioning Layer. In AAAI.
        """
        super().__init__()
        self.block1 = Block(in_channels, out_channels, groups)
        self.time_mlp = nn.Linear(time_embed_dim, out_channels * 2) # For scale and shift
        self.block2 = Block(out_channels, out_channels, groups)

        # 1x1 convolution for skip connection if input and output channels differ
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity() # If channels are same, no change needed

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ResBlock.

        Args:
            x (torch.Tensor): Input feature map tensor.
            t_emb (torch.Tensor): Time embedding tensor of shape (batch_size, time_embed_dim).

        Returns:
            torch.Tensor: Output feature map tensor after residual connection and time conditioning.
        """
        h = self.block1(x) # Apply first convolutional block

        # Apply time embedding: scale and shift
        # Reshape time_mlp output to (B, C*2, 1, 1) to broadcast correctly
        time_embedding_processed = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = time_embedding_processed.chunk(2, dim=1) # Split into scale and shift components
        h = h * (1 + scale) + shift # Apply FiLM conditioning

        h = self.block2(h) # Apply second convolutional block

        # Add residual connection
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    """
    Implements a simple self-attention mechanism.

    This module allows the network to weigh the importance of different spatial
    locations within a feature map when processing a specific location. It helps
    capture long-range dependencies and global contextual information, which is
    beneficial for image generation tasks.
    """
    def __init__(self, channels: int):
        """
        Initializes the SelfAttention module.

        Args:
            channels (int): The number of input channels for the feature map.

        Returns:
            None: The constructor initializes the module.

        Potential Exceptions Raised:
            - ValueError: If `channels` is not positive.

        Example of Usage:
        ```python
        import torch
        attention_block = SelfAttention(channels=128)
        dummy_input = torch.randn(1, 128, 16, 16)
        output = attention_block(dummy_input)
        print(f"SelfAttention output shape: {output.shape}") # Expected: torch.Size([1, 128, 16, 16])
        ```

        Relationships with other functions/modules:
        - Used within `CNF_UNet` in the bottleneck.

        Explanation of the theory:
        - **Self-Attention:** A mechanism that allows each element in a sequence
          (or spatial location in an image) to attend to all other elements in the
          same sequence. It computes a weighted sum of values based on similarity
          (dot product) between queries and keys. In this spatial attention, it helps
          the model focus on relevant parts of the image across potentially large distances.
        - **Query, Key, Value:** The input feature map is transformed into three
          different representations: Query (Q), Key (K), and Value (V). The attention
          scores are computed by `softmax(Q * K_transpose)`, which are then multiplied by V.

        References for the theory:
        - Vaswani, A., et al. (2017). Attention Is All You Need. (Foundational for Transformers)
        - Wang, X., Girshick, R., Gupta, A., & He, K. (2018). Non-local Neural Networks.
          In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
          (Applies self-attention to CNNs)
        """
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scaling parameter for the residual connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SelfAttention module.

        Args:
            x (torch.Tensor): Input feature map tensor of shape (batch_size, channels, H, W).

        Returns:
            torch.Tensor: Output feature map tensor after applying self-attention,
                          with the same shape as input.
        """
        batch_size, C, H, W = x.size()

        # Reshape query, key, value for matrix multiplication
        # (B, C/8, H*W)
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1) # B, H*W, C/8
        key = self.key(x).view(batch_size, -1, H * W) # B, C/8, H*W
        value = self.value(x).view(batch_size, -1, H * W) # B, C, H*W

        # Calculate attention map: (B, H*W, H*W)
        attention = torch.bmm(query, key) # Q * K_transpose
        attention = F.softmax(attention, dim=-1) # Apply softmax

        # Apply attention to values: (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1)) # V * Attention_transpose
        out = out.view(batch_size, C, H, W) # Reshape back to image format

        # Add residual connection with learned gamma scaling
        return self.gamma * out + x


class UNetBlock(nn.Module):
    """
    A single block in the U-Net architecture, composed of two Residual Blocks.

    This block represents a segment of either the downsampling or upsampling path
    of the U-Net. It applies a sequence of convolutional operations with residual
    connections and time conditioning, allowing for effective feature transformation
    at different scales.
    """
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int, groups: int = 8):
        """
        Initializes a UNetBlock.

        Args:
            in_channels (int): Number of input channels for the first ResBlock.
            out_channels (int): Number of output channels for both ResBlocks.
            time_embed_dim (int): Dimension of the time embedding.
            groups (int): Number of groups for Group Normalization within ResBlocks. Defaults to 8.

        Returns:
            None: The constructor initializes the module.

        Potential Exceptions Raised:
            - ValueError: If channel dimensions are non-positive.

        Example of Usage:
        ```python
        import torch
        unet_block = UNetBlock(in_channels=3, out_channels=64, time_embed_dim=256)
        dummy_input = torch.randn(1, 3, 64, 64)
        dummy_time_emb = torch.randn(1, 256)
        output = unet_block(dummy_input, dummy_time_emb)
        print(f"UNetBlock output shape: {output.shape}") # Expected: torch.Size([1, 64, 64, 64])
        ```

        Relationships with other functions/modules:
        - Composed of two `ResBlock` instances.
        - Used to build the main `CNF_UNet` architecture.

        Explanation of the theory:
        - **U-Net Architecture:** A convolutional network architecture originally
          developed for biomedical image segmentation. Its "U" shape comes from
          its symmetrical expanding path (upsampling) that enables precise localization,
          combined with a contracting path (downsampling) that captures context.
          Skip connections from the contracting path to the expanding path are key
          to its effectiveness. This `UNetBlock` is a building block of such a path.
        """
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_embed_dim, groups)
        self.res2 = ResBlock(out_channels, out_channels, time_embed_dim, groups)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UNetBlock.

        Args:
            x (torch.Tensor): Input feature map tensor.
            t_emb (torch.Tensor): Time embedding tensor.

        Returns:
            torch.Tensor: Output feature map tensor.
        """
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


class CNF_UNet(nn.Module):
    """
    The main Conditional Normalizing Flow U-Net model architecture.

    This model is a U-Net variant specifically designed for Conditional
    Normalizing Flows. It takes an image tensor and a time embedding as
    input and predicts a velocity field of the same spatial dimensions as
    the input image. The U-Net structure with skip connections, residual
    blocks, time conditioning, and self-attention enables it to learn
    complex, multi-scale transformations required for generative modeling.
    """
    def __init__(self, img_channels: int, time_embed_dim: int = 256, base_channels: int = 64):
        """
        Initializes the CNF_UNet model.

        Args:
            img_channels (int): Number of input and output image channels (e.g., 1 for grayscale).
            time_embed_dim (int): Dimension for the time embedding. Defaults to 256.
            base_channels (int): The number of channels in the first U-Net block.
                                 Subsequent blocks will scale these channels. Defaults to 64.

        Returns:
            None: The constructor initializes the entire U-Net architecture.

        Potential Exceptions Raised:
            - ValueError: If `img_channels` or `base_channels` are non-positive.

        Example of Usage:
        ```python
        import torch
        # For grayscale images, 1 channel, with 64x64 image size
        model = CNF_UNet(img_channels=1, time_embed_dim=256, base_channels=64)
        print(model)
        # Dummy input: batch_size=2, 1 channel, 64x64 image
        dummy_input = torch.randn(2, 1, 64, 64)
        # Dummy time: batch_size=2
        dummy_time = torch.tensor([0.1, 0.5])
        output_velocity = model(dummy_input, dummy_time)
        print(f"Output velocity field shape: {output_velocity.shape}") # Expected: torch.Size([2, 1, 64, 64])
        ```

        Relationships with other functions/modules:
        - Uses `TimestepEmbedder` for time encoding.
        - Uses `UNetBlock` for downsampling and upsampling paths.
        - Uses `ResBlock` (indirectly via `UNetBlock`) and `SelfAttention` in the bottleneck.
        - Consumed by `train.py` for training and `generate.py` for image generation.

        Explanation of the theory:
        - **U-Net for Generative Modeling:** While traditionally for segmentation,
          U-Nets are highly effective in generative tasks (e.g., in diffusion models,
          flow-based models) because their skip connections allow the network to
          propagate fine-grained spatial details from earlier layers to later layers,
          which is crucial for producing high-fidelity images.
        - **Conditional Generation:** The U-Net is conditioned on time `t`, meaning
          its output (the predicted velocity field) changes based on the current
          time step. This is fundamental for continuous normalizing flows, where
          the transformation from noise to data happens progressively over time.

        References for the theory:
        - Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
          In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 234-241).
        - Lipman, Y., Chen, R. T. Q., & Duvenaud, D. K. (2022). Flow Matching for Generative Modeling.
          (How U-Nets are adapted for flow-based models)
        """
        super().__init__()
        self.time_embedder = TimestepEmbedder(time_embed_dim, time_embed_dim)

        # Initial convolution to map input channels to base_channels
        self.inc = Block(img_channels, base_channels)

        # Downsampling path
        self.down1 = UNetBlock(base_channels, base_channels * 2, time_embed_dim)
        self.down2 = UNetBlock(base_channels * 2, base_channels * 4, time_embed_dim)

        # Bottleneck (middle part of U-Net)
        # Includes residual blocks and self-attention for global context
        self.bottleneck = nn.Sequential(
            ResBlock(base_channels * 4, base_channels * 4, time_embed_dim),
            SelfAttention(base_channels * 4), # Apply self-attention at the bottleneck
            ResBlock(base_channels * 4, base_channels * 4, time_embed_dim),
        )

        # Upsampling path
        self.up1 = UNetBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_embed_dim) # Skip connection from down2
        self.up2 = UNetBlock(base_channels * 2 + base_channels, base_channels, time_embed_dim) # Skip connection from down1

        # Output convolution to map back to original image channels
        self.outc = nn.Conv2d(base_channels, img_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNF_UNet model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, img_channels, H, W).
                              Expected pixel range: [-1, 1].
            t (torch.Tensor): Time tensor of shape (batch_size,). Values typically in [0, 1].

        Returns:
            torch.Tensor: Predicted velocity field tensor of shape (batch_size, img_channels, H, W).
                          This has the same shape and roughly same value range as the input image.
        """
        # Embed the time step
        t_emb: torch.Tensor = self.time_embedder(t) # Output: (B, time_embed_dim)

        # Downsampling path:
        # Pass through the initial convolution (inc).
        x1: torch.Tensor = self.inc(x) # Output: (B, base_channels, H, W)

        # Pass through the first UNetBlock (down1).
        # AvgPool2d reduces spatial dimensions by half.
        d1: torch.Tensor = self.down1(F.avg_pool2d(x1, 2), t_emb) # Output: (B, base_channels*2, H/2, W/2)
        # Apply average pooling and then pass through the second UNetBlock (down2).
        d2: torch.Tensor = self.down2(F.avg_pool2d(d1, 2), t_emb) # Output: (B, base_channels*4, H/4, W/4)

        # Bottleneck:
        # Apply average pooling, then pass through the bottleneck sequence (ResBlocks + SelfAttention).
        m: torch.Tensor = self.bottleneck[0](F.avg_pool2d(d2, 2), t_emb) # First ResBlock: (B, base_channels*4, H/8, W/8)
        m = self.bottleneck[1](m) # Self-Attention: (B, base_channels*4, H/8, W/8)
        m = self.bottleneck[2](m, t_emb) # Second ResBlock: (B, base_channels*4, H/8, W/8)

        # Upsampling path:
        # Upsample bottleneck output (m) to match spatial dimensions of d2.
        u1: torch.Tensor = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenate with skip connection from d2 and pass through up1 block.
        # The concatenation doubles the channels before processing in up1.
        u1 = self.up1(torch.cat([u1, d2], dim=1), t_emb) # Output: (B, base_channels*2, H/4, W/4)

        # Upsample u1 to match spatial dimensions of d1.
        u2: torch.Tensor = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenate with skip connection from d1 and pass through up2 block.
        u2 = self.up2(torch.cat([u2, d1], dim=1), t_emb) # Output: (B, base_channels, H/2, W/2)

        # Upsample u2 to match spatial dimensions of original input x.
        # This is the final spatial upsampling step.
        u3: torch.Tensor = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        # Concatenate with skip connection from initial conv (x1) and pass through output conv.
        # Note: x1 is not downsampled, so its dimensions are (B, base_channels, H, W).
        # We concatenate u3 with x1 to get the final full-resolution features.
        # The outc then maps these features to the final image channels.
        output: torch.Tensor = self.outc(torch.cat([u3, x1], dim=1)) # Output: (B, img_channels, H, W)

        return output