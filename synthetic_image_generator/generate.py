# Synthetic Image Generator/generate.py

import torch
import torch.nn as nn
import logging
from typing import Union
from tqdm import tqdm

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module provides the core functionality for generating synthetic images
using a trained Conditional Normalizing Flow (CNF) model. It implements
the reverse process of the flow, transforming initial Gaussian noise
into realistic images through a series of integration steps.
"""

@torch.no_grad() # Decorator to disable gradient calculations for inference, saving memory and speeding up.
def generate_images(
    model: nn.Module,
    initial_noise: torch.Tensor,
    steps: int = 200,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Generates synthetic images by integrating the predicted velocity field
    from an initial Gaussian noise distribution to the target data distribution.
    This process simulates the reverse flow learned by the CNF model.

    The generation process starts with random Gaussian noise and iteratively
    updates it by moving along the velocity field predicted by the CNF model
    at each time step. The Euler integration method is used for this process,
    approximating the continuous flow. Disabling gradient calculation (`torch.no_grad()`)
    is essential for efficient inference.

    Args:
        model (nn.Module): The trained generator model (CNF_UNet instance)
                           that predicts the velocity field. It must be in evaluation mode.
        initial_noise (torch.Tensor): A batch of initial Gaussian noise tensors.
                                      This serves as the starting point for the generation process.
                                      Shape: (batch_size, C, H, W). Expected values: standard normal.
        steps (int): The number of discrete steps to use for the Euler integration.
                     More steps generally lead to higher quality generated images
                     but also increase generation time. Defaults to 200.
        device (Union[str, torch.device]): The device ('cpu' or 'cuda') on which
                                           to perform image generation. Defaults to 'cpu'.

    Returns:
        torch.Tensor: A batch of generated synthetic image tensors.
                      Shape: (batch_size, C, H, W). Pixel values are in the
                      same range as the training data (e.g., [-1, 1] if normalized).

    Potential Exceptions Raised:
        - RuntimeError: If tensors are not on the specified `device` or if
                        CUDA operations fail.
        - ValueError: If `steps` is less than 1.

    Example of Usage:
    ```python
    import torch
    from .model import CNF_UNet # Assuming CNF_UNet is defined in model.py
    # from . import config # If using config for image_size, num_generated_samples, generation_steps

    # Initialize a dummy model (replace with your trained model loading)
    # generator_model = CNF_UNet(img_channels=1, time_embed_dim=256, base_channels=64)
    # generator_model.load_state_dict(torch.load("path/to/your/generator_final.pth"))
    # generator_model.eval()

    # Create initial noise
    # noise_shape = (config.NUM_GENERATED_SAMPLES, 1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1])
    # noise = torch.randn(noise_shape)

    # generated_imgs = generate_images(generator_model, noise, steps=config.GENERATION_STEPS, device='cpu')
    # print(f"Generated images shape: {generated_imgs.shape}")
    ```

    Relationships with other functions/modules:
    - Calls `model` (the CNF_UNet instance) in evaluation mode to predict velocities.
    - Used by `main.py` to produce synthetic images after training.
    - `config.py` provides `steps` (GENERATION_STEPS) and `initial_noise` dimensions.

    Explanation of the theory:
    - **Euler Integration:** A simple numerical method for approximating the solution
      to a first-order ordinary differential equation (ODE). In CNFs, the model learns
      a velocity field `v(x, t)` that describes how data points `x` evolve over time `t`.
      Image generation is the reverse process: starting from `z0` (noise) at `t=0`,
      we integrate `dz/dt = v(z, t)` to find `z1` (image) at `t=1`.
      The update rule is approximately: `z(t + dt) = z(t) + v(z(t), t) * dt`,
      where `dt = 1 / steps`.
    - **Conditional Normalizing Flows (CNFs):** A class of generative models that
      learn a continuous, invertible transformation from a simple base distribution
      (e.g., Gaussian noise) to a complex data distribution. This is achieved by
      learning a time-dependent vector field (velocity field) that governs the flow
      of samples in the data space. Image generation involves integrating this
      velocity field.

    References for the theory:
    - Grathwohl, J., Chen, R. T. Q., Bettencourt, J., Hälvä, I., & Duvenaud, D. K. (2018).
      Fjord: https://arxiv.org/abs/1806.02373 (Continuous Normalizing Flows)
    - Lipman, Y., Chen, R. T. Q., & Duvenaud, D. K. (2022). Flow Matching for Generative Modeling.
      In International Conference on Learning Representations. https://arxiv.org/abs/2210.02747
    """
    logger.info(f"Starting image generation with {steps} steps on device: {device}.")
    model.eval() # Set the model to evaluation mode (disables dropout, batch norm updates)

    # Initialize the current state 'z' with the initial noise.
    # Clone to ensure the original `initial_noise` tensor is not modified.
    z: torch.Tensor = initial_noise.clone().to(device)

    # Perform Euler integration to transform noise into images.
    # The loop iterates from t=0 to t=1 (approximately) in `steps` increments.
    for i in tqdm(range(steps), desc="Generating Images"):
        # Calculate the current time 't' in the [0, 1] range.
        # It's crucial that 't' is a tensor with batch_size elements for conditioning.
        t_val: float = i / (steps - 1) if steps > 1 else 0.0 # Avoid division by zero if steps=1
        t: torch.Tensor = torch.full((z.shape[0],), t_val, device=device) # Create a tensor of `t_val` for each sample in batch

        # Predict the velocity field 'v' at the current state 'z' and time 't'.
        v: torch.Tensor = model(z, t)

        # Update the state 'z' by moving it along the predicted velocity field.
        # The step size is v / steps.
        z = z + v / steps

    return z