# Synthetic Image Generator/generate.py

import torch
import torch.nn as nn
import logging
from typing import Union

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

    Args:
        model (nn.Module): The trained generator model (CNF_UNet instance)
                           that predicts the velocity field.
        initial_noise (torch.Tensor): A batch of initial Gaussian noise tensors.
                                      This serves as the starting point for the generation process.
                                      Shape: (batch_size, C, H, W).
        steps (int): The number of discrete steps to use for the Euler integration.
                     More steps generally lead to higher quality but slower generation.
        device (Union[str, torch.device]): The device ('cuda' or 'cpu') on which
                                           to perform image generation.

    Returns:
        torch.Tensor: A batch of generated synthetic image tensors.
                      Shape: (batch_size, C, H, W).
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
    
    logger.info("Image generation complete.")
    return z
