# Synthetic Image Generator/generate.py

import torch
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

@torch.no_grad() # Decorator to disable gradient calculations for inference
def generate_images(model, initial_noise, steps=200, device='cpu'):
    """
    Generates images by simulating the CNF's forward pass from noise to data.

    Args:
        model (torch.nn.Module): The trained generator (CNF_UNet) model.
        initial_noise (torch.Tensor): The starting noise tensor (e.g., random Gaussian noise).
                                      Shape: (batch_size, 1, H, W)
        steps (int): Number of discrete steps to simulate the continuous flow.
        device (str or torch.device): The device to run generation on ('cuda' or 'cpu').

    Returns:
        torch.Tensor: The generated image tensor.
    """
    model.eval() # Set model to evaluation mode
    logger.info(f"Starting image generation with {steps} steps on device: {device}.")

    # Clone noise and move to device
    z = initial_noise.clone().to(device)

    # Iterate through the steps to simulate the flow
    for i in range(steps):
        # Calculate current time 't' for the current step
        t_val = i / (steps - 1)
        # Create a time tensor for the batch
        t = torch.tensor(t_val, device=device).repeat(z.shape[0])

        # Predict the velocity field at the current state 'z' and time 't'
        v = model(z, t)
        # Update the state 'z' by moving along the predicted velocity field
        # The division by 'steps' scales the velocity for a discrete step
        z = z + v / steps

    logger.info("Image generation complete.")
    return z