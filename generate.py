import torch

# Assuming model classes are imported from model.py
# from model import CNF_UNet

@torch.no_grad()
def generate(model, z0, steps=200):
    """
    Generates images from initial noise (z0) using the trained generator model.

    Args:
        model (CNF_UNet): The trained generator model.
        z0 (torch.Tensor): Initial noise tensor.
        steps (int): Number of steps to integrate the flow.

    Returns:
        torch.Tensor: Generated images.
    """
    model.eval() # Set model to evaluation mode
    z = z0.clone().to(z0.device) # Ensure z is on the correct device
    for i in range(steps):
        t = torch.tensor(i / (steps - 1), device=z.device).repeat(z.shape[0])
        v = model(z, t)
        z = z + v / steps
    return z