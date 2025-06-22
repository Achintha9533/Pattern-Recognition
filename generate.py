import torch
from tqdm import tqdm

@torch.no_grad()
def generate(model, z0, steps=100, device="cpu"):
    """
    Generates images from a given noise input using the trained CNF_UNet model.

    Args:
        model (torch.nn.Module): The trained CNF_UNet generator model.
        z0 (torch.Tensor): Initial noise tensor (e.g., from torch.randn_like).
                           Shape should be (batch_size, channels, height, width).
        steps (int): Number of steps for the Euler integration. Higher steps
                     generally lead to better quality but slower generation.
        device (str or torch.device): The device to perform generation on.

    Returns:
        torch.Tensor: Generated images, clamped to the range [-1, 1].
                      Shape is (batch_size, channels, height, width).
    """
    model.eval()
    z = z0.clone().to(device)
    dt = 1.0 / steps
    for i in tqdm(range(steps), desc="Generating Sample"):
        t_val = i * dt # Calculate current time in [0, 1]
        t = torch.full((z.shape[0],), t_val, device=device) # Consistent t tensor for batch
        v = model(z, t)
        z = z + v * dt # Euler step
    return z.clamp(-1, 1) # Clamp output to [-1, 1]