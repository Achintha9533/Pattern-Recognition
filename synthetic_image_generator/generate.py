"""
Image Generation Utilities.

This module provides a function for generating synthetic images using a
trained Conditional Normalizing Flow (CNF) model. It handles the iterative
generation process from an initial noise tensor through a series of steps,
leveraging the model's forward pass to refine the images.

Functions
---------
generate(model, num_samples, steps)
    Generates a specified number of synthetic images from random noise
    using the provided generative model.

Notes
-----
- The generation process is performed without gradient tracking (`@torch.no_grad()`)
  as it's an inference task.
- Progress is displayed using `tqdm`.
- Images are generated in batches to manage memory efficiently.
"""

import torch
from tqdm import tqdm
from config import device, image_size # Assuming image_size is still imported from config

@torch.no_grad()
def generate(
    model: torch.nn.Module,
    num_samples: int,
    steps: int = 200
) -> torch.Tensor:
    """
    Generates a specified number of synthetic images from random noise
    using the provided generative model.

    This function sets the model to evaluation mode, initializes a batch
    of random noise, and iteratively refines this noise through the
    model's forward pass over a defined number of steps. The process
    is performed without gradient computation and includes a progress bar.

    Parameters
    ----------
    model : torch.nn.Module
        The trained generative model (e.g., a CNF_UNet instance) used for
        image generation.
    num_samples : int
        The total number of synthetic images to generate.
    steps : int, optional
        The number of iterative steps to perform for each sample's generation.
        Defaults to 200.

    Returns
    -------
    torch.Tensor
        A tensor containing the generated images, concatenated along the batch
        dimension. The tensor is moved to CPU before being returned.
        Shape: (num_samples, C, H, W).

    Notes
    -----
    - The function operates in `torch.no_grad()` context for efficient inference.
    - Images are generated in smaller batches (`batch_size_gen`) to manage
      memory, especially when `num_samples` is large.
    - The `t` variable represents the time step in the generation process,
      scaled from 0 to 1.
    """
    model.eval() # Set the model to evaluation mode (disables dropout, batchnorm updates, etc.)
    
    # Initialize a batch of random noise tensors on the specified device.
    # This serves as the starting point for the generation process.
    initial_noise = torch.randn(num_samples, 1, *image_size).to(device) 
    
    generated_images = []
    # Generate images in smaller batches to manage GPU/CPU memory efficiently.
    batch_size_gen = 4 # Adjustable: Number of samples to process at once
    
    # Iterate through the total number of samples, processing them in batches.
    # tqdm provides a progress bar for visual feedback during generation.
    for i in tqdm(range(0, num_samples, batch_size_gen), desc="Generating images"):
        # Select a slice of the initial noise tensor for the current batch.
        noise_batch = initial_noise[i : i + batch_size_gen] 
        
        # Clone the noise batch to ensure that operations within the loop
        # do not modify the original `initial_noise` tensor.
        current_z = noise_batch.clone() 
        
        # Perform the iterative generation steps.
        for step in range(steps):
            # Calculate the time value (t) for the current step, normalized to [0, 1].
            t_val = step / (steps - 1)
            # Create a tensor for the time value, repeated for each sample in the batch.
            t = torch.tensor(t_val, device=device).repeat(current_z.shape[0])
            
            # Pass the current latent state (z) and time (t) through the model.
            # The model predicts a velocity (v) or update direction.
            v = model(current_z, t)
            
            # Update the latent state based on the predicted velocity and step size.
            current_z = current_z + v / steps
        
        # After all steps, append the generated images (moved to CPU) to the list.
        generated_images.append(current_z.cpu())
    
    # Concatenate all generated image batches into a single tensor and return.
    return torch.cat(generated_images, dim=0)