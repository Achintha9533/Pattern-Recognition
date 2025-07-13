"""
Training loop for the Conditional Normalizing Flow (CNF) model.

This module implements the training procedure for a generative model based on
the Flow Matching objective. It guides the generator to learn a velocity field
that effectively transforms samples from a simple base distribution (Gaussian noise)
into samples resembling the complex target data distribution. The training
process minimizes the L2 distance between the generator's predicted velocity
and a target velocity derived from linear interpolation.

Functions
---------
train_model(generator_model, dataloader, optimizer_gen, epochs, device)
    Executes the main training loop for the CNF_UNet model using the Flow Matching objective.

Notes
-----
- The training process utilizes a continuous-time formulation, where a random
  time 't' is sampled for each interpolation.
- The core of Flow Matching lies in training the model to predict the
  instantaneous velocity required to move from a noisy state `xt` towards
  the real data `x1_real` along a simple path (here, a straight line).
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Any, Union, List

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_model(
    generator_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer_gen: torch.optim.Optimizer,
    epochs: int = 100,
    device: Union[str, torch.device] = 'cpu'
) -> Dict[str, Any]:
    """
    Trains the generator model (CNF_UNet) using the Flow Matching objective.

    The Flow Matching objective trains the model to predict the velocity field
    that transports samples from a simple base distribution (Gaussian noise)
    to the target data distribution. This is achieved by interpolating between
    noise and real data at random time steps and training the model to predict
    the difference vector (the "true" velocity for linear interpolation).
    The training loop iterates over a specified number of epochs, processes
    data in batches, and updates model weights using an optimizer.

    Parameters
    ----------
    generator_model : torch.nn.Module
        The generator model (an instance of CNF_UNet) to be trained. This model
        is expected to predict a velocity field.
    dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of (noise, real_image) pairs.
    optimizer_gen : torch.optim.Optimizer
        The optimizer responsible for updating the `generator_model`'s parameters
        (e.g., `torch.optim.Adam`).
    epochs : int, optional
        The total number of training epochs. Defaults to 100.
    device : Union[str, torch.device], optional
        The device ('cpu' or 'cuda') on which to perform training.
        Defaults to 'cpu'.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing training statistics, specifically:
        - 'gen_flow_losses' (List[float]): A list of average Flow Matching
          losses for each epoch.

    Raises
    ------
    RuntimeError
        If PyTorch operations fail (e.g., out of GPU memory, invalid tensor operations).
    ValueError
        If `epochs` is non-positive or `dataloader` is empty.
    Exception
        Any exceptions occurring during data loading from `dataloader` or
        during model forward/backward passes.

    See Also
    --------
    - `model.CNF_UNet`: The architecture of the generator model.
    - `torch.nn.functional.mse_loss`: Used for the loss calculation.

    Notes
    -----
    **Explanation of Flow Matching Theory:**
    - **Flow Matching:** A recent and powerful technique for training continuous
      normalizing flows and diffusion models. Instead of directly learning the
      data likelihood (which can be complex), it learns a conditional vector field
      (velocity field) that transports samples from a simple distribution (noise)
      to the data distribution. The key idea is to define a "target" velocity
      field for simple paths (like straight lines between noise and data) and
      then train the model to predict this target velocity. This simplifies
      the optimization problem and improves training stability.
    - **Linear Interpolation:** In this implementation, the path between noise `z0`
      and real data `x1_real` is assumed to be a straight line. For a point `xt`
      on this line at time `t`, the "true" velocity `v_target` is simply
      `x1_real - z0`.
    - **Mean Squared Error (MSE) Loss:** Used as the objective function to minimize
      the difference between the model's predicted velocity `v_pred` and the
      target velocity `v_target`.

    References
    ----------
    - Lipman, Y., Chen, R. T. Q., & Duvenaud, D. K. (2022). Flow Matching for Generative Modeling.
      In International Conference on Learning Representations. https://arxiv.org/abs/2210.02747

    Examples
    --------
    ```python
    import torch
    import torch.optim as optim
    from pathlib import Path
    from synthetic_image_generator.model import CNF_UNet # Assuming CNF_UNet is defined
    from synthetic_image_generator.dataset import LungCTWithGaussianDataset # Assuming dataset is defined
    from torchvision import transforms as T

    # Dummy setup (replace with actual data and model initialization)
    img_size = (96, 96)
    data_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Ensure a dummy_data directory exists or point to actual data
    # Path("path/to/dummy_data").mkdir(parents=True, exist_ok=True)
    # dataset = LungCTWithGaussianDataset(
    #     base_dir=Path("path/to/dummy_data"), # Replace with actual path if needed
    #     transform=data_transform,
    #     image_size=img_size
    # )
    # # Handle empty dataset case for example
    # if len(dataset) == 0:
    #     print("Warning: Dummy dataset is empty. Cannot run example training.")
    #     # sys.exit(1) or raise ValueError
    #     dataloader = [] # Empty dataloader for example to pass
    # else:
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

    # generator = CNF_UNet(time_embed_dim=256)
    # optimizer = optim.Adam(generator.parameters(), lr=1e-4)

    # # Ensure dataloader is not empty for the actual training call
    # if dataloader:
    #     training_stats = train_model(
    #         generator_model=generator,
    #         dataloader=dataloader,
    #         optimizer_gen=optimizer,
    #         epochs=1, # Reduced epochs for quick example
    #         device='cpu'
    #     )
    #     print("Training completed. Losses:", training_stats['gen_flow_losses'])
    # else:
    #     print("Skipping training example due to empty dataloader.")
    ```
    """
    logger.info(f"Starting training for {epochs} epochs on device: {device}")
    
    if epochs <= 0:
        raise ValueError("Number of epochs must be positive.")
    if not dataloader: # Checks if dataloader iterable is empty or None
        raise ValueError("Dataloader is empty or invalid. Cannot train model.")

    generator_model.to(device) # Move the generator model to the specified device
    generator_model.train() # Set the model to training mode (enables dropout, batchnorm updates, etc.)

    gen_flow_losses: List[float] = [] # List to store average flow matching loss per epoch

    for epoch in range(epochs):
        epoch_gen_flow_loss = 0.0 # Accumulator for loss over the current epoch
        
        # Iterate over batches in the dataloader. tqdm provides a progress bar.
        for batch_idx, (noise_batch, real_images_batch) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move the noise (z0) and real images (x1_real) to the appropriate device
            z0: torch.Tensor = noise_batch.to(device) # Initial noise sample
            x1_real: torch.Tensor = real_images_batch.to(device) # Corresponding real data sample

            # Zero out gradients accumulated from previous steps for the generator's optimizer
            optimizer_gen.zero_grad()

            # 1. Sample time 't' uniformly from [0, 1] for each sample in the batch.
            # `t` needs to be a 1D tensor with `batch_size` elements for conditioning.
            t: torch.Tensor = torch.rand(z0.shape[0], device=device) # t ~ U(0, 1)

            # 2. Linearly interpolate between z0 (noise) and x1_real (real data) at time t.
            # This creates 'xt', a point along the linear path.
            # `t.view(-1, 1, 1, 1)` reshapes 't' to allow broadcasting across image dimensions.
            xt: torch.Tensor = t.view(-1, 1, 1, 1) * x1_real + (1 - t).view(-1, 1, 1, 1) * z0

            # 3. Define the target velocity vector (v_target).
            # For linear interpolation, the true velocity at any point `t` along the path
            # from `z0` to `x1_real` is simply the difference vector `x1_real - z0`.
            v_target: torch.Tensor = (x1_real - z0)

            # 4. The generator predicts the velocity field (v_pred) at the interpolated state 'xt' and time 't'.
            v_pred: torch.Tensor = generator_model(xt, t)

            # 5. Calculate the loss as the Mean Squared Error between the predicted and target velocities.
            loss_flow_matching: torch.Tensor = F.mse_loss(v_pred, v_target)

            # Assign the flow matching loss as the total loss for the generator.
            total_gen_loss: torch.Tensor = loss_flow_matching
            total_gen_loss.backward() # Perform backpropagation to compute gradients
            optimizer_gen.step() # Update model parameters based on computed gradients

            # Accumulate the loss for the current batch for reporting the epoch's average loss.
            epoch_gen_flow_loss += loss_flow_matching.item()

        # Calculate the average loss for the current epoch.
        avg_gen_flow_loss: float = epoch_gen_flow_loss / len(dataloader)
        gen_flow_losses.append(avg_gen_flow_loss) # Store the average loss

        # Log the average loss for the epoch.
        logger.info(f"Epoch {epoch+1}: G_Flow_Loss={avg_gen_flow_loss:.6f}")

    logger.info("Training complete.")
    return {
        'gen_flow_losses': gen_flow_losses, # Return the list of epoch-wise losses
    }