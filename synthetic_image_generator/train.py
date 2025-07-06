# Synthetic Image Generator/train.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Any, Union

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module contains the training loop for the Conditional Normalizing Flow (CNF)
model. It implements the Flow Matching objective to train the generator to predict
the velocity field that transforms Gaussian noise into target images.
"""

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
    the difference vector.

    Args:
        generator_model (torch.nn.Module): The generator model (an instance of CNF_UNet)
                                           to be trained.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of
                                                  (noise, real_image) pairs.
        optimizer_gen (torch.optim.Optimizer): The optimizer configured for the
                                               generator_model's parameters.
        epochs (int): The total number of training epochs to run.
        device (Union[str, torch.device]): The device ('cuda' or 'cpu') on which
                                           to perform training.

    Returns:
        Dict[str, Any]: A dictionary containing training statistics, specifically:
                        - 'gen_flow_losses' (List[float]): A list of average Flow Matching
                                                         losses for the generator, per epoch.
    """
    generator_model.train() # Set the model to training mode
    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    gen_flow_losses: list[float] = [] # List to store average generator flow losses per epoch

    for epoch in range(epochs):
        epoch_gen_flow_loss: float = 0.0
        # Use tqdm for a progress bar during the epoch to visualize training progress.
        for i, (z0, x1_real) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move data to the specified device
            z0, x1_real = z0.to(device), x1_real.to(device)
            batch_size: int = x1_real.size(0)

            # --- Train Generator (CNF_UNet) ---
            optimizer_gen.zero_grad() # Clear gradients from the previous iteration

            # CNF Flow Matching Loss Calculation:
            # 1. Sample time 't' uniformly from [0, 1] for each sample in the batch.
            #    This ensures the model learns across the entire continuous flow path.
            t: torch.Tensor = torch.rand(batch_size, device=device)

            # 2. Interpolate between initial noise (z0) and real data (x1_real) at time 't'.
            #    xt represents an intermediate state along the flow from noise to data.
            #    .view(-1, 1, 1, 1) broadcasts 't' across image dimensions for element-wise multiplication.
            xt: torch.Tensor = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1_real

            # 3. Define the target velocity vector (v_target).
            #    For linear interpolation, the true velocity is simply the difference between
            #    the end point (real data) and the start point (noise).
            v_target: torch.Tensor = (x1_real - z0)

            # 4. The generator predicts the velocity field (v_pred) at the interpolated state xt and time t.
            v_pred: torch.Tensor = generator_model(xt, t)

            # 5. Calculate the loss as the Mean Squared Error between the predicted and target velocities.
            loss_flow_matching: torch.Tensor = F.mse_loss(v_pred, v_target)

            # Combine losses (currently, only flow matching loss is used for the generator).
            total_gen_loss: torch.Tensor = loss_flow_matching
            total_gen_loss.backward() # Perform backpropagation to compute gradients
            optimizer_gen.step() # Update model parameters based on computed gradients

            # Accumulate loss for reporting the average loss of the current epoch.
            epoch_gen_flow_loss += loss_flow_matching.item()

        # Calculate the average loss for the current epoch.
        avg_gen_flow_loss: float = epoch_gen_flow_loss / len(dataloader)
        gen_flow_losses.append(avg_gen_flow_loss)

        # Log the average loss for the epoch.
        logger.info(f"Epoch {epoch+1}: G_Flow_Loss={avg_gen_flow_loss:.6f}")

    logger.info("Training complete.")
    return {
        'gen_flow_losses': gen_flow_losses,
    }
