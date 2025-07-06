# Synthetic Image Generator/train.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def train_model(generator_model, dataloader, optimizer_gen, epochs=100, device='cpu'):
    """
    Trains the generator model using the Flow Matching objective.

    Args:
        generator_model (torch.nn.Module): The generator (CNF_UNet) model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        optimizer_gen (torch.optim.Optimizer): Optimizer for the generator.
        epochs (int): Number of training epochs.
        device (str or torch.device): The device to run training on ('cuda' or 'cpu').

    Returns:
        dict: A dictionary containing training losses (e.g., 'gen_flow_losses').
    """
    generator_model.train() # Set model to training mode
    logger.info(f"Starting training for {epochs} epochs on device: {device}")

    gen_flow_losses = []

    for epoch in range(epochs):
        epoch_gen_flow_loss = 0.0
        # Use tqdm for a progress bar during the epoch
        for i, (z0, x1_real) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            z0, x1_real = z0.to(device), x1_real.to(device)
            batch_size = x1_real.size(0)

            # --- Train Generator (CNF_UNet) ---
            optimizer_gen.zero_grad() # Zero the gradients before backpropagation

            # CNF Flow Matching Loss (main objective)
            # Sample time 't' uniformly from [0, 1] for each sample in the batch
            t = torch.rand(batch_size, device=device)
            # Interpolate between initial noise (z0) and real data (x1_real) at time 't'
            # The .view(-1, 1, 1, 1) is to broadcast 't' across image dimensions
            xt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1_real
            # The target velocity vector is the direct difference between real data and noise
            v_target = (x1_real - z0)
            # The generator predicts the velocity field at xt and time t
            v_pred = generator_model(xt, t)
            # The loss is the Mean Squared Error between the predicted and target velocities
            loss_flow_matching = F.mse_loss(v_pred, v_target)

            # Combine losses (currently only flow matching loss)
            total_gen_loss = loss_flow_matching
            total_gen_loss.backward() # Perform backpropagation
            optimizer_gen.step() # Update model parameters

            # Store losses for reporting
            epoch_gen_flow_loss += loss_flow_matching.item()

        # Calculate average loss for the epoch
        avg_gen_flow_loss = epoch_gen_flow_loss / len(dataloader)
        gen_flow_losses.append(avg_gen_flow_loss)

        logger.info(f"Epoch {epoch+1}: G_Flow_Loss={avg_gen_flow_loss:.6f}")

    logger.info("Training complete.")
    return {
        'gen_flow_losses': gen_flow_losses,
    }