# tests/test_train.py

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Import modules from your package
from Synthetic_Image_Generator.train import train_model
from Synthetic_Image_Generator.model import CNF_UNet # Need a model for training

# Suppress logging during tests to keep output clean
logging.getLogger().setLevel(logging.CRITICAL)

@pytest.fixture(scope="module")
def dummy_data_for_training():
    """
    Fixture to provide a small, simple dummy dataset for training tests.
    Generates dummy noise (z0) and target images (x1_real).
    """
    batch_size = 4
    image_size = (1, 8, 8) # Small image for quick tests
    num_samples = 10 * batch_size # Enough samples for a few batches

    z0_data = torch.randn(num_samples, *image_size)
    # Simple target: a slightly modified version of noise, or a constant image
    x1_real_data = torch.ones(num_samples, *image_size) * 0.5 + 0.1 * torch.randn(num_samples, *image_size)

    dataset = TensorDataset(z0_data, x1_real_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, image_size

def test_train_model_loss_decrease(dummy_data_for_training, device):
    """
    Test that the training loss decreases over a few epochs for a simple dummy dataset.
    This is a basic sanity check, not a guarantee of full convergence.
    """
    dataloader, _ = dummy_data_for_training
    
    # Initialize a fresh model and optimizer for this test
    generator = CNF_UNet(time_embed_dim=16).to(device) # Smaller embed_dim for speed
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-3)

    # Train for a small number of epochs
    epochs = 5
    training_losses = train_model(
        generator_model=generator,
        dataloader=dataloader,
        optimizer_gen=optimizer_gen,
        epochs=epochs,
        device=device
    )

    assert 'gen_flow_losses' in training_losses
    losses = training_losses['gen_flow_losses']
    assert len(losses) == epochs, "Should have a loss entry for each epoch."

    # Assert that the loss generally decreases (or at least doesn't increase drastically)
    # This is a weak check but prevents obvious training failures.
    # For very simple data, loss should decrease.
    if len(losses) > 1:
        # Check if the last loss is less than the first loss (general trend)
        assert losses[-1] < losses[0] * 1.1, "Loss should generally decrease or stay stable."
        # Check if loss is not NaN or Inf
        assert not any(torch.isnan(torch.tensor(l)) for l in losses)
        assert not any(torch.isinf(torch.tensor(l)) for l in losses)

def test_train_model_model_weights_change(dummy_data_for_training, device):
    """
    Test that the model's weights change after training, indicating that optimization occurred.
    """
    dataloader, _ = dummy_data_for_training
    
    generator = CNF_UNet(time_embed_dim=16).to(device)
    optimizer_gen = optim.Adam(generator.parameters(), lr=1e-3)

    # Capture initial weights
    initial_weights = {name: param.clone() for name, param in generator.named_parameters()}

    # Train for a few epochs
    train_model(
        generator_model=generator,
        dataloader=dataloader,
        optimizer_gen=optimizer_gen,
        epochs=3,
        device=device
    )

    # Check if any weights have changed
    weights_changed = False
    for name, param in generator.named_parameters():
        if not torch.equal(initial_weights[name], param):
            weights_changed = True
            break
    assert weights_changed, "Model weights should change after training."