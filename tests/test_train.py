# tests/test_train.py

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Tuple, Dict, Any, Generator, Union

# Import modules from your package
from Synthetic_Image_Generator.train import train_model
from Synthetic_Image_Generator.model import CNF_UNet # Need a model for training

"""
Test suite for the training module.

This module contains unit tests for the `train_model` function,
verifying that the training loop correctly updates model weights,
that the loss generally decreases, and that no NaN/Inf values are produced.
A small, simple dummy dataset is used for efficient testing.
"""

# Suppress logging during tests to keep output clean, allowing specific logs via caplog.
logging.getLogger().setLevel(logging.CRITICAL)

@pytest.fixture(scope="module")
def device() -> torch.device:
    """Fixture to provide a PyTorch device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def dummy_data_for_training() -> Generator[Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, ...]], None, None]:
    """
    Fixture to provide a small, simple dummy dataset for training tests.

    Generates dummy noise (`z0`) and target images (`x1_real`) as `torch.Tensor`s,
    then wraps them in a `TensorDataset` and `DataLoader`. This allows for
    quick and predictable training behavior for testing purposes.

    Yields:
        Tuple[DataLoader, Tuple[int, ...]]: A tuple containing:
            - dataloader: A PyTorch DataLoader with dummy data.
            - image_size: The (C, H, W) tuple representing the size of images.
    """
    batch_size: int = 4
    image_size: Tuple[int, int] = (1, 8, 8) # Small image for quick tests (C, H, W)
    num_samples: int = 10 * batch_size # Enough samples for a few batches

    # Generate random noise data.
    z0_data: torch.Tensor = torch.randn(num_samples, *image_size)
    # Generate simple target data: a slightly modified version of a constant image
    # with some noise, to make it a learnable target.
    x1_real_data: torch.Tensor = torch.ones(num_samples, *image_size) * 0.5 + 0.1 * torch.randn(num_samples, *image_size)

    dataset: TensorDataset = TensorDataset(z0_data, x1_real_data)
    dataloader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    yield dataloader, image_size

def test_train_model_loss_decrease(
    dummy_data_for_training: Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, ...]],
    device: torch.device
) -> None:
    """
    Test that the training loss generally decreases over a few epochs for a simple dummy dataset.

    Given a dummy dataset, an untrained generator model, and an optimizer,
    When `train_model` is called for a small number of epochs,
    Then the final average loss should be less than the initial average loss,
    and no NaN or Inf values should be present in the loss history.
    """
    dataloader, _ = dummy_data_for_training
    
    # Initialize a fresh model and optimizer for this test to ensure isolation.
    generator: CNF_UNet = CNF_UNet(time_embed_dim=16).to(device) # Smaller embed_dim for speed
    optimizer_gen: optim.Adam = optim.Adam(generator.parameters(), lr=1e-3)

    # Train for a small number of epochs to observe a trend.
    epochs: int = 5
    training_losses: Dict[str, Any] = train_model(
        generator_model=generator,
        dataloader=dataloader,
        optimizer_gen=optimizer_gen,
        epochs=epochs,
        device=device
    )

    assert 'gen_flow_losses' in training_losses, "Training losses dictionary should contain 'gen_flow_losses'."
    losses: list[float] = training_losses['gen_flow_losses']
    assert len(losses) == epochs, f"Should have a loss entry for each epoch ({epochs} epochs), but got {len(losses)}."

    # Assert that the loss generally decreases (or at least doesn't increase drastically).
    # This is a sanity check for basic training functionality.
    if len(losses) > 1:
        assert losses[-1] < losses[0] * 1.1, \
            f"Loss should generally decrease or stay stable. Final loss {losses[-1]:.6f} was not less than 110% of initial loss {losses[0]:.6f}."
        
        # Check if loss is not NaN or Inf, which indicates training instability.
        for i, l in enumerate(losses):
            assert not torch.isnan(torch.tensor(l)), f"Loss at epoch {i+1} is NaN."
            assert not torch.isinf(torch.tensor(l)), f"Loss at epoch {i+1} is Inf."

def test_train_model_model_weights_change(
    dummy_data_for_training: Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], Tuple[int, ...]],
    device: torch.device
) -> None:
    """
    Test that the model's weights change after training, indicating that optimization occurred.

    Given a dummy dataset, an untrained generator model, and an optimizer,
    When `train_model` is called for a few epochs,
    Then at least one model parameter's weight should have changed from its initial value.
    """
    dataloader, _ = dummy_data_for_training
    
    generator: CNF_UNet = CNF_UNet(time_embed_dim=16).to(device)
    optimizer_gen: optim.Adam = optim.Adam(generator.parameters(), lr=1e-3)

    # Capture initial weights of all parameters for comparison.
    initial_weights: Dict[str, torch.Tensor] = {name: param.clone() for name, param in generator.named_parameters()}

    # Train for a few epochs to allow weights to update.
    train_model(
        generator_model=generator,
        dataloader=dataloader,
        optimizer_gen=optimizer_gen,
        epochs=3,
        device=device
    )

    # Check if any weights have changed.
    weights_changed: bool = False
    for name, param in generator.named_parameters():
        if not torch.equal(initial_weights[name], param):
            weights_changed = True
            break
    
    assert weights_changed, "Model weights should change after training, indicating successful optimization."
