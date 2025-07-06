# tests/test_generate.py

import pytest
import torch
import torch.nn as nn
import logging
from typing import Union, Tuple, Any, Generator

# Import modules from your package
from Synthetic_Image_Generator.generate import generate_images
from Synthetic_Image_Generator.model import CNF_UNet # Need a model for generation

"""
Test suite for the image generation module.

This module contains unit tests for the `generate_images` function,
ensuring that the generation process produces outputs of the correct shape and type,
and that different initial noise inputs lead to different generated images.
"""

# Suppress logging during tests to keep output clean
logging.getLogger().setLevel(logging.CRITICAL)

@pytest.fixture(scope="module")
def device() -> torch.device:
    """
    Fixture to provide a PyTorch device (CUDA if available, else CPU).
    This ensures tests run on the appropriate hardware.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_generator_model(device: torch.device) -> CNF_UNet:
    """
    Fixture for a dummy CNF_UNet model (untrained).

    This provides a basic model instance for testing the `generate_images` function
    without requiring a fully trained model. The `time_embed_dim` is set smaller
    for faster test execution.

    Args:
        device (torch.device): The device on which the model should be placed.

    Returns:
        CNF_UNet: An untrained instance of the CNF_UNet model.
    """
    model: CNF_UNet = CNF_UNet(time_embed_dim=16).to(device) # Smaller for speed
    return model

def test_generate_images_output_shape_and_type(dummy_generator_model: CNF_UNet, device: torch.device) -> None:
    """
    Test that `generate_images` produces output with the correct shape and data type.

    Given an untrained generator model, initial noise of a specific shape, and a number of steps,
    When `generate_images` is called,
    Then the output should be a `torch.Tensor` with the same shape as the initial noise
    and `float32` data type.
    """
    batch_size: int = 2
    image_size: Tuple[int, int] = (1, 32, 32) # Small image size for test (C, H, W)
    initial_noise: torch.Tensor = torch.randn(batch_size, *image_size).to(device)
    steps: int = 10 # Small number of steps for quick test

    generated_images: torch.Tensor = generate_images(
        model=dummy_generator_model,
        initial_noise=initial_noise,
        steps=steps,
        device=device
    )

    assert isinstance(generated_images, torch.Tensor), "Generated images should be a torch.Tensor."
    assert generated_images.shape == initial_noise.shape, \
        f"Generated images shape should match initial noise shape {initial_noise.shape}, but got {generated_images.shape}."
    assert generated_images.dtype == torch.float32, f"Generated images dtype should be float32, but got {generated_images.dtype}."

def test_generate_images_output_range(dummy_generator_model: CNF_UNet, device: torch.device) -> None:
    """
    Test that generated image pixel values are within a reasonable range.

    Given an untrained generator model and initial noise,
    When `generate_images` is called,
    Then the pixel values in the generated images should not be extremely large or small,
    indicating a plausible output range for normalized images, even without a final activation.
    """
    batch_size: int = 1
    image_size: Tuple[int, int] = (1, 16, 16)
    initial_noise: torch.Tensor = torch.randn(batch_size, *image_size).to(device)
    steps: int = 10

    generated_images: torch.Tensor = generate_images(
        model=dummy_generator_model,
        initial_noise=initial_noise,
        steps=steps,
        device=device
    )

    # The CNF_UNet's final layer is a Conv2d without an explicit activation (like tanh or sigmoid)
    # to strictly constrain output to [-1, 1]. However, the integration process should keep
    # values within a reasonable bound. A broad check for non-extreme values is appropriate here.
    assert generated_images.min() > -10.0, f"Generated image min pixel value {generated_images.min().item()} is too low."
    assert generated_images.max() < 10.0, f"Generated image max pixel value {generated_images.max().item()} is too high."

def test_generate_images_different_noise_different_output(dummy_generator_model: CNF_UNet, device: torch.device) -> None:
    """
    Test that different initial noise inputs lead to different generated outputs.

    Given an untrained generator model and two distinct initial noise tensors,
    When `generate_images` is called for each noise tensor,
    Then the resulting generated images should be perceptibly different,
    assuming the model is not degenerate.
    """
    batch_size: int = 1
    image_size: Tuple[int, int] = (1, 16, 16)
    steps: int = 10

    noise1: torch.Tensor = torch.randn(batch_size, *image_size).to(device)
    noise2: torch.randn = torch.randn(batch_size, *image_size).to(device)
    
    # Ensure noise inputs are actually different to make the test meaningful.
    # Use `not torch.equal` for exact tensor equality.
    assert not torch.equal(noise1, noise2), "Initial noise tensors should be different for this test."

    generated_images1: torch.Tensor = generate_images(
        model=dummy_generator_model,
        initial_noise=noise1,
        steps=steps,
        device=device
    )
    generated_images2: torch.Tensor = generate_images(
        model=dummy_generator_model,
        initial_noise=noise2,
        steps=steps,
        device=device
    )

    # Outputs should be different given different noise inputs.
    # Use `torch.allclose` with a tolerance for floating-point comparisons.
    assert not torch.allclose(generated_images1, generated_images2, atol=1e-5), \
        "Different noise inputs should lead to different generated images, but they were too similar."
