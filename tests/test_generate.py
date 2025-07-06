# tests/test_generate.py

import pytest
import torch
import torch.nn as nn
import logging

# Import modules from your package
from Synthetic_Image_Generator.generate import generate_images
from Synthetic_Image_Generator.model import CNF_UNet # Need a model for generation

# Suppress logging during tests
logging.getLogger().setLevel(logging.CRITICAL)

@pytest.fixture(scope="module")
def device():
    """Fixture to provide a device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_generator_model(device):
    """Fixture for a dummy CNF_UNet model (untrained)."""
    model = CNF_UNet(time_embed_dim=16).to(device) # Smaller for speed
    return model

def test_generate_images_output_shape_and_type(dummy_generator_model, device):
    """
    Test that generate_images produces output with the correct shape and data type.
    """
    batch_size = 2
    image_size = (1, 32, 32) # Small image size for test
    initial_noise = torch.randn(batch_size, *image_size).to(device)
    steps = 10 # Small number of steps for quick test

    generated_images = generate_images(
        model=dummy_generator_model,
        initial_noise=initial_noise,
        steps=steps,
        device=device
    )

    assert isinstance(generated_images, torch.Tensor)
    assert generated_images.shape == initial_noise.shape, \
        "Generated images shape should match initial noise shape."
    assert generated_images.dtype == torch.float32, "Generated images dtype should be float32."

def test_generate_images_output_range(dummy_generator_model, device):
    """
    Test that generated image pixel values are within a reasonable range (e.g., [-1, 1]).
    Since the last layer of CNF_UNet is a Conv2d without activation, the range isn't strictly [-1,1],
    but it should be within a plausible range for normalized images.
    """
    batch_size = 1
    image_size = (1, 16, 16)
    initial_noise = torch.randn(batch_size, *image_size).to(device)
    steps = 10

    generated_images = generate_images(
        model=dummy_generator_model,
        initial_noise=initial_noise,
        steps=steps,
        device=device
    )

    # Given the model's output is a velocity and it's integrated,
    # the final range might not be strictly [-1, 1] without a final activation.
    # However, it should not be extremely large or small.
    # We can check if the values are within a typical float range,
    # or a more specific range if the model's last layer implies it.
    # For now, a broad check for non-extreme values.
    assert generated_images.min() > -10.0 and generated_images.max() < 10.0, \
        "Generated image pixel values should be within a reasonable range."
    # A more specific test would involve training the model to converge to [-1,1] range.

def test_generate_images_different_noise_different_output(dummy_generator_model, device):
    """
    Test that different initial noise inputs lead to different generated outputs.
    This checks basic functionality, assuming the model is not completely degenerate.
    """
    batch_size = 1
    image_size = (1, 16, 16)
    steps = 10

    noise1 = torch.randn(batch_size, *image_size).to(device)
    noise2 = torch.randn(batch_size, *image_size).to(device)
    
    # Ensure noise inputs are actually different
    assert not torch.equal(noise1, noise2)

    generated_images1 = generate_images(
        model=dummy_generator_model,
        initial_noise=noise1,
        steps=steps,
        device=device
    )
    generated_images2 = generate_images(
        model=dummy_generator_model,
        initial_noise=noise2,
        steps=steps,
        device=device
    )

    # Outputs should be different given different noise inputs
    assert not torch.allclose(generated_images1, generated_images2, atol=1e-5), \
        "Different noise inputs should lead to different generated images."