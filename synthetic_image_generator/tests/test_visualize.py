# tests/test_visualize.py
import torch
import pytest
from unittest.mock import patch

# Adjust import path
from visualize import (
    plot_pixel_distributions, plot_sample_images, 
    plot_generated_samples, plot_real_vs_generated_side_by_side
)

@pytest.fixture(autouse=True)
def mock_matplotlib(mocker):
    """
    GIVEN: a test context
    WHEN: a test in this file is about to run
    THEN: `matplotlib.pyplot.show` is automatically mocked to prevent plots from
          appearing during test execution, ensuring tests run non-interactively
    """
    mocker.patch('matplotlib.pyplot.show')

def test_plot_pixel_distributions():
    """
    GIVEN: batches of real images, noise, and optionally flattened generated images
    WHEN: the `plot_pixel_distributions` function is called
    THEN: the function should execute without raising errors,
          indicating that the plotting logic for pixel distributions is sound
    """
    real_batch = torch.randn(16, 1, 32, 32)
    noise_batch = torch.randn(16, 1, 32, 32)
    gen_flat = torch.randn(16 * 32 * 32)
    
    # Test without generated images
    plot_pixel_distributions(real_batch, noise_batch)
    
    # Test with generated images
    plot_pixel_distributions(real_batch, noise_batch, generated_images_flat=gen_flat)

def test_plot_sample_images():
    """
    GIVEN: batches of real images and noise
    WHEN: the `plot_sample_images` function is called
    THEN: the function should execute without raising errors,
          indicating that the plotting logic for sample images is sound
    """
    real_batch = torch.randn(4, 1, 32, 32)
    noise_batch = torch.randn(4, 1, 32, 32)
    plot_sample_images(real_batch, noise_batch)

def test_plot_generated_samples():
    """
    GIVEN: a batch of generated images
    WHEN: the `plot_generated_samples` function is called
    THEN: the function should execute without raising errors,
          indicating that the plotting logic for generated samples is sound
    """
    gen_images = torch.randn(16, 1, 32, 32)
    plot_generated_samples(gen_images)

def test_plot_real_vs_generated_side_by_side():
    """
    GIVEN: batches of real and generated images (which may be empty for the real images)
    WHEN: the `plot_real_vs_generated_side_by_side` function is called
    THEN: the function should execute without raising errors,
          indicating that the plotting logic for side-by-side comparison is sound
    """
    real_images = torch.randn(4, 1, 32, 32)
    gen_images = torch.randn(4, 1, 32, 32)
    plot_real_vs_generated_side_by_side(real_images, gen_images)

    # Test with empty tensor (should not fail)
    plot_real_vs_generated_side_by_side(torch.tensor([]), gen_images)