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
    """Automatically mock matplotlib.pyplot.show for all tests in this file."""
    mocker.patch('matplotlib.pyplot.show')

def test_plot_pixel_distributions():
    """Test pixel distribution plotting function."""
    real_batch = torch.randn(16, 1, 32, 32)
    noise_batch = torch.randn(16, 1, 32, 32)
    gen_flat = torch.randn(16 * 32 * 32)
    
    # Test without generated images
    plot_pixel_distributions(real_batch, noise_batch)
    
    # Test with generated images
    plot_pixel_distributions(real_batch, noise_batch, generated_images_flat=gen_flat)

def test_plot_sample_images():
    """Test sample image plotting function."""
    real_batch = torch.randn(4, 1, 32, 32)
    noise_batch = torch.randn(4, 1, 32, 32)
    plot_sample_images(real_batch, noise_batch)

def test_plot_generated_samples():
    """Test generated samples plotting function."""
    gen_images = torch.randn(16, 1, 32, 32)
    plot_generated_samples(gen_images)

def test_plot_real_vs_generated_side_by_side():
    """Test side-by-side comparison plotting function."""
    real_images = torch.randn(4, 1, 32, 32)
    gen_images = torch.randn(4, 1, 32, 32)
    plot_real_vs_generated_side_by_side(real_images, gen_images)

    # Test with empty tensor (should not fail)
    plot_real_vs_generated_side_by_side(torch.tensor([]), gen_images)