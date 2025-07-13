"""
Visualization utilities for image generation models.

This module provides functions to visualize various aspects of the image
generation process, including pixel distributions of real, noise, and
generated images, as well as sample image grids. These visualizations
are crucial for understanding the model's performance and the
characteristics of the data it processes and generates.

Functions
---------
plot_pixel_distributions(sample_real_batch, sample_noise_batch, generated_images_flat=None)
    Plots histograms of pixel value distributions for real, initial noise,
    and optionally generated images.

plot_sample_images(sample_real_batch, sample_noise_batch)
    Displays a small grid of real CT images and their corresponding initial
    Gaussian noise inputs.

plot_generated_samples(generated_images, num_display=16)
    Displays a grid of generated images.

plot_real_vs_generated_side_by_side(real_images_batch_tensor, generated_images, num_side_by_side=4)
    Compares real and generated images side-by-side in a grid.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pixel_distributions(
    sample_real_batch: torch.Tensor,
    sample_noise_batch: torch.Tensor,
    generated_images_flat: np.ndarray = None
) -> None:
    """
    Plots histograms of pixel value distributions for real, initial noise,
    and optionally generated images.

    This function creates two sets of plots:
    1. Histograms of real image pixel values and initial Gaussian noise pixel values.
    2. (Optional) A comparative histogram showing real image pixel distribution
       against generated image pixel distribution.

    Parameters
    ----------
    sample_real_batch : torch.Tensor
        A batch of real images (e.g., from the dataset), expected to be
        denormalized or in the range relevant for visualization (e.g., [-1, 1]).
        Shape: (N, C, H, W).
    sample_noise_batch : torch.Tensor
        A batch of initial Gaussian noise images corresponding to the real images,
        used as input to the generator. Shape: (N, C, H, W).
    generated_images_flat : numpy.ndarray, optional
        A 1D numpy array of flattened pixel values from generated images.
        If provided, a third plot comparing real vs. generated distributions is shown.
        Defaults to None.
    """
    plt.figure(figsize=(12, 5))
    
    # Flatten tensors to 1D numpy arrays for histogram plotting
    flat_real = sample_real_batch.view(-1).cpu().numpy()
    flat_noise = sample_noise_batch.view(-1).cpu().numpy()

    # Plot 1: Real CT Image Pixel Distribution
    plt.subplot(1, 2, 1)
    plt.hist(flat_real, bins=50, color='blue', alpha=0.7)
    plt.title('Real CT Image Pixel Distribution (Sample)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    
    # Plot 2: Initial Gaussian Noise Distribution
    plt.subplot(1, 2, 2)
    plt.hist(flat_noise, bins=50, color='red', alpha=0.7)
    plt.title('Initial Gaussian Noise Distribution (Sample)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # Plot 3 (Optional): Comparison of Real vs. Generated Pixel Distributions
    if generated_images_flat is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(flat_real, bins=50, color='blue', alpha=0.6, label='Real CT Image Pixel Distribution (Sampled)')
        plt.hist(generated_images_flat, bins=50, color='green', alpha=0.6, label='Generated Image Pixel Distribution')
        plt.title('Comparison of Pixel Distributions: Real vs. Generated')
        plt.xlabel('Pixel Value (Normalized [-1, 1])')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.show()


def plot_sample_images(sample_real_batch: torch.Tensor, sample_noise_batch: torch.Tensor) -> None:
    """
    Displays a small grid of real CT images and their corresponding initial
    Gaussian noise inputs.

    Parameters
    ----------
    sample_real_batch : torch.Tensor
        A batch of real images, typically from the DataLoader.
        Expected shape: (N, C, H, W).
    sample_noise_batch : torch.Tensor
        A batch of corresponding initial noise images.
        Expected shape: (N, C, H, W).
    """
    plt.figure(figsize=(10, 4))
    num_display = min(4, sample_real_batch.shape[0]) # Display up to 4 pairs
    for i in range(num_display):
        # Display real image
        plt.subplot(2, num_display, i + 1)
        plt.imshow(sample_real_batch[i, 0].cpu().numpy(), cmap='gray') # Assuming grayscale, channel 0
        plt.title("Real CT Image")
        plt.axis('off')
        
        # Display corresponding initial noise
        plt.subplot(2, num_display, i + num_display + 1)
        plt.imshow(sample_noise_batch[i, 0].cpu().numpy(), cmap='gray') # Assuming grayscale, channel 0
        plt.title("Initial Noise")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_generated_samples(generated_images: torch.Tensor, num_display: int = 16) -> None:
    """
    Displays a grid of generated images.

    Parameters
    ----------
    generated_images : torch.Tensor
        A batch of images generated by the model.
        Expected shape: (N, C, H, W).
    num_display : int, optional
        The maximum number of generated images to display in the grid.
        Defaults to 16.
    """
    plt.figure(figsize=(10, 8))
    # Ensure num_display does not exceed the number of available generated images
    num_display = min(num_display, generated_images.shape[0])
    
    # Calculate grid dimensions (e.g., 4x4 for 16 images)
    rows = int(np.ceil(np.sqrt(num_display)))
    cols = int(np.ceil(num_display / rows))

    for i in range(num_display):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray') # Assuming grayscale
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_real_vs_generated_side_by_side(
    real_images_batch_tensor: torch.Tensor,
    generated_images: torch.Tensor,
    num_side_by_side: int = 4
) -> None:
    """
    Compares real and generated images side-by-side in a grid.

    This function displays pairs of real and generated images to allow for a
    direct visual comparison of quality and fidelity.

    Parameters
    ----------
    real_images_batch_tensor : torch.Tensor
        A batch of real images to be displayed. Expected shape: (N, C, H, W).
    generated_images : torch.Tensor
        A batch of generated images to be displayed. Expected shape: (N, C, H, W).
    num_side_by_side : int, optional
        The number of real/generated pairs to display. Defaults to 4.
    """
    # Only proceed if there are actual images to display
    if real_images_batch_tensor.numel() > 0 and generated_images.numel() > 0:
        plt.figure(figsize=(12, 6))
        # Determine the minimum number of images to compare, limited by desired count
        num_compare = min(num_side_by_side, real_images_batch_tensor.shape[0], generated_images.shape[0])
        
        for i in range(num_compare):
            # Display real image
            plt.subplot(2, num_side_by_side, i + 1)
            plt.imshow(real_images_batch_tensor[i, 0].cpu().numpy(), cmap='gray') # Assuming grayscale
            plt.title(f"Real {i+1}")
            plt.axis('off')

            # Display corresponding generated image
            plt.subplot(2, num_side_by_side, i + num_side_by_side + 1)
            plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray') # Assuming grayscale
            plt.title(f"Generated {i+1}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle("Real vs. Generated Images (Side-by-Side)", y=1.02) # Add a super title
        plt.show()
    else:
        print("No images available for side-by-side comparison.")