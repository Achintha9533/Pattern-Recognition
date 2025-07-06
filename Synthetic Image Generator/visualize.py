# Synthetic Image Generator/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from typing import Dict, Any

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module provides various visualization utilities for the Synthetic Image Generator project.
It includes functions to plot pixel distributions, display sample images (real, noise, generated),
visualize training loss curves, and compare real vs. generated images side-by-side.

As per the project guidelines, visualization functions do not perform data preprocessing
and are generally not unit-tested, focusing solely on graphical representation.
"""

def plot_pixel_distributions(sample_image_batch: torch.Tensor, sample_noise_batch: torch.Tensor) -> None:
    """
    Plots the pixel distribution histograms for a batch of real CT images and
    a batch of initial Gaussian noise samples. This helps to visualize the
    input distributions to the generative model.

    Args:
        sample_image_batch (torch.Tensor): A batch of real CT image tensors (CPU).
                                           Expected pixel range: [-1, 1].
        sample_noise_batch (torch.Tensor): A batch of initial Gaussian noise tensors (CPU).
                                           Expected pixel range: typically centered around 0.
    """
    logger.info("Plotting initial data and noise pixel distributions.")
    # Flatten the tensors to get all pixel values into a 1D array for histogram plotting.
    flat_noise: np.ndarray = sample_noise_batch.view(-1).numpy()
    flat_image: np.ndarray = sample_image_batch.view(-1).numpy()

    plt.figure(figsize=(12, 5))

    # Plot histogram for real CT images
    plt.subplot(1, 2, 1)
    plt.hist(flat_image, bins=50, color='blue', alpha=0.7)
    plt.title('Real CT Image Pixel Distribution (Sample)')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')

    # Plot histogram for initial Gaussian noise
    plt.subplot(1, 2, 2)
    plt.hist(flat_noise, bins=50, color='red', alpha=0.7)
    plt.title('Initial Gaussian Noise Distribution (Sample)')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')

    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()
    logger.debug("Initial pixel distribution plots displayed.")

def plot_sample_images_and_noise(sample_image_batch: torch.Tensor, sample_noise_batch: torch.Tensor, num_display: int = 4) -> None:
    """
    Displays a grid of sample real CT images and their corresponding initial Gaussian noise
    samples side-by-side. This provides a visual comparison of the source and target
    distributions at the beginning of the generative process.

    Args:
        sample_image_batch (torch.Tensor): A batch of real CT image tensors (CPU).
        sample_noise_batch (torch.Tensor): A batch of initial Gaussian noise tensors (CPU).
        num_display (int): The number of image pairs (real image + noise) to display.
                           Displays up to the minimum of available samples or `num_display`.
    """
    logger.info(f"Displaying {num_display} sample real images and initial noise.")
    plt.figure(figsize=(10, 4))
    # Iterate up to `num_display` or the actual batch size, whichever is smaller.
    for i in range(min(num_display, sample_image_batch.shape[0])):
        # Display real CT image
        plt.subplot(2, num_display, i + 1)
        plt.imshow(sample_image_batch[i, 0].cpu().numpy(), cmap='gray') # .cpu().numpy() for plotting
        plt.title("Real CT Image")
        plt.axis('off') # Hide axes for cleaner image display

        # Display corresponding initial noise
        plt.subplot(2, num_display, i + num_display + 1)
        plt.imshow(sample_noise_batch[i, 0].cpu().numpy(), cmap='gray')
        plt.title("Initial Noise")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    logger.debug("Sample real images and noise plots displayed.")

def plot_training_losses(training_losses: Dict[str, Any]) -> None:
    """
    Plots the generator's training loss over epochs. This visualization helps
    to monitor the training progress and identify potential issues like
    non-convergence or oscillations.

    Args:
        training_losses (Dict[str, Any]): A dictionary containing training loss lists.
                                          Expected to have a key 'gen_flow_losses'
                                          with a list of float values.
    """
    logger.info("Plotting training losses.")
    plt.figure(figsize=(8, 6))
    plt.plot(training_losses['gen_flow_losses'], label='Generator Flow Matching Loss', color='blue')
    plt.title('Generator Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True) # Add a grid for easier readability
    plt.tight_layout()
    plt.show()
    logger.debug("Training loss plot displayed.")

def plot_generated_pixel_distribution_comparison(all_real_pixels_flat: np.ndarray, flat_generated_images: np.ndarray) -> None:
    """
    Compares the pixel distributions of real CT images and generated images
    using histograms. This helps to assess how well the generated images
    match the statistical properties of the real data.

    Args:
        all_real_pixels_flat (np.ndarray): Flattened array of pixel values from a sample of real images.
                                           Expected pixel range: [-1, 1].
        flat_generated_images (np.ndarray): Flattened array of pixel values from all generated images.
                                            Expected pixel range: [-1, 1].
    """
    logger.info("Plotting comparison of pixel distributions: Real vs. Generated.")
    plt.figure(figsize=(10, 6))
    plt.hist(all_real_pixels_flat, bins=50, color='blue', alpha=0.6, label='Real CT Image Pixel Distribution (Sampled)')
    plt.hist(flat_generated_images, bins=50, color='green', alpha=0.6, label='Generated Image Pixel Distribution')
    plt.title('Comparison of Pixel Distributions: Real vs. Generated')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.75) # Add horizontal grid lines
    plt.show()
    logger.debug("Pixel distribution comparison plot displayed.")

def plot_sample_generated_images(generated_images: torch.Tensor, num_display: int = 16) -> None:
    """
    Displays a grid of sample generated images. This provides a quick visual
    assessment of the quality and diversity of the synthetic outputs.

    Args:
        generated_images (torch.Tensor): A batch of generated image tensors (CPU).
                                         Expected pixel range: [-1, 1].
        num_display (int): The number of generated images to display in the grid.
                           Displays up to the minimum of available samples or `num_display`.
    """
    logger.info(f"Displaying {num_display} sample generated images.")
    plt.figure(figsize=(10, 8))
    # Determine the number of images to display, up to 16, or fewer if not enough samples.
    num_effective_display: int = min(num_display, generated_images.shape[0])
    # Calculate grid dimensions (e.g., 4x4 for 16 images)
    rows: int = int(np.ceil(np.sqrt(num_effective_display)))
    cols: int = int(np.ceil(num_effective_display / rows))

    for i in range(num_effective_display):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    logger.debug("Sample generated images plot displayed.")

def plot_real_vs_generated_side_by_side(real_images_batch_tensor: torch.Tensor, generated_images: torch.Tensor, num_side_by_side: int = 4) -> None:
    """
    Displays pairs of real and generated images side-by-side for direct visual comparison.
    This helps in qualitatively assessing the realism and similarity of generated samples.

    Args:
        real_images_batch_tensor (torch.Tensor): A batch of real image tensors (CPU).
                                                 Expected pixel range: [-1, 1].
        generated_images (torch.Tensor): A batch of generated image tensors (CPU).
                                         Expected pixel range: [-1, 1].
        num_side_by_side (int): Number of real/generated pairs to display.
                                Displays up to the minimum of available samples or `num_side_by_side`.
    """
    logger.info(f"Displaying {num_side_by_side} pairs of real vs. generated images side-by-side.")
    # Check if there are enough real images to perform the comparison.
    if real_images_batch_tensor.numel() == 0:
        logger.warning("No real images provided for side-by-side comparison. Skipping plot.")
        return
    
    # Ensure we display up to the minimum of available samples or `num_side_by_side`.
    num_effective_pairs: int = min(num_side_by_side, real_images_batch_tensor.shape[0], generated_images.shape[0])
    if num_effective_pairs == 0:
        logger.warning("Not enough real or generated images to create side-by-side comparison. Skipping plot.")
        return

    plt.figure(figsize=(12, 6))
    for i in range(num_effective_pairs):
        # Display Real Image
        plt.subplot(2, num_effective_pairs, i + 1)
        plt.imshow(real_images_batch_tensor[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Real {i+1}")
        plt.axis('off')

        # Display Generated Image
        plt.subplot(2, num_effective_pairs, i + num_effective_pairs + 1)
        plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    # Add a main title for the entire figure, positioned slightly above the subplots.
    plt.suptitle("Real vs. Generated Images (Side-by-Side)", y=1.02)
    plt.show()
    logger.debug("Real vs. Generated side-by-side plots displayed.")
