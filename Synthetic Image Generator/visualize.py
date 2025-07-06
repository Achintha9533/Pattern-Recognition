# Synthetic Image Generator/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def plot_pixel_distributions(sample_image_batch, sample_noise_batch):
    """
    Plots the pixel distribution histograms for real CT images and initial Gaussian noise.

    Args:
        sample_image_batch (torch.Tensor): A batch of real CT image tensors (CPU).
        sample_noise_batch (torch.Tensor): A batch of initial Gaussian noise tensors (CPU).
    """
    logger.info("Plotting initial data and noise pixel distributions.")
    flat_noise = sample_noise_batch.view(-1).numpy()
    flat_image = sample_image_batch.view(-1).numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(flat_image, bins=50, color='blue', alpha=0.7)
    plt.title('Real CT Image Pixel Distribution (Sample)')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')
    plt.subplot(1, 2, 2)
    plt.hist(flat_noise, bins=50, color='red', alpha=0.7)
    plt.title('Initial Gaussian Noise Distribution (Sample)')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    logger.debug("Initial pixel distribution plots displayed.")

def plot_sample_images_and_noise(sample_image_batch, sample_noise_batch, num_display=4):
    """
    Displays sample real CT images and their corresponding initial noise.

    Args:
        sample_image_batch (torch.Tensor): A batch of real CT image tensors (CPU).
        sample_noise_batch (torch.Tensor): A batch of initial Gaussian noise tensors (CPU).
        num_display (int): Number of image pairs to display.
    """
    logger.info(f"Displaying {num_display} sample real images and initial noise.")
    plt.figure(figsize=(10, 4))
    for i in range(min(num_display, sample_image_batch.shape[0])):
        plt.subplot(2, num_display, i + 1)
        plt.imshow(sample_image_batch[i, 0], cmap='gray')
        plt.title("Real CT Image")
        plt.axis('off')
        plt.subplot(2, num_display, i + num_display + 1)
        plt.imshow(sample_noise_batch[i, 0], cmap='gray')
        plt.title("Initial Noise")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    logger.debug("Sample real images and noise plots displayed.")

def plot_training_losses(training_losses):
    """
    Plots the generator's training loss over epochs.

    Args:
        training_losses (dict): Dictionary containing training loss lists (e.g., 'gen_flow_losses').
    """
    logger.info("Plotting training losses.")
    plt.figure(figsize=(8, 6))
    plt.plot(training_losses['gen_flow_losses'], label='Generator Flow Matching Loss', color='blue')
    plt.title('Generator Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    logger.debug("Training loss plot displayed.")

def plot_generated_pixel_distribution_comparison(all_real_pixels_flat, flat_generated_images):
    """
    Compares the pixel distributions of real and generated images.

    Args:
        all_real_pixels_flat (np.ndarray): Flattened array of pixel values from real images.
        flat_generated_images (np.ndarray): Flattened array of pixel values from generated images.
    """
    logger.info("Plotting comparison of pixel distributions: Real vs. Generated.")
    plt.figure(figsize=(10, 6))
    plt.hist(all_real_pixels_flat, bins=50, color='blue', alpha=0.6, label='Real CT Image Pixel Distribution (Sampled)')
    plt.hist(flat_generated_images, bins=50, color='green', alpha=0.6, label='Generated Image Pixel Distribution')
    plt.title('Comparison of Pixel Distributions: Real vs. Generated')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()
    logger.debug("Pixel distribution comparison plot displayed.")

def plot_sample_generated_images(generated_images, num_display=16):
    """
    Displays a grid of sample generated images.

    Args:
        generated_images (torch.Tensor): A batch of generated image tensors (CPU).
        num_display (int): Number of generated images to display.
    """
    logger.info(f"Displaying {num_display} sample generated images.")
    plt.figure(figsize=(10, 8))
    for i in range(min(num_display, generated_images.shape[0])):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, 0], cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    logger.debug("Sample generated images plot displayed.")

def plot_real_vs_generated_side_by_side(real_images_batch_tensor, generated_images, num_side_by_side=4):
    """
    Displays real and generated images side-by-side for direct comparison.

    Args:
        real_images_batch_tensor (torch.Tensor): A batch of real image tensors (CPU).
        generated_images (torch.Tensor): A batch of generated image tensors (CPU).
        num_side_by_side (int): Number of real/generated pairs to display.
    """
    logger.info(f"Displaying {num_side_by_side} pairs of real vs. generated images side-by-side.")
    if real_images_batch_tensor.numel() == 0:
        logger.warning("No real images provided for side-by-side comparison.")
        return

    plt.figure(figsize=(12, 6))
    for i in range(min(num_side_by_side, real_images_batch_tensor.shape[0])):
        # Real Image
        plt.subplot(2, num_side_by_side, i + 1)
        plt.imshow(real_images_batch_tensor[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Real {i+1}")
        plt.axis('off')

        # Generated Image
        plt.subplot(2, num_side_by_side, i + num_side_by_side + 1)
        plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle("Real vs. Generated Images (Side-by-Side)", y=1.02)
    plt.show()
    logger.debug("Real vs. Generated side-by-side plots displayed.")