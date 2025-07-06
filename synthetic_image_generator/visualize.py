# Synthetic Image Generator/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from typing import Dict, Any, List

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module provides various visualization utilities for the Synthetic Image Generator project.
It includes functions to plot pixel distributions, display sample images (real, noise, generated),
visualize training loss curves, and compare real vs. generated images side-by-side.

These functions are designed to help understand the data, monitor training progress,
and qualitatively assess the quality of generated images. As per the project guidelines,
visualization functions do not perform data preprocessing and are generally not unit-tested,
focusing solely on graphical representation.
"""

def plot_pixel_distributions(sample_image_batch: torch.Tensor, sample_noise_batch: torch.Tensor) -> None:
    """
    Plots the pixel distribution histograms for a batch of real CT images and
    a batch of initial Gaussian noise samples. This helps to visualize the
    input distributions to the generative model.

    Visualizing these distributions is important to understand the range and
    spread of pixel values in the real data that the model is trying to learn,
    and to confirm that the generated noise has a Gaussian distribution.

    Args:
        sample_image_batch (torch.Tensor): A batch of real CT image tensors (CPU).
                                           Expected pixel range: [-1, 1].
                                           Shape: (batch_size, C, H, W).
        sample_noise_batch (torch.Tensor): A batch of initial Gaussian noise tensors (CPU).
                                           Expected pixel range: typically centered around 0
                                           with std dev 1 (standard normal).
                                           Shape: (batch_size, C, H, W).

    Returns:
        None: Displays a matplotlib plot.

    Potential Exceptions Raised:
        - ValueError: If input tensors are empty.
        - RuntimeError: If `torch.Tensor.view(-1).numpy()` operations fail.

    Example of Usage:
    ```python
    import torch
    # Assuming sample_real_images and sample_noise are torch.Tensors in their respective ranges
    # sample_real_images = torch.randn(10, 1, 64, 64) * 0.5 + 0.5 # Example real images
    # sample_noise = torch.randn(10, 1, 64, 64) # Example noise
    # plot_pixel_distributions(sample_real_images, sample_noise)
    ```

    Relationships with other functions:
    - Called by `main.py` during initial setup.

    Explanation of the theory:
    - **Histogram:** A graphical representation of the distribution of numerical data.
      It is an estimate of the probability distribution of a continuous variable.
    - **Pixel Distribution:** Shows the frequency of different pixel intensity values
      in an image or a collection of images. For CT images, this typically reflects
      the Hounsfield Unit (HU) values after normalization. For Gaussian noise, it
      should approximate a bell curve.
    """
    logger.info("Plotting initial data and noise pixel distributions.")
    # Flatten the tensors to get all pixel values into a 1D array for histogram plotting.
    flat_images = sample_image_batch.cpu().view(-1).numpy()
    flat_noise = sample_noise_batch.cpu().view(-1).numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(flat_images, bins=50, density=True, alpha=0.7, color='blue')
    plt.title('Real Image Pixel Distribution')
    plt.xlabel('Pixel Value (Normalized)')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.hist(flat_noise, bins=50, density=True, alpha=0.7, color='green')
    plt.title('Gaussian Noise Pixel Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_sample_images_and_noise(sample_image_batch: torch.Tensor, sample_noise_batch: torch.Tensor, num_samples: int = 5) -> None:
    """
    Displays a few sample real images and their corresponding initial Gaussian noise samples.

    This function provides a visual sanity check, allowing users to see typical
    inputs to the generative model and understand the raw material (noise) from
    which synthetic images will be generated. It helps confirm data loading
    and basic transformations.

    Args:
        sample_image_batch (torch.Tensor): A batch of real CT image tensors (CPU).
                                           Expected pixel range: [-1, 1].
                                           Shape: (batch_size, C, H, W).
        sample_noise_batch (torch.Tensor): A batch of initial Gaussian noise tensors (CPU).
                                           Shape: (batch_size, C, H, W).
        num_samples (int): The number of sample pairs (image, noise) to display. Defaults to 5.

    Returns:
        None: Displays a matplotlib plot.

    Potential Exceptions Raised:
        - ValueError: If input tensors are empty or `num_samples` is larger than batch size.

    Example of Usage:
    ```python
    import torch
    # Assuming sample_real_images and sample_noise are torch.Tensors
    # plot_sample_images_and_noise(sample_real_images, sample_noise, num_samples=3)
    ```

    Relationships with other functions:
    - Called by `main.py` during initial setup.

    Explanation of the theory:
    - **Visual Inspection:** A fundamental step in any machine learning project
      to quickly identify issues with data loading, preprocessing, or model outputs.
      For generative models, visually comparing inputs (noise) to desired outputs (real images)
      helps set expectations for the model's task.
    """
    logger.info(f"Displaying {num_samples} sample real images and corresponding noise.")
    num_to_display: int = min(num_samples, sample_image_batch.shape[0], sample_noise_batch.shape[0])

    if num_to_display == 0:
        logger.warning("No samples to display for initial images and noise. Skipping plot.")
        return

    plt.figure(figsize=(num_to_display * 2, 4)) # Adjust figure size dynamically
    for i in range(num_to_display):
        # Display Real Image
        plt.subplot(2, num_to_display, i + 1)
        # Ensure image is single channel and move to CPU, then convert to NumPy
        plt.imshow(sample_image_batch[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Real {i+1}")
        plt.axis('off')

        # Display Noise Image
        plt.subplot(2, num_to_display, i + num_to_display + 1)
        plt.imshow(sample_noise_batch[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Noise {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_training_losses(gen_flow_losses: List[float]) -> None:
    """
    Plots the training loss curve (Generator Flow Matching Loss) over epochs.

    Monitoring the loss curve is essential to assess training stability and
    convergence. A decreasing loss generally indicates that the model is learning,
    while oscillations or divergence might signal issues with hyperparameters
    or the model architecture.

    Args:
        gen_flow_losses (List[float]): A list of average Flow Matching losses,
                                       one value per epoch.

    Returns:
        None: Displays a matplotlib plot.

    Potential Exceptions Raised:
        - ValueError: If `gen_flow_losses` is empty.

    Example of Usage:
    ```python
    # Assuming `losses` is a list of floats from training
    # plot_training_losses([0.8, 0.6, 0.4, 0.3, 0.25])
    ```

    Relationships with other functions:
    - Called by `main.py` after the training loop completes.
    - Data typically comes from the return value of `train.train_model`.

    Explanation of the theory:
    - **Loss Function:** A mathematical function that quantifies the difference
      between the predicted output of a model and the true target values. The goal
      of training is to minimize this loss.
    - **Training Curve:** A plot of the loss function (or other metrics) against
      the number of training epochs or iterations. It provides insights into
      whether the model is overfitting, underfitting, or converging.
    """
    logger.info("Plotting training loss curve.")
    if not gen_flow_losses:
        logger.warning("No generator flow losses provided to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(gen_flow_losses, label='Generator Flow Matching Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_generated_pixel_distribution_comparison(real_pixels: np.ndarray, generated_pixels: np.ndarray) -> None:
    """
    Plots the pixel distribution histograms comparing real images and generated synthetic images.

    This visualization helps to quantitatively assess how well the generative model
    is capturing the statistical properties of the real data. A good generative
    model should produce images whose pixel distributions closely match those
    of the real images.

    Args:
        real_pixels (np.ndarray): A 1D NumPy array containing all pixel values
                                  from a sample of real images (flattened).
                                  Expected pixel range: [-1, 1].
        generated_pixels (np.ndarray): A 1D NumPy array containing all pixel values
                                       from a sample of generated images (flattened).
                                       Expected pixel range: [-1, 1].

    Returns:
        None: Displays a matplotlib plot.

    Potential Exceptions Raised:
        - ValueError: If input arrays are empty.

    Example of Usage:
    ```python
    import numpy as np
    # Assume real_data_pixels and gen_data_pixels are flattened NumPy arrays
    # plot_generated_pixel_distribution_comparison(real_data_pixels, gen_data_pixels)
    ```

    Relationships with other functions:
    - Called by `main.py` after image generation and evaluation.
    - Data prepared in `main.py` by flattening tensors.

    Explanation of the theory:
    - **Distribution Matching:** A core goal of generative modeling is to learn
      the underlying data distribution such that generated samples resemble
      real samples not just visually, but also statistically. Comparing pixel
      histograms is a simple way to check for this.
    """
    logger.info("Plotting generated vs. real pixel distribution comparison.")
    if real_pixels.size == 0 or generated_pixels.size == 0:
        logger.warning("No pixel data to compare. Skipping distribution plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(real_pixels, bins=50, density=True, alpha=0.7, label='Real Images', color='blue')
    plt.hist(generated_pixels, bins=50, density=True, alpha=0.7, label='Generated Images', color='red')
    plt.title('Pixel Distribution: Real vs. Generated Images')
    plt.xlabel('Pixel Value (Normalized)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sample_generated_images(generated_images: torch.Tensor, num_samples: int = 25) -> None:
    """
    Displays a grid of sample synthetic images generated by the model.

    This function provides a qualitative assessment of the model's generative
    capabilities. It allows for visual inspection of the realism, diversity,
    and overall quality of the images produced by the Conditional Normalizing Flow.

    Args:
        generated_images (torch.Tensor): A batch of generated image tensors (CPU).
                                         Expected pixel range: [-1, 1].
                                         Shape: (batch_size, C, H, W).
        num_samples (int): The number of generated images to display in the grid.
                           Defaults to 25 (5x5 grid).

    Returns:
        None: Displays a matplotlib plot.

    Potential Exceptions Raised:
        - ValueError: If `generated_images` is empty or `num_samples` is larger than batch size.

    Example of Usage:
    ```python
    import torch
    # Assuming `generated_imgs` is a batch of generated tensors from your model
    # plot_sample_generated_images(generated_imgs, num_samples=16)
    ```

    Relationships with other functions:
    - Called by `main.py` after the image generation phase.
    - Uses images generated by `generate.py`.

    Explanation of the theory:
    - **Qualitative Assessment:** Visual inspection remains a critical component
      of evaluating generative models. While quantitative metrics (like FID) are important,
      human perception of image quality, realism, and diversity is often the ultimate
      measure of success.
    """
    logger.info(f"Displaying {num_samples} sample generated images.")
    num_to_display: int = min(num_samples, generated_images.shape[0])
    if num_to_display == 0:
        logger.warning("No generated images to display. Skipping plot.")
        return

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_to_display)))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))

    for i in range(num_to_display):
        plt.subplot(grid_size, grid_size, i + 1)
        # Ensure image is single channel and move to CPU, then convert to NumPy
        plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_real_vs_generated_side_by_side(
    real_images_batch_tensor: torch.Tensor,
    generated_images: torch.Tensor,
    num_side_by_side: int = 5
) -> None:
    """
    Displays pairs of real and generated images side-by-side for direct visual comparison.

    This function facilitates a direct qualitative comparison between the actual
    training data and the synthetic data produced by the model. It allows for
    a quick assessment of visual realism, texture, and structural details learned
    by the generative model.

    Args:
        real_images_batch_tensor (torch.Tensor): A batch of real image tensors (CPU).
                                                 Expected pixel range: [-1, 1].
                                                 Shape: (batch_size, C, H, W).
        generated_images (torch.Tensor): A batch of generated image tensors (CPU).
                                         Expected pixel range: [-1, 1].
                                         Shape: (batch_size, C, H, W).
        num_side_by_side (int): The number of real-vs-generated pairs to display.
                                Defaults to 5.

    Returns:
        None: Displays a matplotlib plot.

    Potential Exceptions Raised:
        - ValueError: If either input tensor is empty, or `num_side_by_side`
                      exceeds the available samples in either batch.

    Example of Usage:
    ```python
    import torch
    # Assuming real_imgs_batch and gen_imgs_batch are batches of torch.Tensors
    # plot_real_vs_generated_side_by_side(real_imgs_batch, gen_imgs_batch, num_side_by_side=3)
    ```

    Relationships with other functions:
    - Called by `main.py` as a final visualization step.

    Explanation of the theory:
    - **Visual Comparison:** Directly juxtaposing real and synthetic samples is
      one of the most intuitive ways to evaluate a generative model. It highlights
      whether the model can replicate the characteristics of the real data,
      including textures, edges, and overall composition.
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

    plt.figure(figsize=(num_effective_pairs * 2, 4)) # Adjust figure size dynamically
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
    plt.show()