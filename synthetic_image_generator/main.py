# Synthetic Image Generator/main.py

import torch
import torch.optim as optim
import logging
import numpy as np
from typing import Dict, Any, Tuple, Union, List # Added 'List' import
from tqdm import tqdm # Import tqdm for progress bars
import torch.nn.functional as F # Import F for functional operations
import os

# Import configurations
from . import config

# Import modules from your package
from .dataset import LungCTWithGaussianDataset
from .transforms import get_transforms, get_fid_transforms
from .model import CNF_UNet
from .generate import generate_images
from .evaluate import evaluate_model
from .visualize import (
    plot_pixel_distributions,
    plot_sample_images_and_noise,
    # plot_training_losses, # Removed as training is removed
    plot_generated_pixel_distribution_comparison,
    plot_sample_generated_images,
    plot_real_vs_generated_side_by_side
)
from .load_model import load_model_from_drive # Added for model loading

# Configure logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Module for orchestrating the synthetic image generation workflow using a pre-trained CNF-UNet model.

This `main` module serves as the primary entry point for the Synthetic Image Generator project. It
integrates various components including data loading, preprocessing, model initialization,
pre-trained model loading, synthetic image generation, and comprehensive evaluation
of the generated images. Its design facilitates a streamlined execution flow for
demonstrating the capabilities of the generative model without requiring on-the-fly training.

Inputs:
    None (The module's behavior is configured via `config.py` and command-line execution).

Outputs:
    Side effects include console logs, saved images to disk (via evaluation module),
    and displayed plots (via visualization module).

Potential Exceptions Raised:
    -   `FileNotFoundError`: If required data directories or model checkpoint files are not found.
    -   `RuntimeError`: For general PyTorch execution errors, e.g., device-related issues,
        or numerical instability during generation.
    -   `ImportError`: If dependent modules within the project structure (e.g., `.config`,
        `.model`) cannot be imported.

Theory:
The core theoretical basis of this project relies on Conditional Normalizing Flows (CNF),
specifically implementing a U-Net architecture for the flow's transformation function.
CNFs learn a continuous transformation from a simple base distribution (e.g., Gaussian noise)
to a complex target data distribution (e.g., medical images). Unlike discrete generative models,
CNFs allow for exact likelihood evaluation and efficient sampling by solving an ordinary
differential equation (ODE). The pre-trained model handles the reverse process of this flow,
transforming noise back into realistic images.

References for the theory:
-   Grathwohl, J., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. K. (2018).
    *Continuous Normalizing Flows*. arXiv preprint arXiv:1806.02373.
-   Lipman, Y., Chen, R. T. Q., & Duvenaud, D. K. (2022).
    *Flow Matching for Generative Modeling*. International Conference on Learning Representations.
-   Ronneberger, O., Fischer, P., & Brox, T. (2015).
    *U-Net: Convolutional Networks for Biomedical Image Segmentation*. International Conference on Medical Image Computing and Computer-Assisted Intervention.
-   Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017).
    *FID: Fréchet Inception Distance for evaluating generative models*. arXiv preprint arXiv:1706.08500.
"""

def main() -> None:
    """
    Executes the end-to-end workflow for synthetic image generation and evaluation.

    This function coordinates the setup of the computing environment, data handling,
    model loading, image synthesis, and result analysis. It ensures a complete
    demonstration of the generative model's capabilities from loading a
    pre-trained model to visualizing the final generated output and evaluating its quality.

    Args:
        None

    Returns:
        None (The function performs actions such as saving images and displaying plots).

    Raises:
        RuntimeError: May occur if a specified PyTorch operation fails,
            e.g., due to CUDA memory issues or if the `device` setup fails.
        FileNotFoundError: If paths specified in `config.py` (e.g., `BASE_DIR`,
            `GENERATOR_CHECKPOINT_PATH`) do not exist.
        Exception: Broader exceptions might be raised by external libraries
            (e.g., `gdown` for downloading, `torch_fidelity` for evaluation)
            due to network issues, corrupted files, or internal library errors.

    Example:
        To run this main pipeline, ensure all dependencies are installed and
        execute the script from your terminal within the project's root directory:
        ```bash
        python -m your_package_name.main
        ```
        (Replace `your_package_name` with the actual name of your Python package,
        e.g., `Synthetic_Image_Generator`).

    Relationships with other functions:
        -   Calls `config` for all global parameters (`IMAGE_SIZE`, `BASE_DIR`, etc.).
        -   Initializes `dataset.LungCTWithGaussianDataset` for data loading.
        -   Utilizes `transforms.get_transforms` and `transforms.get_fid_transforms`
            for image preprocessing.
        -   Instantiates `model.CNF_UNet` to define the generator's architecture.
        -   Invokes `load_model.load_model_from_drive` to load the pre-trained model weights.
        -   Employs `generate.generate_images` to perform the synthetic image generation.
        -   Uses `evaluate.evaluate_model` to compute quantitative metrics like FID, MSE, PSNR, SSIM.
        -   Relies on `visualize.plot_*` functions (`plot_pixel_distributions`, `plot_sample_images_and_noise`,
            `plot_generated_pixel_distribution_comparison`, `plot_sample_generated_images`,
            `plot_real_vs_generated_side_by_side`) for qualitative assessment and debugging.

    Theory:
        The workflow encompasses key aspects of generative modeling inference.
        It begins by configuring the `torch.device` (GPU for speed or CPU).
        Dataset loading involves reading DICOM files and applying `transforms` to
        normalize and resize images for the model. The core step is loading a
        pre-trained `CNF_UNet` which has already learned the velocity field
        mapping from noise to data. Image generation is then performed by
        numerically integrating this velocity field using random Gaussian noise
        as the starting point. Finally, `evaluate_model` quantifies the quality
        and realism of the generated samples against real data using metrics
        like FID, while `visualize` functions provide intuitive visual insights
        into the model's performance.
    """
    # === Device Configuration ===
    # Set up the device to use for processing (GPU if available, else CPU).
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # === Data Transformations ===
    # Define image transformations for preprocessing.
    # These transforms will resize images and normalize pixel values to [-1, 1].
    transforms = get_transforms(image_size=config.IMAGE_SIZE)
    # Define FID specific transformations for evaluation metrics.
    # These transform images to a format suitable for FID calculation (denormalized to [0, 1]).
    fid_transform = get_fid_transforms(image_size=config.IMAGE_SIZE)

    # === Dataset and DataLoader Initialization ===
    logger.info(f"Loading dataset from: {config.BASE_DIR}")
    # Initialize the custom dataset for Lung CT images, applying defined transformations.
    full_dataset = LungCTWithGaussianDataset(
        base_dir=config.BASE_DIR,
        num_images_per_folder=config.NUM_IMAGES_PER_FOLDER,
        image_size=config.IMAGE_SIZE,
        transform=transforms
    )

    # Create DataLoaders for efficient batch processing during evaluation.
    # train_dataloader is primarily for sampling real images for initial visualization and comparisons.
    train_dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True # Speeds up data transfer to GPU
    )
    # eval_dataloader is used for evaluation, shuffle=False to maintain order if needed.
    eval_dataloader = torch.utils.data.DataLoader(
        full_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    logger.info(f"Dataset loaded. Total images: {len(full_dataset)}")

    # === Model and Optimizer Setup ===
    # Initialize the generator model (CNF_UNet) with its architecture parameters.
    generator = CNF_UNet(
        image_channels=1, # Single channel for grayscale CT images
        base_channels=64, # Base number of channels for the U-Net
        embed_dim=256 # Dimension for time embeddings
    ).to(device)

    # === Loading a Pre-trained Model ===
    logger.info("Loading pre-trained generator model...")
    # Using a placeholder Google Drive link for the pre-trained model weights.
    # You should replace this with the actual link to your pre-trained model.
    google_drive_link = "1P-2cR47f1_wR_o08Q9hL8GjK4b0B0B0" # Example link from load_model.py
    generator = load_model_from_drive(
        drive_url=google_drive_link,
        output_path=config.GENERATOR_CHECKPOINT_PATH, # Reusing checkpoint path for saving downloaded weights temporarily
        image_size=config.IMAGE_SIZE,
        device=device
    )
    logger.info("Pre-trained model loaded successfully.")

    # === Initial Data Visualization (before any generation) ===
    # Sample a batch of real images and their corresponding Gaussian noise for initial visualization.
    sample_batch_real, sample_batch_noise = next(iter(train_dataloader))
    # Ensure samples are on CPU for plotting as matplotlib typically works with numpy arrays.
    plot_pixel_distributions(sample_batch_real.cpu(), sample_batch_noise.cpu())
    plot_sample_images_and_noise(sample_batch_real.cpu(), sample_batch_noise.cpu())

    # === Image Generation ===
    logger.info("Generating synthetic images...")
    # Generate initial noise samples for synthetic image generation.
    # The number of samples is defined in config.NUM_GENERATED_SAMPLES.
    initial_noise_for_generation: torch.Tensor = torch.randn(
        config.NUM_GENERATED_SAMPLES,
        1, # Single channel
        *config.IMAGE_SIZE # Height and width from configuration
    ).to(device)

    # Use the loaded generator model to produce synthetic images from noise.
    # The generation process involves a series of Euler integration steps.
    generated_images: torch.Tensor = generate_images(
        model=generator,
        initial_noise=initial_noise_for_generation,
        steps=config.GENERATION_STEPS,
        device=device
    )
    logger.info("Synthetic image generation complete.")

    # Convert generated images to CPU and denormalize for evaluation and visualization.
    # From [-1, 1] to [0, 1] and then to numpy arrays for metric calculation and plotting.
    generated_images = (generated_images.cpu() + 1) / 2.0 # Denormalize from [-1, 1] to [0, 1]

    # === Evaluation of Generated Images ===
    logger.info("Starting model evaluation...")
    real_images_for_eval: List[torch.Tensor] = []
    # Collect real images from the evaluation dataloader for comparison.
    # Only collect up to NUM_GENERATED_SAMPLES to match the generated batch size for fair comparison.
    for i, (_, real_img_batch) in enumerate(eval_dataloader):
        real_images_for_eval.append((real_img_batch.cpu() + 1) / 2.0) # Denormalize real images as well
        if len(real_images_for_eval) * config.BATCH_SIZE >= config.NUM_GENERATED_SAMPLES:
            break
    # Concatenate and select the required number of real images.
    real_images_batch_tensor = torch.cat(real_images_for_eval, dim=0)[:config.NUM_GENERATED_SAMPLES]

    # Call the evaluation module to calculate and log metrics.
    evaluate_model(
        real_images_batch_tensor=real_images_batch_tensor,
        generated_images=generated_images,
        fid_transform=fid_transform,
        num_compare=config.NUM_GENERATED_SAMPLES
    )

    # === Distribution Plot for Generated Images vs. Real Images ===
    logger.info("Preparing pixel distribution comparison plot.")
    all_real_pixels: List[np.ndarray] = []
    # Collect pixel data from a few batches of real images for a representative distribution.
    num_batches_to_sample: int = config.NUM_BATCHES_FOR_DIST_PLOT
    for i, (_, real_img_batch) in enumerate(eval_dataloader):
        all_real_pixels.append(real_img_batch.cpu().view(-1).numpy())
        if i >= num_batches_to_sample - 1:
            break
    all_real_pixels_flat: np.ndarray = np.concatenate(all_real_pixels)
    flat_generated_images: np.ndarray = generated_images.view(-1).numpy()

    plot_generated_pixel_distribution_comparison(all_real_pixels_flat, flat_generated_images)

    # === Visualize Sample Generated Images ===
    plot_sample_generated_images(generated_images)

    # === Visualize Real vs. Generated Side-by-Side ===
    plot_real_vs_generated_side_by_side(real_images_batch_tensor, generated_images)

    logger.info("Main script execution finished.")

if __name__ == "__main__":
    main()