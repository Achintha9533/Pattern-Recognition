# Synthetic Image Generator/main.py

import torch
import torch.optim as optim
import logging
import numpy as np
from typing import Dict, Any, Tuple, Union
from tqdm import tqdm # Import tqdm for progress bars in training
import torch.nn.functional as F # Import F for functional operations in training
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
    plot_training_losses,
    plot_generated_pixel_distribution_comparison,
    plot_sample_generated_images,
    plot_real_vs_generated_side_by_side
)

# Configure logging for the main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Main application entry point for the Synthetic Image Generator.

This script orchestrates the entire workflow of the project:
1.  Device setup (CUDA or CPU).
2.  Image preprocessing transformations.
3.  Dataset and DataLoader initialization.
4.  Model (CNF-UNet) and optimizer setup.
5.  Initial data visualization.
6.  Training of the generative model using the Flow Matching objective.
7.  Saving of trained model weights.
8.  Plotting of training losses.
9.  Generation of synthetic images.
10. Evaluation of generated images using various metrics (MSE, PSNR, SSIM, FID).
11. Further visualization of generated image quality and distributions.

The application uses a Conditional Normalizing Flow (CNF) model, specifically a U-Net
architecture, to generate synthetic medical images (e.g., Lung CT scans) from Gaussian noise.
It leverages Flow Matching for training stability and efficiency.
"""

def main() -> None:
    """
    Executes the full pipeline for training, generating, and evaluating the
    Synthetic Image Generator model.

    This function encapsulates the entire workflow:
    - Initializes hardware device (GPU if available, else CPU).
    - Configures data transformations.
    - Sets up the dataset and data loaders.
    - Instantiates the CNF_UNet model and its optimizer.
    - Performs initial data visualization (pixel distributions, sample images).
    - Runs the training loop, saving model checkpoints.
    - Generates new synthetic images using the trained model.
    - Evaluates the generated images against real data using various metrics.
    - Visualizes training progress and generated image quality.

    No direct inputs are taken as arguments; all configurations are loaded
    from the `config` module.

    Returns:
        None: The function completes the entire training and evaluation process.

    Potential Exceptions Raised:
        - FileNotFoundError: If `config.BASE_DIR` is incorrect or data files are missing.
        - RuntimeError: If PyTorch operations fail (e.g., out of GPU memory).
        - Any exceptions propagated from `dataset.py`, `train.py`, `generate.py`,
          `evaluate.py`, or `visualize.py` during their respective operations.

    Example of Usage:
    ```python
    # To run the entire pipeline, simply execute this module:
    # python -m your_package_name.main
    # or if main.py is in the top level and runnable:
    # python main.py
    ```

    Relationships with other functions/modules:
    - Heavily relies on `config` for all hyperparameters and paths.
    - Imports and calls functions from `dataset`, `transforms`, `model`,
      `train`, `generate`, `evaluate`, and `visualize`. This is the orchestrator.

    Explanation of the theory:
    - **Orchestration:** This `main` function serves as the central control flow,
      integrating all disparate components of the generative modeling pipeline.
      It demonstrates a typical machine learning project structure.
    - **Hyperparameter Management:** All critical parameters are externalized to
      `config.py`, making the system flexible and easy to experiment with different
      settings without modifying the core logic.

    References for the theory:
    - Standard practices in machine learning project structuring.
    - Best practices for experiment management.
    """
    logger.info("Starting Synthetic Image Generator pipeline.")

    # === Device Setup ===
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Data Transformations ===
    # Transforms for preparing input images for the model (resize, normalize to [-1, 1])
    data_transform = get_transforms(image_size=config.IMAGE_SIZE)
    # Transforms specifically for FID calculation (de-normalize to [0,1], convert to PIL)
    fid_transform = get_fid_transforms()

    # === Dataset and DataLoader ===
    # Initialize the custom dataset
    dataset = LungCTWithGaussianDataset(
        base_dir=config.BASE_DIR,
        num_images_per_folder=config.NUM_IMAGES_PER_FOLDER,
        image_size=config.IMAGE_SIZE,
        transform=data_transform
    )
    # Create DataLoader for batching and shuffling data
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    logger.info(f"Dataset initialized with {len(dataset)} images. DataLoader batch size: {config.BATCH_SIZE}")

    # Create a separate DataLoader for evaluation images for FID
    # This is to ensure that a diverse set of real images is used for FID calculation
    # without being limited by the training batch size or order.
    # It takes all found images, ensuring a larger sample for FID.
    eval_dataset = LungCTWithGaussianDataset(
        base_dir=config.BASE_DIR,
        # Fetch all available images for a robust FID calculation
        num_images_per_folder=dataset.num_images_per_folder * 50, # Use a large number to get all images
        image_size=config.IMAGE_SIZE,
        transform=data_transform
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False, # No need to shuffle for evaluation
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    logger.info(f"Evaluation dataset initialized with {len(eval_dataset)} images.")


    # === Model Setup ===
    # Initialize the CNF_UNet generator model
    # Assuming single-channel (grayscale) images
    generator_model = CNF_UNet(img_channels=1, time_embed_dim=256, base_channels=64).to(device)
    logger.info(f"Generator model initialized and moved to {device}.")

    # Load pre-trained model if it exists
    if config.GENERATOR_CHECKPOINT_PATH.exists():
        logger.info(f"Loading generator checkpoint from {config.GENERATOR_CHECKPOINT_PATH}")
        try:
            generator_model.load_state_dict(torch.load(config.GENERATOR_CHECKPOINT_PATH, map_location=device))
            logger.info("Generator model checkpoint loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading generator checkpoint: {e}. Starting training from scratch.")
    else:
        logger.info("No generator checkpoint found. Starting training from scratch.")

    # Initialize the optimizer for the generator
    optimizer_gen = optim.Adam(generator_model.parameters(), lr=config.G_LR)
    logger.info(f"Optimizer initialized with learning rate: {config.G_LR}")

    # === Initial Data Visualization ===
    # Get a sample batch for initial visualization of distributions
    if len(dataset) > 0:
        sample_noise_batch, sample_image_batch = next(iter(dataloader))
        plot_pixel_distributions(sample_image_batch, sample_noise_batch)
        plot_sample_images_and_noise(sample_image_batch, sample_noise_batch)
    else:
        logger.warning("Dataset is empty. Skipping initial data visualizations.")


    # === Training the Model ===
    logger.info("Starting model training...")
    training_results = {}
    try:
        from .train import train_model # Import here to avoid circular dependencies during initial imports

        training_results = train_model(
            generator_model=generator_model,
            dataloader=dataloader,
            optimizer_gen=optimizer_gen,
            epochs=config.EPOCHS,
            device=device
        )
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        # Optionally exit or handle the error gracefully
        return

    # Save the trained generator model
    try:
        torch.save(generator_model.state_dict(), config.GENERATOR_CHECKPOINT_PATH)
        logger.info(f"Trained generator model saved to {config.GENERATOR_CHECKPOINT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save generator model checkpoint: {e}")

    # Plot training losses
    if 'gen_flow_losses' in training_results and training_results['gen_flow_losses']:
        plot_training_losses(training_results['gen_flow_losses'])
    else:
        logger.warning("No training loss data to plot.")

    # === Image Generation ===
    logger.info("Generating synthetic images for evaluation.")
    # Create initial noise for generation (same shape as expected image output)
    # The number of samples for generation is defined in config.
    generation_noise = torch.randn(
        config.NUM_GENERATED_SAMPLES,
        1, # Single channel for grayscale images
        config.IMAGE_SIZE[0],
        config.IMAGE_SIZE[1]
    ).to(device)

    generated_images = generate_images(
        model=generator_model,
        initial_noise=generation_noise,
        steps=config.GENERATION_STEPS,
        device=device
    )
    # Move generated images back to CPU for evaluation/visualization
    generated_images = generated_images.cpu()
    logger.info(f"Generated {generated_images.shape[0]} synthetic images.")

    # === Model Evaluation ===
    logger.info("Evaluating generated images against real data.")
    real_images_for_eval: List[torch.Tensor] = []
    # Collect real images from the eval_dataloader up to NUM_GENERATED_SAMPLES
    # to ensure consistency for metrics that compare real vs generated sets.
    # It's important to collect enough real images to match the count of generated ones.
    for _, real_img_batch in tqdm(eval_dataloader, desc="Collecting Real Images for Evaluation"):
        real_images_for_eval.append(real_img_batch)
        if sum(t.shape[0] for t in real_images_for_eval) >= config.NUM_GENERATED_SAMPLES:
            break
    # Concatenate and potentially truncate to match the exact number of generated samples.
    real_images_batch_tensor: torch.Tensor = torch.cat(real_images_for_eval, dim=0)[:config.NUM_GENERATED_SAMPLES]

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
    plot_real_vs_generated_side_by_side(real_images_batch_tensor, generated_images, config.NUM_SAMPLES_SIDE_BY_SIDE)

    logger.info("Synthetic Image Generator pipeline finished.")


if __name__ == "__main__":
    main()