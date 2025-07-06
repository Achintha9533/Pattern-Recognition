# Synthetic Image Generator/main.py

import torch
import torch.optim as optim
import logging
import numpy as np
from typing import Dict, Any, Tuple

# Import configurations
from . import config

# Import modules from your package
from .dataset import LungCTWithGaussianDataset
from .transforms import get_transforms, get_fid_transforms
from .model import CNF_UNet
from .train import train_model
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

The application leverages a modular structure, with distinct functionalities
separated into `config`, `dataset`, `transforms`, `model`, `train`, `generate`,
`evaluate`, and `visualize` modules.
"""

# Configure basic logging for the entire application.
# Messages with INFO level and above will be displayed.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main function to orchestrate the data loading, model setup, training,
    generation, evaluation, and visualization process for the CNF-UNet.

    This function serves as the primary execution flow of the synthetic
    image generation application. It handles initialization, calls
    sub-modules for specific tasks, and manages the overall lifecycle.
    """
    logger.info("Starting Synthetic Image Generator application.")

    # === Device Setup ===
    # Determine whether to use CUDA (GPU) if available, otherwise fall back to CPU.
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Image Preprocessing Transforms ===
    # Get the transformation pipeline for model input (resizing, normalization).
    transform: T.Compose = get_transforms(config.IMAGE_SIZE)
    # Get the transformation pipeline specifically for FID calculation (denormalization, PIL conversion).
    fid_transform: T.Compose = get_fid_transforms()
    logger.info("Image transformations initialized.")

    # === Dataset and DataLoader ===
    # Initialize the custom dataset and DataLoader for efficient batch processing.
    try:
        dataset: LungCTWithGaussianDataset = LungCTWithGaussianDataset(
            base_dir=config.BASE_DIR,
            transform=transform,
            num_images_per_folder=config.NUM_IMAGES_PER_FOLDER,
            image_size=config.IMAGE_SIZE # Passed for black image fallback consistency
        )
        dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True # Speeds up data transfer to GPU if available
        )
        logger.info(f"Dataset loaded successfully with {len(dataset)} images.")
    except ValueError as e:
        # Log a critical error and exit if the dataset cannot be loaded,
        # as the application cannot proceed without data.
        logger.critical(f"Failed to load dataset: {e}. Exiting application.")
        return

    # === Model, Optimizer Setup ===
    # Initialize the CNF_UNet generator model and move it to the selected device.
    generator: CNF_UNet = CNF_UNet(time_embed_dim=256).to(device)
    # Initialize the Adam optimizer for the generator.
    optimizer_gen: optim.Adam = optim.Adam(generator.parameters(), lr=config.G_LR, betas=(0.5, 0.999))
    logger.info("Generator model and optimizer initialized.")

    # === Initial Data Distribution Plots ===
    # Retrieve one batch from the DataLoader for initial visualization of data and noise distributions.
    sample_noise_batch: Optional[torch.Tensor] = None
    sample_image_batch: Optional[torch.Tensor] = None
    for noise, image in dataloader:
        sample_noise_batch = noise.cpu() # Move to CPU for plotting
        sample_image_batch = image.cpu() # Move to CPU for plotting
        break # Only need one batch for initial samples

    if sample_noise_batch is not None and sample_image_batch is not None:
        plot_pixel_distributions(sample_image_batch, sample_noise_batch)
        plot_sample_images_and_noise(sample_image_batch, sample_noise_batch)
    else:
        logger.warning("Could not retrieve sample batch for initial visualization. Dataloader might be empty.")


    # === Training the CNF-UNet model ===
    logger.info(f"Starting training of the CNF-UNet model for {config.EPOCHS} epochs.")
    training_losses: Dict[str, Any] = train_model(
        generator_model=generator,
        dataloader=dataloader,
        optimizer_gen=optimizer_gen,
        epochs=config.EPOCHS,
        device=device
    )
    logger.info("Training complete.")

    # === Save Model Weights ===
    logger.info(f"Saving generator weights to {config.GENERATOR_CHECKPOINT_PATH}")
    # Save only the model's state dictionary (learnable parameters) for efficient storage.
    torch.save(generator.state_dict(), config.GENERATOR_CHECKPOINT_PATH)
    logger.info("Model weights saved successfully.")

    # === Plotting Training Losses ===
    plot_training_losses(training_losses)

    # === Sample Generation ===
    logger.info(f"Generating {config.NUM_GENERATED_SAMPLES} sample images for evaluation.")
    # Generate initial random noise for the generation process.
    initial_noise_for_generation: torch.Tensor = torch.randn(
        config.NUM_GENERATED_SAMPLES, 1, *config.IMAGE_SIZE
    ).to(device)
    # Generate images using the trained generator model.
    generated_images: torch.Tensor = generate_images(
        model=generator,
        initial_noise=initial_noise_for_generation,
        steps=config.GENERATION_STEPS,
        device=device
    )
    generated_images = generated_images.cpu() # Move generated images to CPU for evaluation and plotting

    # === Evaluation Metrics ===
    # Prepare real images for comparison with generated images.
    # A new DataLoader is created to ensure fresh, shuffled samples for evaluation.
    eval_dataloader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Shuffle to get different images for evaluation
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    real_images_for_eval: list[torch.Tensor] = []
    # Collect enough real images to match the number of generated samples.
    for _, real_img_batch_val in eval_dataloader:
        real_images_for_eval.append(real_img_batch_val)
        if len(real_images_for_eval) * eval_dataloader.batch_size >= config.NUM_GENERATED_SAMPLES:
            break
    # Concatenate collected batches and slice to match the exact number of generated samples.
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
    all_real_pixels: list[np.ndarray] = []
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

    logger.info("Application finished.")

if __name__ == "__main__":
    main()
