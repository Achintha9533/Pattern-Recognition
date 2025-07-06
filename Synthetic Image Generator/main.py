# Synthetic Image Generator/main.py

import torch
import torch.optim as optim
import logging

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

# Configure basic logging for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """
    Main function to orchestrate the data loading, model setup, training,
    generation, evaluation, and visualization process for the CNF-UNet.
    """
    logger.info("Starting Synthetic Image Generator application.")

    # === Device Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Image Preprocessing Transforms ===
    transform = get_transforms(config.IMAGE_SIZE)
    fid_transform = get_fid_transforms()

    # === Dataset and DataLoader ===
    try:
        dataset = LungCTWithGaussianDataset(
            base_dir=config.BASE_DIR,
            transform=transform,
            num_images_per_folder=config.NUM_IMAGES_PER_FOLDER,
            image_size=config.IMAGE_SIZE
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
        logger.info(f"Dataset loaded successfully with {len(dataset)} images.")
    except ValueError as e:
        logger.critical(f"Failed to load dataset: {e}. Exiting.")
        return

    # === Model, Optimizer Setup ===
    generator = CNF_UNet(time_embed_dim=256).to(device)
    optimizer_gen = optim.Adam(generator.parameters(), lr=config.G_LR, betas=(0.5, 0.999))
    logger.info("Generator model and optimizer initialized.")

    # === Initial Data Distribution Plots ===
    sample_noise_batch = None
    sample_image_batch = None
    # Get one batch for initial visualization
    for noise, image in dataloader:
        sample_noise_batch = noise.cpu()
        sample_image_batch = image.cpu()
        break
    if sample_noise_batch is not None and sample_image_batch is not None:
        plot_pixel_distributions(sample_image_batch, sample_noise_batch)
        plot_sample_images_and_noise(sample_image_batch, sample_noise_batch)
    else:
        logger.warning("Could not retrieve sample batch for initial visualization.")


    # === Training the CNF-UNet model ===
    logger.info("Starting training of the CNF-UNet model.")
    training_losses = train_model(
        generator_model=generator,
        dataloader=dataloader,
        optimizer_gen=optimizer_gen,
        epochs=config.EPOCHS,
        device=device
    )

    # === Save Model Weights ===
    logger.info(f"Saving generator weights to {config.GENERATOR_CHECKPOINT_PATH}")
    torch.save(generator.state_dict(), config.GENERATOR_CHECKPOINT_PATH)
    logger.info("Model weights saved successfully.")

    # === Plotting Training Losses ===
    plot_training_losses(training_losses)

    # === Sample Generation ===
    logger.info(f"Generating {config.NUM_GENERATED_SAMPLES} sample images for evaluation.")
    initial_noise_for_generation = torch.randn(config.NUM_GENERATED_SAMPLES, 1, *config.IMAGE_SIZE).to(device)
    generated_images = generate_images(
        model=generator,
        initial_noise=initial_noise_for_generation,
        steps=config.GENERATION_STEPS,
        device=device
    )
    generated_images = generated_images.cpu() # Move to CPU for evaluation and plotting

    # === Evaluation Metrics ===
    # Get a batch of real images for comparison.
    # Reset dataloader iterator to get fresh samples for evaluation
    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True, # Shuffle to get different images
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    real_images_for_eval = []
    for _, real_img_batch_val in eval_dataloader:
        real_images_for_eval.append(real_img_batch_val)
        if len(real_images_for_eval) * eval_dataloader.batch_size >= config.NUM_GENERATED_SAMPLES:
            break
    real_images_batch_tensor = torch.cat(real_images_for_eval, dim=0)[:config.NUM_GENERATED_SAMPLES]

    evaluate_model(
        real_images_batch_tensor=real_images_batch_tensor,
        generated_images=generated_images,
        fid_transform=fid_transform,
        num_compare=config.NUM_GENERATED_SAMPLES
    )

    # === Distribution Plot for Generated Images vs. Real Images ===
    all_real_pixels = []
    # Use the eval_dataloader to get a diverse sample for distribution plot
    num_batches_to_sample = config.NUM_BATCHES_FOR_DIST_PLOT
    for i, (_, real_img_batch) in enumerate(eval_dataloader):
        all_real_pixels.append(real_img_batch.cpu().view(-1).numpy())
        if i >= num_batches_to_sample - 1:
            break
    all_real_pixels_flat = np.concatenate(all_real_pixels)
    flat_generated_images = generated_images.view(-1).numpy()

    plot_generated_pixel_distribution_comparison(all_real_pixels_flat, flat_generated_images)

    # === Visualize Sample Generated Images ===
    plot_sample_generated_images(generated_images)

    # === Visualize Real vs. Generated Side-by-Side ===
    plot_real_vs_generated_side_by_side(real_images_batch_tensor, generated_images)

    logger.info("Application finished.")

if __name__ == "__main__":
    main()