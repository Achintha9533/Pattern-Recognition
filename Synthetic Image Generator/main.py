# ====================================================
# File: main.py
# Description: Main script to orchestrate data loading, model training,
#              image generation, and visualization/evaluation.
# ====================================================

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import sys

# Import components from other files
# Adjust these imports based on how you structure your project's folders
# Assuming all .py files are in the same directory for simplicity here

# Parameters (can be moved to a config file)
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")
image_size = (64, 64)
lambda_gan = 0.1
G_LR = 1e-4
D_LR = 1e-4
epochs = 100
batch_size = 64
num_workers = 0 # Set to 0 for macOS compatibility

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # --- 1. Imports and Global Settings (Implicitly handled by importing modules) ---
    # Imports: os, pathlib, pydicom, numpy, torch, torchvision.transforms, warnings, matplotlib.pyplot, tqdm, torch.nn, torch.nn.functional, skimage.metrics
    # These would typically be at the top of a monolithic script, but for separation,
    # they are imported by the specific modules that need them.
    # Global constants like base_dir, image_size, lambda_gan, G_LR, D_LR are set here in main.

    # --- 2. Transform ---
    # The 'transform' object and 'load_dicom_image' function are defined in transform.py
    # Ensure the directory containing transform.py is in sys.path
    sys.path.append(str(Path(__file__).parent))
    from transform import transform, load_dicom_image

    # --- 3. Dataset ---
    # The 'LungCTWithGaussianDataset' class is defined in dataset.py
    # import from dataset.py if running as separate files
    from dataset import LungCTWithGaussianDataset

    print("Initializing dataset...")
    dataset = LungCTWithGaussianDataset(base_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    print(f"Total images loaded: {len(dataset)}")

    # Ensure dataloader is not empty
    if len(dataloader) == 0:
        raise RuntimeError("Dataloader is empty. No data to process. Check dataset path and contents.")


    # --- 4. Model ---
    # The 'UNetBlock', 'CNF_UNet', and 'Discriminator' classes are defined in model.py
    # import from model.py if running as separate files
    from model import CNF_UNet, Discriminator

    print("Initializing models...")
    generator = CNF_UNet().to(device)
    discriminator = Discriminator(img_channels=1, features_d=64).to(device)

    # Optimizers and Loss Function (part of model/training setup)
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=G_LR, betas=(0.5, 0.999))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=D_LR, betas=(0.5, 0.999))
    criterion_gan = nn.BCEWithLogitsLoss()

    # --- 5. Initial Data Visualization (using functions from visualize.py) ---
    from visualize import plot_initial_distributions, plot_initial_samples

    sample_noise_batch = None
    sample_image_batch = None
    for noise, image in dataloader:
        sample_noise_batch = noise.cpu()
        sample_image_batch = image.cpu()
        break
    
    if sample_image_batch is not None and sample_noise_batch is not None:
        plot_initial_distributions(sample_noise_batch, sample_image_batch)
        plot_initial_samples(sample_noise_batch, sample_image_batch)
    else:
        print("Could not get sample batches for initial visualization.")


    # --- 6. Train ---
    # The 'train' function is defined in train.py
    # import from train.py if running as separate files
    from train import train

    print("Starting training with GAN discriminator...")
    # Pass necessary components to the train function
    training_losses = train(
        generator_model=generator,
        discriminator_model=discriminator,
        dataloader=dataloader,
        epochs=epochs,
        lambda_gan=lambda_gan,
        optimizer_gen=optimizer_gen, # Pass optimizer and criterion
        optimizer_disc=optimizer_disc,
        criterion_gan=criterion_gan
    )
    print("Training finished.")

    # --- 7. Visualization and Evaluation ---
    # The 'generate' function is defined in generate.py
    # The plotting and evaluation functions are in visualize.py
    # import from generate.py and visualize.py
    from generate import generate
    from visualize import (
        plot_training_losses, evaluate_and_print_metrics,
        plot_pixel_distributions_comparison, plot_generated_samples,
        plot_real_vs_generated_side_by_side
    )

    # Plot training losses
    plot_training_losses(training_losses)

    # Sample generation
    num_generated_samples = 64
    initial_noise_for_generation = torch.randn(num_generated_samples, 1, *image_size).to(device)
    generated_images = generate(generator, initial_noise_for_generation, steps=200)
    generated_images = generated_images.cpu()

    # Get a batch of real images for comparison
    real_images_batch = None
    for _, real_img_batch_val in dataloader:
        real_images_batch = real_img_batch_val
        break # Get only one batch

    # Evaluate metrics
    evaluate_and_print_metrics(real_images_batch, generated_images, num_generated_samples, image_size)

    # Pixel Distribution Comparison
    all_real_pixels = []
    num_batches_to_sample = 5
    for i, (_, real_img_batch) in enumerate(dataloader):
        all_real_pixels.append(real_img_batch.cpu().view(-1).numpy())
        if i >= num_batches_to_sample - 1:
            break
    all_real_pixels_flat = np.concatenate(all_real_pixels)
    flat_generated_images = generated_images.view(-1).numpy()
    plot_pixel_distributions_comparison(all_real_pixels_flat, flat_generated_images)

    # Visualize generated samples
    plot_generated_samples(generated_images)

    # Visualize Real vs. Generated Side-by-Side
    plot_real_vs_generated_side_by_side(real_images_batch, generated_images)