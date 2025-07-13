"""
Main execution script for the Synthetic Image Generator project.

This script orchestrates the entire workflow of the synthetic image generator,
from setting up the environment and loading data to model initialization,
weight management (downloading and loading), image generation, evaluation
of performance metrics, and visualization of results.

The pipeline includes:
- Directory setup for checkpoints.
- Loading the custom Lung CT dataset and creating data loaders.
- Initializing the CNF_UNet generative model.
- Automatically downloading pre-trained model weights from Google Drive
  if they are not found locally, with robust error handling for downloads
  and file corruption.
- Loading the generator's state dictionary.
- Generating synthetic images.
- Evaluating various image quality metrics (MSE, PSNR, SSIM, FID).
- Visualizing pixel distributions and sample images (real, noise, generated,
  and side-by-side comparisons).

Functions
---------
is_html_file(filepath)
    Checks if a given file contains HTML content, which can indicate a
    failed or corrupted Google Drive download.
main()
    The primary function that executes the full pipeline of the synthetic
    image generator.

Notes
-----
- Relies on configuration settings defined in `config.py`.
- Utilizes `dataset.py` for data loading, `model.py` for the generator
  architecture, `utils.py` for Google Drive downloads, `generate.py` for
  image synthesis, `evaluate.py` for metric calculations, and `visualize.py`
  for plotting results.
- Includes robust checks for downloaded weights to prevent errors from
  HTML content often returned by Google Drive for large files without `gdown`.
"""

import torch
from torch.utils.data import DataLoader
import os
import numpy as np

# Import all necessary configurations, dataset, model, and utility functions
from config import (
    base_dir, image_size, G_LR, checkpoint_dir, generator_checkpoint_path,
    GOOGLE_DRIVE_FILE_ID, transform, fid_transform, device, time_embed_dim
)
from dataset import LungCTWithGaussianDataset
from model import CNF_UNet
from utils import download_file_from_google_drive
from generate import generate
from evaluate import evaluate_metrics
from visualize import (
    plot_pixel_distributions, plot_sample_images,
    plot_generated_samples, plot_real_vs_generated_side_by_side
)


def is_html_file(filepath: str) -> bool:
    """
    Checks if a file contains HTML content instead of expected binary weights.

    This utility function is used to detect common issues with Google Drive
    downloads where, instead of a binary file, an HTML warning page (e.g.,
    "Google Drive can't scan this file for viruses") is downloaded.

    Parameters
    ----------
    filepath : str
        The path to the file to be checked.

    Returns
    -------
    bool
        `True` if the file's header contains common HTML tags, indicating it's
        likely an HTML document; `False` otherwise.
    """
    try:
        with open(filepath, "rb") as f:
            # Read a small portion of the file to check for HTML header
            header = f.read(1024)
        return b"<!DOCTYPE html>" in header or b"<html" in header or b"<head>" in header
    except Exception as e:
        print(f"Warning: Could not read file {filepath} to check for HTML content: {e}")
        return False


def main() -> None:
    """
    Executes the main pipeline of the Synthetic Image Generator project.

    This function performs the following steps:
    1. Ensures the checkpoint directory exists.
    2. Initializes the dataset and data loaders for training and evaluation.
    3. Prints the total number of images loaded.
    4. Initializes the `CNF_UNet` generator model.
    5. Checks for and downloads pre-trained generator weights from Google Drive
       if they are not present locally. Includes robust error handling for
       download failures or corrupted (HTML) downloads.
    6. Loads the pre-trained weights into the generator model.
    7. Samples a batch of real and noise images for initial visualization.
    8. Generates a specified number of synthetic images using the trained generator.
    9. Evaluates the generated images against real images using MSE, PSNR, SSIM, and FID.
    10. Plots various visualizations, including pixel distributions, sample images,
        generated samples, and side-by-side comparisons of real vs. generated images.

    Raises
    ------
    SystemExit
        Exits the script if critical steps like weight download or loading fail,
        or if a downloaded file is detected as corrupted HTML.
    """
    # Ensure the checkpoint directory exists for saving/loading models.
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the dataset and data loaders.
    # `num_workers=0` is used due to potential issues with multiprocessing and GUI frameworks
    # or specific dataset loading methods (like pydicom's heavy resource usage).
    dataset = LungCTWithGaussianDataset(base_dir, transform=transform, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    eval_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    print(f"Total images loaded: {len(dataset)}")

    # Create the generator model instance and move it to the specified device (CPU/CUDA).
    generator = CNF_UNet(time_embed_dim=time_embed_dim).to(device)

    # --- Weight Management: Download and Load ---
    if not generator_checkpoint_path.exists():
        print(f"Pre-trained weights not found at {generator_checkpoint_path}. Attempting to download from Google Drive...")
        try:
            download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, generator_checkpoint_path)
            print("Downloaded weights to checkpoints/generator_final.pth")
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Please ensure the Google Drive link is correct and accessible.")
            # Exit if download fails, as model weights are crucial.
            return # Using return here instead of sys.exit() for cleaner testability
    else:
        print(f"Found existing weights at {generator_checkpoint_path}. Loading...")

    # Validate the downloaded .pth file to ensure it's not a Google Drive warning page.
    if is_html_file(generator_checkpoint_path):
        print("❌ Error: The downloaded .pth file is an HTML document. This typically means Google Drive returned a warning page instead of the actual file.")
        print("Please ensure the Google Drive ID is correct and accessible, and use `gdown` or equivalent proper download methods to get the actual PyTorch checkpoint.")
        # Exit if file is corrupted, as it cannot be loaded.
        return # Using return here instead of sys.exit() for cleaner testability

    # Load generator weights into the model.
    try:
        generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=device))
        print("✅ Generator weights loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading generator weights: {e}")
        print("Please ensure the weights match the model architecture and are in a valid PyTorch format.")
        # Exit if weight loading fails, as the model won't function without them.
        return # Using return here instead of sys.exit() for cleaner testability

    # --- Initial Visualization of Data ---
    # Sample one batch from the dataloader for initial visualization.
    for noise, image in dataloader:
        sample_noise_batch = noise.cpu() # Move to CPU for plotting
        sample_image_batch = image.cpu() # Move to CPU for plotting
        break # Take only the first batch

    plot_pixel_distributions(sample_image_batch, sample_noise_batch)
    plot_sample_images(sample_image_batch, sample_noise_batch)

    # --- Image Generation ---
    num_generated_samples = 16 # Number of synthetic images to generate
    generated_images = generate(generator, num_generated_samples, steps=200)

    # --- Evaluation of Metrics ---
    mse, psnr, ssim, fid = evaluate_metrics(
        generator,
        eval_dataloader,
        num_generated_samples=num_generated_samples,
        steps=200
    )

    # --- Further Visualizations ---
    # Collect a subset of real image pixels for distribution comparison with generated images.
    all_real_pixels = []
    # Limit collection to a few batches to keep it manageable.
    for i, (_, real_img_batch) in enumerate(eval_dataloader):
        all_real_pixels.append(real_img_batch.cpu().view(-1).numpy()) # Flatten and convert to NumPy
        if i >= 4:  # Collect from 5 batches (0 to 4)
            break
    all_real_pixels_flat = np.concatenate(all_real_pixels)

    # Flatten generated images for pixel distribution plot.
    flat_generated_images = generated_images.view(-1).numpy()
    plot_pixel_distributions(sample_image_batch, sample_noise_batch, generated_images_flat=flat_generated_images)

    # Plot a selection of the newly generated images.
    plot_generated_samples(generated_images)

    # Collect real images matching the number of generated samples for side-by-side comparison.
    real_images_for_viz = []
    collected_count_viz = 0
    for _, real_img_batch_viz in eval_dataloader:
        real_images_for_viz.append(real_img_batch_viz)
        collected_count_viz += real_img_batch_viz.shape[0]
        if collected_count_viz >= num_generated_samples:
            break
    # Concatenate and slice to ensure exactly `num_generated_samples` real images.
    real_images_batch_tensor_viz = torch.cat(real_images_for_viz, dim=0)[:num_generated_samples]

    # Plot real vs. generated images side-by-side for visual comparison.
    plot_real_vs_generated_side_by_side(real_images_batch_tensor_viz, generated_images)


if __name__ == "__main__":
    main()