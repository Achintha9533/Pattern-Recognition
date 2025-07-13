"""
Evaluation metrics for image generation models.

This module provides functions to calculate various image quality metrics
for evaluating generative models, including Mean Squared Error (MSE),
Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index (SSIM),
and Frechet Inception Distance (FID). It integrates with `torch_fidelity`
for advanced metrics and handles image preprocessing for accurate calculations.

Functions
---------
save_images_to_temp_dir(images_tensor, path)
    Saves a batch of image tensors to a temporary directory as PNG files
    for use with external evaluation tools.

calculate_mse_psnr_ssim(real_images, generated_images)
    Calculates pixel-wise similarity metrics (MSE, PSNR, SSIM) between
    batches of real and generated images.

evaluate_metrics(generator, eval_dataloader, num_generated_samples, steps)
    Orchestrates the full evaluation process, generating samples, collecting
    real images, and computing all defined metrics.

Dependencies
------------
- `pytorch_msssim`: Required for SSIM calculation. A warning is issued if not found.
- `torch_fidelity`: Used for calculating Frechet Inception Distance (FID).

Notes
-----
- Images are assumed to be in the [-1, 1] range internally but are denormalized
  to [0, 1] or [0, 255] as required by specific metric calculations.
- Temporary directories are created and cleaned up for FID calculation.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch_fidelity import calculate_metrics
import numpy as np
import os
import shutil # For cleaning up temporary directories
import tempfile # For creating temporary directories
from tqdm import tqdm # For progress bars

# Import necessary modules from your project
from config import image_size, device, fid_transform # Assuming these are needed for evaluation
from generate import generate # Assuming generate function is in generate.py
from typing import Tuple, Union # Import Tuple and Union for type hinting

try:
    from pytorch_msssim import ssim
except ImportError:
    # Define a placeholder or set to None if the library is not installed.
    # This allows the module to load without crashing, but SSIM calculation will be skipped.
    ssim = None
    print("Warning: pytorch_msssim not found. SSIM calculation will be skipped. Please install it (`pip install pytorch-msssim`) for SSIM metric.")


def save_images_to_temp_dir(images_tensor: torch.Tensor, path: str) -> None:
    """
    Saves a batch of image tensors to a temporary directory as PNG files.

    This function denormalizes image tensors from the [-1, 1] range to [0, 1],
    converts them to PIL images, and saves them as PNG files in the specified path.
    This is typically used to prepare images for `torch_fidelity` which often
    expects images on disk.

    Parameters
    ----------
    images_tensor : torch.Tensor
        A batch of image tensors, expected to be in the range [-1, 1] and on CPU
        or moved to CPU within the function. Shape should be (N, C, H, W).
    path : str
        The path to the directory where the images will be saved.
        The directory will be created if it does not exist.
    """
    os.makedirs(path, exist_ok=True)
    to_pil = ToPILImage()
    # Denormalize images from [-1, 1] to [0, 1] before saving
    images_tensor = (images_tensor + 1) / 2.0
    for i, img_tensor in enumerate(images_tensor):
        # Ensure the tensor is on CPU and convert to PIL for saving
        img_pil = to_pil(img_tensor.cpu())
        img_pil.save(os.path.join(path, f"{i:04d}.png"))


def calculate_mse_psnr_ssim(
    real_images: torch.Tensor, generated_images: torch.Tensor
) -> Tuple[float, float, float]: # Changed to Tuple
    """
    Calculates Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR),
    and Structural Similarity Index (SSIM) between two batches of images.

    All input images are assumed to be already denormalized to the [0, 1] range.

    Parameters
    ----------
    real_images : torch.Tensor
        A batch of real images, expected in the range [0, 1].
        Shape: (N, C, H, W).
    generated_images : torch.Tensor
        A batch of generated images, expected in the range [0, 1].
        Shape: (N, C, H, W).

    Returns
    -------
    tuple of float
        A tuple containing:
        - mse_val : float
            The Mean Squared Error.
        - psnr_val : float
            The Peak Signal-to-Noise Ratio in dB. Returns `inf` if MSE is 0.
        - ssim_val : float
            The Structural Similarity Index. Returns -1.0 if `pytorch_msssim`
            is not installed or an error occurs during calculation.

    Notes
    -----
    - PSNR calculation assumes a maximum possible pixel value of 1.0 (for [0, 1] images).
    - SSIM requires the `pytorch_msssim` library. A warning is printed if it's not available.
    """

    # Calculate MSE
    mse_tensor = F.mse_loss(generated_images, real_images, reduction='mean')
    mse_val = mse_tensor.item() # Get the float value for reporting

    # PSNR calculation (assuming max_val=1 for [0,1] images)
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        # PSNR = 10 * log10(MAX_I^2 / MSE)
        # For images in [0, 1] range, MAX_I = 1. So PSNR = 10 * log10(1 / MSE)
        psnr_val = 10 * torch.log10(1 / mse_tensor).item()

    # SSIM (Structural Similarity Index)
    if ssim is not None: # Check if ssim was successfully imported
        try:
            # Ensure images are in [0, 1] and have shape (N, C, H, W) as required by pytorch_msssim
            ssim_val = ssim(generated_images, real_images, data_range=1.0, size_average=True).item()
        except Exception as e:
            ssim_val = -1.0
            print(f"Warning: Error calculating SSIM: {e}. SSIM not calculated.")
    else:
        ssim_val = -1.0
        print("Warning: pytorch_msssim was not available. SSIM not calculated.")

    return mse_val, psnr_val, ssim_val


def evaluate_metrics(
    generator: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    num_generated_samples: int,
    steps: int
) -> Tuple[float, float, float, float]: # Changed to Tuple
    """
    Evaluates the generator model using FID, MSE, PSNR, and SSIM.

    This function sets the generator to evaluation mode, collects a specified
    number of real images from the dataloader, generates an equivalent number
    of synthetic images, calculates pixel-wise metrics (MSE, PSNR, SSIM),
    and computes the Frechet Inception Distance (FID). Temporary directories
    are used for FID calculation and are cleaned up afterwards.

    Parameters
    ----------
    generator : torch.nn.Module
        The trained generator model (e.g., CNF_UNet instance) to be evaluated.
    eval_dataloader : torch.utils.data.DataLoader
        DataLoader providing batches of real images for comparison.
    num_generated_samples : int
        The total number of synthetic samples to generate for evaluation.
    steps : int
        The number of steps to use for the image generation process (e.g., in a diffusion model).

    Returns
    -------
    tuple of float
        A tuple containing the calculated metrics: (MSE, PSNR, SSIM, FID).
        FID will be `float('nan')` if an error occurs during its calculation.

    Notes
    -----
    - The generator is set to evaluation mode (`generator.eval()`) during this process.
    - Images are internally denormalized to the [0, 1] range before metric calculations
      to ensure consistency.
    - Temporary directories for FID calculation (`temp_real_fid_images`, `temp_gen_fid_images`)
      are created and automatically removed upon completion or error.
    - FID calculation leverages the `torch_fidelity` library, which performs its own
      internal preprocessing (e.g., resizing to 299x299 for InceptionV3).
    """
    generator.eval() # Ensure model is in evaluation mode

    real_images_collector = []

    # Collect real images from eval_dataloader up to num_generated_samples
    collected_count = 0
    dataloader_iter = tqdm(eval_dataloader, desc="Collecting real images") # Added tqdm here

    for _, real_img_batch in dataloader_iter:
        real_images_collector.append(real_img_batch.to(device)) # Move to device
        collected_count += real_img_batch.shape[0]
        if collected_count >= num_generated_samples:
            break
    real_images_batch_tensor = torch.cat(real_images_collector, dim=0)[:num_generated_samples]

    # Generate synthetic images
    generated_images = generate(generator, num_generated_samples, steps=steps)

    # Denormalize images from [-1, 1] to [0, 1] for pixel-wise metric calculations.
    real_images_for_metrics = (real_images_batch_tensor + 1) / 2.0
    generated_images_for_metrics = (generated_images + 1) / 2.0

    # Calculate MSE, PSNR, SSIM
    mse, psnr, ssim_val = calculate_mse_psnr_ssim(real_images_for_metrics, generated_images_for_metrics)

    # For FID, torch_fidelity often prefers saving images to disk for consistent pre-processing.
    # Using tempfile.mkdtemp() for robust temporary directory creation and management
    temp_real_fid_dir = tempfile.mkdtemp()
    temp_gen_fid_dir = tempfile.mkdtemp()

    fid_value = float('nan') # Initialize fid_value to NaN in case of error

    try:
        save_images_to_temp_dir(real_images_for_metrics, temp_real_fid_dir)
        save_images_to_temp_dir(generated_images_for_metrics, temp_gen_fid_dir)

        # Call torch_fidelity to calculate FID
        metrics = calculate_metrics(
            input1=temp_real_fid_dir,
            input2=temp_gen_fid_dir,
            cuda=torch.cuda.is_available(), # Use CUDA if available
            isc=False,
            fid=True, # Ensure FID is enabled
            kid=False,
            lpips=False,
            verbose=False,
            # Pass the transform from config.py if it's meant for fidelity calculations
            input1_transform=fid_transform if fid_transform else None,
            input2_transform=fid_transform if fid_transform else None
        )

        fid_value = metrics.get('frechet_inception_distance')
        if fid_value is None:
            print("Warning: 'frechet_inception_distance' key not found in torch_fidelity output.")
            print(f"Full metrics dictionary: {metrics}") # Print full dict for debugging if key is missing
            fid_value = float('nan') # Indicate failure
        else:
            print(f"Frechet Inception Distance: {fid_value:.2f}")

    except Exception as e:
        print(f"Error during FID calculation: {e}")
        fid_value = float('nan') # Return NaN if FID calculation fails
    finally:
        # Clean up temporary directories
        if os.path.exists(temp_real_fid_dir):
            shutil.rmtree(temp_real_fid_dir)
        if os.path.exists(temp_gen_fid_dir):
            shutil.rmtree(temp_gen_fid_dir)

    # Crucial: Return all calculated metrics
    return mse, psnr, ssim_val, fid_value