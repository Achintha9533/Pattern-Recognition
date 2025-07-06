# Synthetic Image Generator/evaluate.py

import torch
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torch_fidelity import calculate_metrics
from pathlib import Path
import shutil
from tqdm import tqdm
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def evaluate_model(real_images_batch_tensor, generated_images, fid_transform, num_compare):
    """
    Calculates and prints image quality metrics (MSE, PSNR, SSIM, FID).

    Args:
        real_images_batch_tensor (torch.Tensor): Batch of real images (CPU tensor).
        generated_images (torch.Tensor): Batch of generated images (CPU tensor).
        fid_transform (torchvision.transforms.Compose): Transform for FID calculation.
        num_compare (int): Number of images to compare for metrics.
    """
    logger.info("\n--- Evaluating Generated Image Fidelity ---")

    if real_images_batch_tensor.numel() == 0:
        logger.error("Could not retrieve enough real images for evaluation. Skipping pixel-wise metrics.")
        return

    # Ensure we compare the same number of images
    num_compare = min(num_compare, real_images_batch_tensor.shape[0])

    # Convert tensors to numpy arrays and de-normalize to [0, 1] for metric calculations
    # Assuming images are normalized to [-1, 1], so (x + 1) / 2 brings them to [0, 1]
    real_images_np_01 = ((real_images_batch_tensor[:num_compare].squeeze().numpy() + 1) / 2)
    generated_images_np_01 = ((generated_images[:num_compare].squeeze().numpy() + 1) / 2)

    # Initialize lists to store metric values
    mses, psnrs, ssims = [], [], []

    for i in range(num_compare):
        real_img = real_images_np_01[i]
        gen_img = generated_images_np_01[i]

        # Calculate MSE
        mse = mean_squared_error(real_img, gen_img)
        mses.append(mse)

        # Calculate PSNR
        # data_range=1 because images are normalized to [0, 1]
        psnr = peak_signal_noise_ratio(real_img, gen_img, data_range=1)
        psnrs.append(psnr)

        # Calculate SSIM
        ssim = structural_similarity(real_img, gen_img, data_range=1)
        ssims.append(ssim)

    logger.info(f"Average MSE: {np.mean(mses):.6f}")
    logger.info(f"Average PSNR: {np.mean(psnrs):.6f} dB")
    logger.info(f"Average SSIM: {np.mean(ssims):.6f}")

    # === FID Calculation ===
    logger.info("\n--- Calculating Fréchet Inception Distance (FID) ---")

    # Create temporary directories for real and generated images
    real_images_dir = Path("./fid_real_images")
    generated_images_dir = Path("./fid_generated_images")

    # Ensure directories are clean before saving
    if real_images_dir.exists():
        shutil.rmtree(real_images_dir)
    if generated_images_dir.exists():
        shutil.rmtree(generated_images_dir)

    real_images_dir.mkdir(exist_ok=True)
    generated_images_dir.mkdir(exist_ok=True)

    try:
        # Save real images
        logger.info(f"Saving {num_compare} real images to {real_images_dir} for FID calculation...")
        for i in tqdm(range(num_compare), desc="Saving real images"):
            img_to_save = fid_transform(real_images_batch_tensor[i])
            # Ensure it's a 3-channel image if Inception v3 expects it, even for grayscale.
            if img_to_save.mode == 'L': # If grayscale (luminance)
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(real_images_dir / f"real_{i:04d}.png")

        # Save generated images
        logger.info(f"Saving {num_compare} generated images to {generated_images_dir} for FID calculation...")
        for i in tqdm(range(num_compare), desc="Saving generated images"):
            img_to_save = fid_transform(generated_images[i])
            if img_to_save.mode == 'L': # If grayscale
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(generated_images_dir / f"gen_{i:04d}.png")

        # Perform FID calculation
        logger.info("Initiating FID calculation (this may download InceptionV3 weights if not cached)...")
        metrics = calculate_metrics(
            input1=str(real_images_dir),
            input2=str(generated_images_dir),
            cuda=torch.cuda.is_available(), # Use CUDA if available
            fid=True,
            verbose=False, # Set to True for more detailed output from torch_fidelity
        )

        if 'fid' in metrics:
            logger.info(f"FID: {metrics['fid']:.4f}")
        elif 'frechet_inception_distance' in metrics: # Handle potential alternative key
            logger.info(f"FID: {metrics['frechet_inception_distance']:.4f}")
        else:
            logger.error("FID value not found in metrics. This might indicate an internal error in torch_fidelity.")
            logger.debug(f"Full metrics dictionary: {metrics}")

    except Exception as e:
        logger.error(f"An error occurred during FID calculation: {e}")
        logger.error("Please ensure 'torch_fidelity' is installed (`pip install torch_fidelity`) "
                     "and that you have the necessary torchvision/Pillow dependencies for image saving."
                     "Also, check if Inception V3 model weights are downloaded (torch_fidelity handles this usually).")
    finally:
        # Clean up temporary directories
        if real_images_dir.exists():
            shutil.rmtree(real_images_dir)
        if generated_images_dir.exists():
            shutil.rmtree(generated_images_dir)
        logger.info("Temporary FID image directories cleaned up.")