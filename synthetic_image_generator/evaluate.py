# Synthetic Image Generator/evaluate.py

import torch
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torch_fidelity import calculate_metrics
from pathlib import Path
import shutil
from tqdm import tqdm
import logging
import torchvision.transforms as T # Imported for type hinting clarity

# Configure logging for this module
logger = logging.getLogger(__name__)

def evaluate_model(
    real_images_batch_tensor: torch.Tensor,
    generated_images: torch.Tensor,
    fid_transform: T.Compose,
    num_compare: int
) -> None:
    """
    Calculates and logs various image quality metrics (MSE, PSNR, SSIM, and FID)
    to assess the fidelity and diversity of generated images compared to real ones.

    Args:
        real_images_batch_tensor (torch.Tensor): A batch of real images (CPU tensor)
                                                 to be used as ground truth.
                                                 Expected pixel range: [-1, 1].
        generated_images (torch.Tensor): A batch of generated images (CPU tensor)
                                         from the model. Expected pixel range: [-1, 1].
        fid_transform (T.Compose): A torchvision.transforms.Compose object used to
                                   prepare images for FID calculation (e.g., denormalize
                                   to [0, 255] and convert to PIL Image).
        num_compare (int): The number of image pairs (real vs. generated) to use
                           for calculating pixel-wise metrics (MSE, PSNR, SSIM)
                           and for saving images for FID calculation.
    """
    logger.info("\n--- Evaluating Generated Image Fidelity ---")

    if real_images_batch_tensor.numel() == 0:
        logger.error("Could not retrieve enough real images for evaluation. Skipping pixel-wise metrics.")
        return

    # Ensure we compare the minimum of available real/generated images and the requested num_compare
    num_effective_compare: int = min(num_compare, real_images_batch_tensor.shape[0], generated_images.shape[0])
    if num_effective_compare == 0:
        logger.error("No images available for comparison after checking effective count. Skipping all metrics.")
        return

    # Convert tensors to numpy arrays and de-normalize to [0, 1] for pixel-wise metric calculations.
    # The metrics (MSE, PSNR, SSIM) typically expect input images in the [0, 1] or [0, 255] range.
    # Assuming images are normalized to [-1, 1], so (x + 1) / 2 brings them to [0, 1].
    real_images_np_01: np.ndarray = ((real_images_batch_tensor[:num_effective_compare].squeeze().numpy() + 1) / 2)
    generated_images_np_01: np.ndarray = ((generated_images[:num_effective_compare].squeeze().numpy() + 1) / 2)

    # Initialize lists to store metric values for averaging
    mses: list[float] = []
    psnrs: list[float] = []
    ssims: list[float] = []

    # Calculate pixel-wise metrics for each image pair
    for i in range(num_effective_compare):
        real_img: np.ndarray = real_images_np_01[i]
        gen_img: np.ndarray = generated_images_np_01[i]

        # Calculate Mean Squared Error (MSE)
        mse: float = mean_squared_error(real_img, gen_img)
        mses.append(mse)

        # Calculate Peak Signal-to-Noise Ratio (PSNR)
        # data_range=1 because images are normalized to [0, 1] for PSNR calculation.
        psnr: float = peak_signal_noise_ratio(real_img, gen_img, data_range=1)
        psnrs.append(psnr)

        # Calculate Structural Similarity Index Measure (SSIM)
        # data_range=1 because images are normalized to [0, 1] for SSIM calculation.
        ssim: float = structural_similarity(real_img, gen_img, data_range=1)
        ssims.append(ssim)

    logger.info(f"Average MSE: {np.mean(mses):.6f}")
    logger.info(f"Average PSNR: {np.mean(psnrs):.6f} dB")
    logger.info(f"Average SSIM: {np.mean(ssims):.6f}")

    # === Fréchet Inception Distance (FID) Calculation ===
    logger.info("\n--- Calculating Fréchet Inception Distance (FID) ---")

    # Create temporary directories for real and generated images required by torch_fidelity.
    real_images_dir: Path = Path("./fid_real_images")
    generated_images_dir: Path = Path("./fid_generated_images")

    # Ensure directories are clean before saving new images for FID.
    try:
        if real_images_dir.exists():
            shutil.rmtree(real_images_dir)
        if generated_images_dir.exists():
            shutil.rmtree(generated_images_dir)
    except OSError as e:
        logger.error(f"Error cleaning up temporary FID directories: {e}. Please manually delete '{real_images_dir}' and '{generated_images_dir}'.")
        return # Exit if cleanup fails, as subsequent operations might be affected.

    real_images_dir.mkdir(exist_ok=False) # Should be new, so exist_ok=False
    generated_images_dir.mkdir(exist_ok=False)

    try:
        # Save real images to the temporary directory.
        logger.info(f"Saving {num_effective_compare} real images to {real_images_dir} for FID calculation...")
        for i in tqdm(range(num_effective_compare), desc="Saving real images"):
            # Apply FID transform (denormalizes to [0, 255] and converts to PIL Image).
            img_to_save = fid_transform(real_images_batch_tensor[i])
            # Convert grayscale (L mode) to RGB if Inception v3 model expects 3 channels.
            if img_to_save.mode == 'L':
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(real_images_dir / f"real_{i:04d}.png")

        # Save generated images to the temporary directory.
        logger.info(f"Saving {num_effective_compare} generated images to {generated_images_dir} for FID calculation...")
        for i in tqdm(range(num_effective_compare), desc="Saving generated images"):
            img_to_save = fid_transform(generated_images[i])
            if img_to_save.mode == 'L':
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(generated_images_dir / f"gen_{i:04d}.png")

        # Perform FID calculation using torch_fidelity.
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
        # Log detailed error information if FID calculation fails.
        logger.error(f"An error occurred during FID calculation: {e}")
        logger.error("Please ensure 'torch_fidelity' is installed (`pip install torch_fidelity`), "
                     "that you have the necessary torchvision/Pillow dependencies for image saving, "
                     "and that Inception V3 model weights can be downloaded (torch_fidelity handles this usually).")
    finally:
        # Always attempt to clean up temporary directories, regardless of success or failure.
        try:
            if real_images_dir.exists():
                shutil.rmtree(real_images_dir)
            if generated_images_dir.exists():
                shutil.rmtree(generated_images_dir)
            logger.info("Temporary FID image directories cleaned up.")
        except OSError as e:
            logger.error(f"Error during final cleanup of temporary FID directories: {e}. Manual deletion may be required.")
