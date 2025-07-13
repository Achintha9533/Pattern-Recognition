import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from torch_fidelity import calculate_metrics
import numpy as np
import os
import shutil # For cleaning up temporary directories

# Import necessary modules from your project
from config import image_size, device, fid_transform # Assuming these are needed for evaluation

# --- FIX START ---
# Move ssim and generate imports to the top level for easier mocking in tests
# and consistent module-level availability.

try:
    from pytorch_msssim import ssim
except ImportError:
    # Define a placeholder or set to None if the library is not installed.
    # This allows the module to load without crashing, but SSIM calculation will be skipped.
    ssim = None
    print("Warning: pytorch_msssim not found. SSIM calculation will be skipped. Please install it (`pip install pytorch-msssim`) for SSIM metric.")

from generate import generate
# --- FIX END ---


# Helper to save tensors to temporary directory for torch_fidelity
def save_images_to_temp_dir(images_tensor, path):
    os.makedirs(path, exist_ok=True)
    to_pil = ToPILImage()
    # Denormalize images from [-1, 1] to [0, 1] before saving
    images_tensor = (images_tensor + 1) / 2.0
    for i, img_tensor in enumerate(images_tensor):
        # Ensure the tensor is on CPU and convert to PIL for saving
        img_pil = to_pil(img_tensor.cpu()) 
        img_pil.save(os.path.join(path, f"{i:04d}.png"))


def calculate_mse_psnr_ssim(real_images, generated_images):
    """
    Calculates Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR),
    and Structural Similarity Index (SSIM) between two batches of images.

    Assumes images are already denormalized to the [0, 1] range.
    """
    
    # Calculate MSE
    mse_tensor = F.mse_loss(generated_images, real_images, reduction='mean') # Use reduction='mean' explicitly for clarity
    mse_val = mse_tensor.item() # Get the float value for reporting

    # PSNR calculation (assuming max_val=1 for [0,1] images)
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        # PSNR = 10 * log10(MAX_I^2 / MSE)
        # For images in [0, 1] range, MAX_I = 1. So PSNR = 10 * log10(1 / MSE)
        psnr_val = 10 * torch.log10(1 / mse_tensor).item() 

    # SSIM (Structural Similarity Index) - now 'ssim' is imported at the top
    # --- FIX START ---
    if ssim is not None: # Check if ssim was successfully imported
        try:
            # Ensure images are in [0, 1] and have shape (N, C, H, W) as required by pytorch_msssim
            ssim_val = ssim(generated_images, real_images, data_range=1.0, size_average=True).item() # size_average=True for scalar
        except Exception as e:
            ssim_val = -1.0
            print(f"Warning: Error calculating SSIM: {e}. SSIM not calculated.")
    else:
        ssim_val = -1.0
        print("Warning: pytorch_msssim was not available. SSIM not calculated.")
    # --- FIX END ---

    return mse_val, psnr_val, ssim_val

def evaluate_metrics(generator, eval_dataloader, num_generated_samples: int, steps: int):
    """
    Evaluates the generator model using FID, MSE, PSNR, and SSIM.

    Parameters
    ----------
    generator : torch.nn.Module
        The trained generator model (e.g., CNF_UNet instance).
    eval_dataloader : torch.utils.data.DataLoader
        DataLoader for real images used for comparison.
    num_generated_samples : int
        Number of samples to generate for evaluation.
    steps : int
        Number of steps for the generation process.

    Returns
    -------
    tuple
        (MSE, PSNR, SSIM, FID) values.
    """
    generator.eval() # Ensure model is in evaluation mode

    real_images_collector = []
    
    # Collect real images from eval_dataloader up to num_generated_samples
    collected_count = 0
    # Use tqdm for progress bar if running long evaluations
    # try:
    #     from tqdm import tqdm
    #     dataloader_iter = tqdm(eval_dataloader, desc="Collecting real images")
    # except ImportError:
    dataloader_iter = eval_dataloader

    for _, real_img_batch in dataloader_iter:
        real_images_collector.append(real_img_batch.to(device)) # Move to device
        collected_count += real_img_batch.shape[0]
        if collected_count >= num_generated_samples:
            break
    real_images_batch_tensor = torch.cat(real_images_collector, dim=0)[:num_generated_samples]

    # Generate synthetic images (assuming generate function is imported)
    # --- FIX START ---
    # 'generate' is now imported at the top of the file.
    generated_images = generate(generator, num_generated_samples, steps=steps)
    # --- FIX END ---

    # Denormalize images from [-1, 1] to [0, 1] for metric calculations.
    # FID (torch_fidelity) typically expects [0, 255] or [0, 1].
    real_images_for_metrics = (real_images_batch_tensor + 1) / 2.0 
    generated_images_for_metrics = (generated_images + 1) / 2.0

    # Calculate MSE, PSNR, SSIM
    mse, psnr, ssim_val = calculate_mse_psnr_ssim(real_images_for_metrics, generated_images_for_metrics)

    # For FID, torch_fidelity often prefers saving images to disk for consistent pre-processing.
    real_fid_dir = "temp_real_fid_images"
    gen_fid_dir = "temp_gen_fid_images"

    try:
        # Note: fid_transform from config is imported but not explicitly applied here.
        # torch_fidelity's calculate_metrics function handles its own internal
        # preprocessing (e.g., resizing to 299x299 for InceptionV3 and normalization)
        # when given directory paths. If 'fid_transform' contains additional
        # domain-specific transformations needed before saving to disk, you should
        # apply it before calling `save_images_to_temp_dir`. Otherwise, its non-usage is expected.
        save_images_to_temp_dir(real_images_for_metrics, real_fid_dir)
        save_images_to_temp_dir(generated_images_for_metrics, gen_fid_dir)

        # Call torch_fidelity to calculate FID
        metrics = calculate_metrics(
            input1=real_fid_dir,
            input2=gen_fid_dir,
            cuda=torch.cuda.is_available(), # Use CUDA if available
            isc=False,
            fid=True, # Ensure FID is enabled
            kid=False,
            lpips=False,
            verbose=False
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
        if os.path.exists(real_fid_dir):
            shutil.rmtree(real_fid_dir)
        if os.path.exists(gen_fid_dir):
            shutil.rmtree(gen_fid_dir)
        # print("Temporary FID image directories cleaned up.") # Often too verbose

    print(f"\nEvaluation Results:")
    print(f"  MSE: {mse:.6f}") # Increased precision for MSE
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim_val:.6f}") # Increased precision for SSIM
    print(f"  FID: {fid_value:.2f}")

    return mse, psnr, ssim_val, fid_value