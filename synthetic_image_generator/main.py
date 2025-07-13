# main.py
import torch
from torch.utils.data import DataLoader
import os
import numpy as np # Ensure numpy is imported for plotting functions

from config import (
    base_dir, image_size, G_LR, checkpoint_dir, generator_checkpoint_path,
    GOOGLE_DRIVE_FILE_ID, transform, fid_transform, device, time_embed_dim
)
from dataset import LungCTWithGaussianDataset
from model import CNF_UNet
from utils import download_file_from_google_drive # Only import download function from utils
from generate import generate # <--- CRUCIAL CHANGE: Import 'generate' from 'generate.py'
from evaluate import evaluate_metrics
from visualize import (
    plot_pixel_distributions, plot_sample_images, 
    plot_generated_samples, plot_real_vs_generated_side_by_side
)

def main():
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate dataset and dataloader
    dataset = LungCTWithGaussianDataset(base_dir, transform=transform, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    eval_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True) # For evaluation

    print(f"Total images loaded: {len(dataset)}")

    # Generator (CNF_UNet)
    generator = CNF_UNet(time_embed_dim=time_embed_dim).to(device)

    # === Download and Load Pre-trained Weights ===
    if not generator_checkpoint_path.exists():
        print(f"Pre-trained weights not found at {generator_checkpoint_path}. Attempting to download from Google Drive...")
        try:
            download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, generator_checkpoint_path)
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Please ensure the Google Drive link is correct and accessible.")
            return # Exit if weights can't be downloaded
    else:
        print(f"Found existing weights at {generator_checkpoint_path}. Loading...")

    # Load the state dictionary
    try:
        # Load weights, mapping to CPU if CUDA is not available
        generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=device))
        print("Generator weights loaded successfully.")
    except Exception as e:
        print(f"Error loading generator weights: {e}")
        print("Please check if the downloaded file is a valid PyTorch state_dict and matches the model architecture.")
        return # Exit if weights can't be loaded

    # === Initial Data Distribution Plot (using a sample batch) ===
    sample_noise_batch = None
    sample_image_batch = None
    for noise, image in dataloader:
        sample_noise_batch = noise.cpu()
        sample_image_batch = image.cpu()
        break

    plot_pixel_distributions(sample_image_batch, sample_noise_batch)
    plot_sample_images(sample_image_batch, sample_noise_batch)

    # === Sample generation ===
    num_generated_samples = 16
    # <--- CRUCIAL CHANGE: Call 'generate' from 'generate.py'
    generated_images = generate(generator, num_generated_samples, steps=200) 
    
    # === Evaluation Metrics ===
    mse, psnr, ssim, fid = evaluate_metrics(generator, eval_dataloader, num_generated_samples=num_generated_samples, steps=200)

    # === Distribution Plot for Generated Images vs. Real Images ===
    all_real_pixels = []
    num_batches_to_sample = 5
    for i, (_, real_img_batch) in enumerate(eval_dataloader): 
        all_real_pixels.append(real_img_batch.cpu().view(-1).numpy())
        if i >= num_batches_to_sample - 1:
            break
    all_real_pixels_flat = np.concatenate(all_real_pixels)
    
    flat_generated_images = generated_images.view(-1).numpy()
    plot_pixel_distributions(sample_image_batch, sample_noise_batch, generated_images_flat=flat_generated_images)


    # === Visualize sample generated images ===
    plot_generated_samples(generated_images)

    # === Visualize Real vs. Generated Side-by-Side ===
    # Re-collect real images for side-by-side visualization to ensure they match generated batch size
    real_images_for_viz = []
    collected_count_viz = 0
    for _, real_img_batch_viz in eval_dataloader:
        real_images_for_viz.append(real_img_batch_viz)
        collected_count_viz += real_img_batch_viz.shape[0]
        if collected_count_viz >= num_generated_samples:
            break
    real_images_batch_tensor_viz = torch.cat(real_images_for_viz, dim=0)[:num_generated_samples]

    plot_real_vs_generated_side_by_side(real_images_batch_tensor_viz, generated_images)

if __name__ == "__main__":
    main()