import torch
from torch.utils.data import DataLoader
import os
import numpy as np

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


def main() -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = LungCTWithGaussianDataset(base_dir, transform=transform, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    eval_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

    print(f"Total images loaded: {len(dataset)}")

    generator = CNF_UNet(time_embed_dim=time_embed_dim).to(device)

    if not generator_checkpoint_path.exists():
        print(f"Pre-trained weights not found at {generator_checkpoint_path}. Attempting to download from Google Drive...")
        try:
            download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, generator_checkpoint_path)
            print("Downloaded weights to checkpoints/generator_final.pth") # Added confirmation print
        except Exception as e:
            print(f"Error downloading weights: {e}")
            print("Please ensure the Google Drive link is correct and accessible.")
            return
    else:
        print(f"Found existing weights at {generator_checkpoint_path}. Loading...")

    try:
        # FIX: Add weights_only=False for compatibility with PyTorch 2.6+
        generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=device, weights_only=False))
        print("Generator weights loaded successfully.")
    except Exception as e:
        print(f"Error loading generator weights: {e}")
        print("Please check if the downloaded file is a valid PyTorch state_dict and matches the model architecture.")
        return

    sample_noise_batch = None
    sample_image_batch = None
    for noise, image in dataloader:
        sample_noise_batch = noise.cpu()
        sample_image_batch = image.cpu()
        break

    plot_pixel_distributions(sample_image_batch, sample_noise_batch)
    plot_sample_images(sample_image_batch, sample_noise_batch)

    num_generated_samples = 16
    generated_images = generate(generator, num_generated_samples, steps=200)

    mse, psnr, ssim, fid = evaluate_metrics(generator, eval_dataloader, num_generated_samples=num_generated_samples, steps=200)

    all_real_pixels = []
    num_batches_to_sample = 5
    for i, (_, real_img_batch) in enumerate(eval_dataloader):
        all_real_pixels.append(real_img_batch.cpu().view(-1).numpy())
        if i >= num_batches_to_sample - 1:
            break
    all_real_pixels_flat = np.concatenate(all_real_pixels)

    flat_generated_images = generated_images.view(-1).numpy()
    plot_pixel_distributions(sample_image_batch, sample_noise_batch, generated_images_flat=flat_generated_images)

    plot_generated_samples(generated_images)

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