# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_pixel_distributions(sample_real_batch, sample_noise_batch, generated_images_flat=None):
    plt.figure(figsize=(12, 5))
    
    flat_real = sample_real_batch.view(-1).numpy()
    flat_noise = sample_noise_batch.view(-1).numpy()

    plt.subplot(1, 2, 1)
    plt.hist(flat_real, bins=50, color='blue', alpha=0.7)
    plt.title('Real CT Image Pixel Distribution (Sample)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(flat_noise, bins=50, color='red', alpha=0.7)
    plt.title('Initial Gaussian Noise Distribution (Sample)')
    plt.xlabel('Pixel Value')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    if generated_images_flat is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(flat_real, bins=50, color='blue', alpha=0.6, label='Real CT Image Pixel Distribution (Sampled)')
        plt.hist(generated_images_flat, bins=50, color='green', alpha=0.6, label='Generated Image Pixel Distribution')
        plt.title('Comparison of Pixel Distributions: Real vs. Generated')
        plt.xlabel('Pixel Value (Normalized [-1, 1])')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.show()


def plot_sample_images(sample_real_batch, sample_noise_batch):
    plt.figure(figsize=(10, 4))
    num_display = min(4, sample_real_batch.shape[0])
    for i in range(num_display):
        plt.subplot(2, num_display, i + 1)
        plt.imshow(sample_real_batch[i, 0], cmap='gray')
        plt.title("Real CT Image")
        plt.axis('off')
        plt.subplot(2, num_display, i + num_display + 1)
        plt.imshow(sample_noise_batch[i, 0], cmap='gray')
        plt.title("Initial Noise")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_generated_samples(generated_images, num_display=16):
    plt.figure(figsize=(10, 8))
    num_display = min(num_display, generated_images.shape[0])
    for i in range(num_display):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, 0], cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_real_vs_generated_side_by_side(real_images_batch_tensor, generated_images, num_side_by_side=4):
    if real_images_batch_tensor.numel() > 0:
        plt.figure(figsize=(12, 6))
        num_compare = min(num_side_by_side, real_images_batch_tensor.shape[0], generated_images.shape[0])
        for i in range(num_compare):
            plt.subplot(2, num_side_by_side, i + 1)
            plt.imshow(real_images_batch_tensor[i, 0].cpu().numpy(), cmap='gray')
            plt.title(f"Real {i+1}")
            plt.axis('off')

            plt.subplot(2, num_side_by_side, i + num_side_by_side + 1)
            plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
            plt.title(f"Generated {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.suptitle("Real vs. Generated Images (Side-by-Side)", y=1.02)
        plt.show()