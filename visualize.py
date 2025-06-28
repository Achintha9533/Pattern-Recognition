import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def plot_initial_distributions(sample_noise_batch, sample_image_batch):
    """
    Plots the pixel distributions of real CT images and initial Gaussian noise.
    """
    flat_noise = sample_noise_batch.view(-1).numpy()
    flat_image = sample_image_batch.view(-1).numpy()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(flat_image, bins=50, color='blue', alpha=0.7)
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

def plot_initial_samples(sample_noise_batch, sample_image_batch):
    """
    Displays sample real CT images and corresponding initial noise.
    """
    plt.figure(figsize=(10, 4))
    for i in range(min(4, sample_image_batch.shape[0])):
        plt.subplot(2, 4, i + 1)
        plt.imshow(sample_image_batch[i, 0], cmap='gray')
        plt.title("Real CT Image")
        plt.axis('off')
        plt.subplot(2, 4, i + 5)
        plt.imshow(sample_noise_batch[i, 0], cmap='gray')
        plt.title("Initial Noise")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_losses(training_losses):
    """
    Plots the generator and discriminator losses over training epochs.
    """
    plt.figure(figsize=(14, 6))

    # Generator Losses
    plt.subplot(1, 2, 1)
    plt.plot(training_losses['gen_flow_losses'], label='Generator Flow Matching Loss', color='blue')
    plt.plot(training_losses['gen_gan_losses'], label='Generator GAN Loss', color='cyan', linestyle='--')
    plt.title('Generator Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Discriminator Losses
    plt.subplot(1, 2, 2)
    plt.plot(training_losses['disc_real_losses'], label='Discriminator Real Loss', color='green')
    plt.plot(training_losses['disc_fake_losses'], label='Discriminator Fake Loss', color='orange')
    plt.plot(training_losses['disc_total_losses'], label='Discriminator Total Loss', color='red', linestyle='-.')
    plt.title('Discriminator Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_and_print_metrics(real_images_batch, generated_images, num_samples_to_compare, image_size):
    """
    Calculates and prints MSE, PSNR, and SSIM between real and generated images.
    """
    print("\n--- Evaluating Generated Image Fidelity ---")

    if real_images_batch is None:
        print("Could not retrieve real images batch for evaluation.")
        return

    num_compare = min(num_samples_to_compare, real_images_batch.shape[0])
    
    # Convert tensors to numpy arrays and de-normalize to [0, 1] for metric calculations
    real_images_np = ((real_images_batch[:num_compare].squeeze().numpy() + 1) / 2)
    generated_images_np = ((generated_images[:num_compare].squeeze().numpy() + 1) / 2)

    mses, psnrs, ssims = [], [], []

    for i in range(num_compare):
        real_img = real_images_np[i]
        gen_img = generated_images_np[i]

        mse = mean_squared_error(real_img, gen_img)
        mses.append(mse)

        psnr = peak_signal_noise_ratio(real_img, gen_img, data_range=1)
        psnrs.append(psnr)

        ssim = structural_similarity(real_img, gen_img, data_range=1)
        ssims.append(ssim)

    print(f"Average MSE: {np.mean(mses):.6f}")
    print(f"Average PSNR: {np.mean(psnrs):.6f} dB")
    print(f"Average SSIM: {np.mean(ssims):.6f}")

def plot_pixel_distributions_comparison(all_real_pixels_flat, flat_generated_images):
    """
    Plots a histogram comparing pixel distributions of real vs. generated images.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(all_real_pixels_flat, bins=50, color='blue', alpha=0.6, label='Real CT Image Pixel Distribution (Sampled)')
    plt.hist(flat_generated_images, bins=50, color='green', alpha=0.6, label='Generated Image Pixel Distribution')
    plt.title('Comparison of Pixel Distributions: Real vs. Generated')
    plt.xlabel('Pixel Value (Normalized [-1, 1])')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def plot_generated_samples(generated_images, num_display=16):
    """
    Displays a grid of sample generated images.
    """
    plt.figure(figsize=(10, 8))
    num_display = min(num_display, generated_images.shape[0])
    for i in range(num_display):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, 0], cmap='gray')
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_real_vs_generated_side_by_side(real_images_batch, generated_images, num_side_by_side=4):
    """
    Displays real and generated images side-by-side for direct comparison.
    """
    if real_images_batch is not None:
        plt.figure(figsize=(12, 6))
        num_to_display = min(num_side_by_side, real_images_batch.shape[0], generated_images.shape[0])
        for i in range(num_to_display):
            # Real Image
            plt.subplot(2, num_to_display, i + 1)
            plt.imshow(real_images_batch[i, 0].cpu().numpy(), cmap='gray')
            plt.title(f"Real {i+1}")
            plt.axis('off')

            # Generated Image
            plt.subplot(2, num_to_display, i + num_to_display + 1)
            plt.imshow(generated_images[i, 0].cpu().numpy(), cmap='gray')
            plt.title(f"Generated {i+1}")
            plt.axis('off')
        plt.tight_layout()
        plt.suptitle("Real vs. Generated Images (Side-by-Side)", y=1.02)
        plt.show()