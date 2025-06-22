import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_images(images, title="Generated Images", num_images=8, cmap='gray', vmin=0, vmax=1):
    """
    Displays a grid of images using matplotlib.

    Args:
        images (torch.Tensor): A batch of images (B, C, H, W) or (B, H, W).
                               Expected to be in [-1, 1] range.
        title (str): Title for the entire plot.
        num_images (int): Maximum number of images to display.
        cmap (str): Colormap for imshow.
        vmin (float): Minimum value for colormap scaling.
        vmax (float): Maximum value for colormap scaling.
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(num_images, images.shape[0])):
        plt.subplot(2, 4, i + 1)
        # Convert to numpy and map from [-1, 1] to [0, 1] for display
        display_img = (images[i, 0].cpu().numpy() + 1) / 2
        plt.imshow(display_img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()