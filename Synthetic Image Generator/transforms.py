# Synthetic Image Generator/transforms.py

import torchvision.transforms as T
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def get_transforms(image_size=(64, 64)):
    """
    Returns the image preprocessing transformations for model input.

    Args:
        image_size (tuple): Desired output image size (height, width).

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    logger.info(f"Setting up image transformation to resize to {image_size} and normalize to [-1, 1].")
    transform = T.Compose([
        T.ToPILImage(),               # Convert numpy array to PIL Image for transformations
        T.Resize(image_size),         # Resize to specified size
        T.ToTensor(),                 # Convert PIL Image to torch tensor (C x H x W), scales [0,255] to [0,1] for uint8, or [0,1] for float
        T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    return transform

def get_fid_transforms():
    """
    Returns the image transformations specifically for FID calculation.
    De-normalizes from [-1, 1] to [0, 1] and converts to PIL Image
    (which will be in [0, 255] for saving as PNG).

    Returns:
        torchvision.transforms.Compose: The FID transformation pipeline.
    """
    logger.info("Setting up FID transformation to de-normalize to [0, 1] and convert to PIL Image.")
    fid_transform = T.Compose([
        T.Normalize(mean=[-1.0], std=[2.0]), # De-normalize from [-1, 1] to [0, 1]
        T.ToPILImage(), # Converts to PIL Image, which will be in [0, 255] for saving as PNG
    ])
    return fid_transform