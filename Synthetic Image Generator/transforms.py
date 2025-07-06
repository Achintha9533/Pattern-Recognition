# Synthetic Image Generator/transforms.py

import torchvision.transforms as T
import logging
from typing import Tuple

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module defines the image preprocessing and post-processing transformation pipelines
used in the Synthetic Image Generator project. It provides functions to create
transformations for model input (resizing, normalization) and for FID calculation
(denormalization and conversion to a format suitable for saving).
"""

def get_transforms(image_size: Tuple[int, int] = (64, 64)) -> T.Compose:
    """
    Returns the image preprocessing transformations for model input.

    This pipeline prepares raw image data (e.g., NumPy arrays from DICOM)
    for consumption by the PyTorch model. It converts the image to a PIL Image,
    resizes it to the desired dimensions, converts it to a PyTorch tensor,
    and normalizes its pixel values to the range [-1, 1].

    Args:
        image_size (Tuple[int, int]): The desired output image size (height, width)
                                      after transformation. Defaults to (64, 64).

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline.
    """
    logger.info(f"Setting up image transformation to resize to {image_size} and normalize to [-1, 1].")
    transform: T.Compose = T.Compose([
        T.ToPILImage(),               # Converts a NumPy array (H, W) or (H, W, C) to a PIL Image.
                                      # Assumes input is float32 [0,1] from load_dicom_image.
        T.Resize(image_size),         # Resizes the PIL Image to the specified (height, width).
        T.ToTensor(),                 # Converts PIL Image to torch.Tensor (C x H x W).
                                      # Scales pixel values from [0, 255] to [0, 1] for uint8,
                                      # or keeps [0, 1] for float inputs.
        T.Normalize(mean=[0.5], std=[0.5])  # Normalizes the tensor from [0, 1] to [-1, 1].
                                            # Formula: (x - mean) / std.
    ])
    return transform

def get_fid_transforms() -> T.Compose:
    """
    Returns the image transformations specifically for Fréchet Inception Distance (FID) calculation.

    The `torch_fidelity` library, which is used for FID calculation, expects input images
    to be in the [0, 255] range and typically in PIL Image format (which it then converts
    internally). This pipeline denormalizes the model's output (which is in [-1, 1])
    back to [0, 1] and then converts it to a PIL Image, which implicitly scales to [0, 255]
    for image saving operations (e.g., as PNG).

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline suitable for FID.
    """
    logger.info("Setting up FID transformation to de-normalize to [0, 1] and convert to PIL Image.")
    fid_transform: T.Compose = T.Compose([
        # De-normalize from [-1, 1] back to [0, 1].
        # Formula: x * std + mean => x * 2.0 + (-1.0)
        T.Normalize(mean=[-1.0], std=[2.0]),
        T.ToPILImage(), # Converts torch.Tensor (C x H x W) with values in [0, 1] to PIL Image.
                        # For saving as PNG, PIL will typically interpret these as [0, 255].
    ])
    return fid_transform
