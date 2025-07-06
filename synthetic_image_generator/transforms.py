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

These transformations are crucial for ensuring that image data is in the correct
format, size, and pixel value range for neural network processing and evaluation metrics.
"""

def get_transforms(image_size: Tuple[int, int] = (64, 64)) -> T.Compose:
    """
    Returns the image preprocessing transformations for model input.

    This pipeline prepares raw image data (e.g., NumPy arrays from DICOM)
    for consumption by the PyTorch model. It converts the image to a PIL Image,
    resizes it to the desired dimensions, converts it to a PyTorch tensor,
    and normalizes its pixel values to the range [-1, 1]. This normalization
    is standard for many deep learning models, especially generative ones.

    Args:
        image_size (Tuple[int, int]): The desired output image size (height, width)
                                      after transformation. Defaults to (64, 64).

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline.

    Potential Exceptions Raised:
        - ValueError: If `image_size` contains non-positive dimensions.
        - RuntimeError: If PyTorch or Pillow operations fail internally.

    Example of Usage:
    ```python
    import numpy as np
    from PIL import Image
    # Example raw image (NumPy array, e.g., from DICOM loader)
    raw_image_np = np.random.rand(128, 128).astype(np.float32) * 255 # Assume [0, 255] or [0, 1] range

    # Get the transformation pipeline
    transform_pipeline = get_transforms(image_size=(64, 64))

    # Apply transformation
    transformed_tensor = transform_pipeline(raw_image_np)
    print(f"Transformed image tensor shape: {transformed_tensor.shape}") # Expected: torch.Size([1, 64, 64])
    print(f"Transformed image pixel range: [{transformed_tensor.min():.2f}, {transformed_tensor.max():.2f}]") # Expected: approx [-1, 1]
    ```

    Relationships with other functions/modules:
    - Used by `dataset.LungCTWithGaussianDataset` to preprocess images before they are batched.
    - Used by `main.py` to set up the data loading pipeline.

    Explanation of the theory:
    - **Image Resizing (`T.Resize`):** Adjusts the spatial dimensions of an image.
      This is crucial for ensuring all input images have a consistent size required
      by the neural network architecture.
    - **To Tensor Conversion (`T.ToTensor`):** Converts a PIL Image or NumPy array
      to a PyTorch `Tensor`. It also implicitly scales pixel values from [0, 255]
      (if uint8) or keeps [0, 1] (if float) to [0, 1] for the tensor.
    - **Normalization (`T.Normalize`):** Standardizes pixel values to a specific range
      (e.g., [-1, 1]) by applying the formula `(x - mean) / std`. This helps stabilize
      training and is a common practice for inputs to deep learning models.

    References for the theory:
    - Standard image preprocessing techniques in deep learning.
    - PyTorch documentation for `torchvision.transforms`.
    """
    logger.info(f"Setting up image transformation to resize to {image_size} and normalize to [-1, 1].")
    transform: T.Compose = T.Compose([
        T.ToPILImage(),               # Converts a NumPy array (H, W) or (H, W, C) to a PIL Image.
                                      # Assumes input is float and scales [0,1] to [0,255] if float.
        T.Resize(image_size),         # Resizes the PIL Image to the specified (H, W).
        T.ToTensor(),                 # Converts PIL Image to PyTorch Tensor (C, H, W) and scales pixel values to [0, 1].
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
    for image saving operations (e.g., as PNG). This ensures compatibility with the
    FID calculation library's input requirements.

    Returns:
        torchvision.transforms.Compose: A composed transformation pipeline suitable for FID.

    Potential Exceptions Raised:
        - RuntimeError: If PyTorch or Pillow operations fail internally.

    Example of Usage:
    ```python
    import torch
    # Example generated image tensor (from model output, in [-1, 1] range)
    generated_image_tensor = torch.randn(1, 64, 64) * 0.5 + 0.5 # Example to get it into [-1, 1] roughly

    # Get the FID transformation pipeline
    fid_transform_pipeline = get_fid_transforms()

    # Apply transformation to prepare for FID saving (e.g., to PIL Image)
    pil_image_for_fid = fid_transform_pipeline(generated_image_tensor)
    print(f"Type after FID transform: {type(pil_image_for_fid)}") # Expected: <class 'PIL.Image.Image'>
    ```

    Relationships with other functions/modules:
    - Used by `evaluate.py` to prepare generated and real images for saving to
      temporary directories, which are then consumed by `torch_fidelity` for FID computation.
    - Used by `main.py` to provide the correct transform to the evaluation module.

    Explanation of the theory:
    - **Denormalization:** The reverse process of normalization, converting pixel values
      from a normalized range (e.g., [-1, 1]) back to their original or desired
      range (e.g., [0, 1] or [0, 255]). This is necessary because FID calculation
      libraries or image saving utilities often expect pixel values in the standard
      [0, 255] range.
    - **PIL Image Conversion (`T.ToPILImage`):** Converts a PyTorch Tensor back into
      a Pillow (PIL) Image object. This is often required for image saving operations
      and for external libraries that expect PIL Image inputs.

    References for the theory:
    - `torch_fidelity` library documentation on input requirements.
    - PyTorch documentation for `torchvision.transforms`.
    """
    logger.info("Setting up FID transformation to de-normalize to [0, 1] and convert to PIL Image.")
    fid_transform: T.Compose = T.Compose([
        # De-normalize from [-1, 1] back to [0, 1].
        # Formula: x * std + mean => x * 2.0 + (-1.0) for original mean=0.5, std=0.5
        T.Normalize(mean=[-1.0], std=[2.0]), # This effectively does (x - (-1)) / 2 = (x+1)/2.0
        T.ToPILImage()                        # Converts PyTorch Tensor (C, H, W) to PIL Image (H, W, C or H, W).
                                              # Implicitly scales [0,1] float to [0,255] for saving.
    ])
    return fid_transform