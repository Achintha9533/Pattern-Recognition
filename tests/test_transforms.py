# tests/test_transforms.py

import pytest
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from typing import Tuple

# Import modules from your package
from Synthetic_Image_Generator.transforms import get_transforms, get_fid_transforms

"""
Test suite for the transformations module.

This module contains unit tests for the `get_transforms` and `get_fid_transforms`
functions, ensuring that image preprocessing and post-processing pipelines
behave as expected, including resizing, type conversion, and pixel normalization/denormalization.
"""

def test_get_transforms_output_type_and_shape() -> None:
    """
    Test that `get_transforms` returns a `torchvision.transforms.Compose` object
    and correctly transforms a dummy image to the expected shape and data type.

    Given a desired `image_size` and a dummy NumPy image,
    When `get_transforms` is called and the transform is applied,
    Then the output should be a `torch.Tensor` with the correct channel-first shape
    (1, H, W) and `float32` data type.
    """
    image_size: Tuple[int, int] = (32, 32)
    transform: T.Compose = get_transforms(image_size)

    assert isinstance(transform, T.Compose), "Expected return type to be torchvision.transforms.Compose."

    # Create a dummy numpy image (H, W) with uint8 dtype, simulating raw image data.
    dummy_np_img: np.ndarray = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    transformed_img: torch.Tensor = transform(dummy_np_img)

    assert isinstance(transformed_img, torch.Tensor), "Transformed image should be a torch.Tensor."
    assert transformed_img.shape == (1, image_size[0], image_size[1]), \
        f"Transformed image shape should be (1, {image_size[0]}, {image_size[1]}), but got {transformed_img.shape}."
    assert transformed_img.dtype == torch.float32, f"Transformed image dtype should be float32, but got {transformed_img.dtype}."

def test_get_transforms_normalization_range() -> None:
    """
    Test that `get_transforms` correctly normalizes pixel values to the [-1, 1] range.

    Given a dummy NumPy image with known min/max pixel values (0 and 255),
    When `get_transforms` is applied,
    Then the minimum pixel value in the output tensor should be approximately -1.0
    and the maximum should be approximately 1.0.
    """
    transform: T.Compose = get_transforms(image_size=(10, 10))

    # Create a dummy numpy image with min/max values to test normalization.
    dummy_np_img: np.ndarray = np.array([[0, 255], [127, 50]], dtype=np.uint8)
    transformed_img: torch.Tensor = transform(dummy_np_img)

    # After `ToTensor`, values are [0, 1]. After `Normalize(mean=0.5, std=0.5)`:
    # 0   -> (0 - 0.5) / 0.5 = -1.0
    # 1   -> (1 - 0.5) / 0.5 = 1.0
    # 0.5 -> (0.5 - 0.5) / 0.5 = 0.0
    # Check approximate values due to floating point precision.
    assert torch.isclose(transformed_img.min(), torch.tensor(-1.0, dtype=torch.float32)), \
        f"Min pixel value should be -1.0 after normalization, but got {transformed_img.min().item()}."
    assert torch.isclose(transformed_img.max(), torch.tensor(1.0, dtype=torch.float32)), \
        f"Max pixel value should be 1.0 after normalization, but got {transformed_img.max().item()}."

def test_get_fid_transforms_output_type_and_range() -> None:
    """
    Test that `get_fid_transforms` returns a `torchvision.transforms.Compose` object
    and correctly denormalizes a tensor from [-1, 1] to [0, 1] before converting to a PIL Image.

    Given a dummy tensor normalized to [-1, 1],
    When `get_fid_transforms` is applied,
    Then the output should be a PIL Image with pixel values in the [0, 255] range,
    and specific pixel values should map correctly after denormalization and conversion.
    """
    fid_transform: T.Compose = get_fid_transforms()

    assert isinstance(fid_transform, T.Compose), "Expected return type to be torchvision.transforms.Compose."

    # Create a dummy tensor normalized to [-1, 1] to simulate model output.
    dummy_tensor: torch.Tensor = torch.tensor([[-1.0, 1.0], [0.0, 0.5]], dtype=torch.float32).unsqueeze(0) # Add channel dim

    transformed_img_pil: Image.Image = fid_transform(dummy_tensor)

    assert isinstance(transformed_img_pil, Image.Image), "Output should be a PIL Image."
    
    # When PIL converts float [0,1] to uint8, 0 -> 0, 1 -> 255.
    # So, check pixel values in the PIL image (converted to numpy for easier checking).
    transformed_img_np: np.ndarray = np.array(transformed_img_pil)

    # Expected denormalization: `(x + 1) / 2`
    # -1.0 -> (-1.0 + 1) / 2 = 0.0
    # 1.0  -> (1.0 + 1) / 2 = 1.0
    # 0.0  -> (0.0 + 1) / 2 = 0.5
    # 0.5  -> (0.5 + 1) / 2 = 0.75
    # Then PIL converts [0,1] float to [0,255] uint8:
    # 0.0  -> 0
    # 1.0  -> 255
    # 0.5  -> 127.5 (rounds to 127 or 128 depending on PIL version/method)
    # 0.75 -> 191.25 (rounds to 191 or 192)

    assert transformed_img_np.min() >= 0, f"PIL image min pixel value should be >= 0, but got {transformed_img_np.min()}."
    assert transformed_img_np.max() <= 255, f"PIL image max pixel value should be <= 255, but got {transformed_img_np.max()}."
    
    assert transformed_img_np[0,0] == 0, f"Pixel at (0,0) should be 0, but got {transformed_img_np[0,0]}." # -1.0 -> 0
    assert transformed_img_np[0,1] == 255, f"Pixel at (0,1) should be 255, but got {transformed_img_np[0,1]}." # 1.0 -> 255
    
    # Use np.isclose for floating point comparisons that are then cast to integer, allowing for small rounding differences.
    assert np.isclose(transformed_img_np[1,0], 127.5, atol=1), f"Pixel at (1,0) should be ~127.5, but got {transformed_img_np[1,0]}." # 0.0 -> ~127.5
    assert np.isclose(transformed_img_np[1,1], 191.25, atol=1), f"Pixel at (1,1) should be ~191.25, but got {transformed_img_np[1,1]}." # 0.5 -> ~191.25
