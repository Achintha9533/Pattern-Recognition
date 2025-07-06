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

    Given a desired `image_size` and a dummy NumPy image representing raw input data,
    When `get_transforms` is called to create a transformation pipeline and this
    pipeline is applied to the dummy image,
    Then the output should be a `torch.Tensor` with the correct channel-first shape
    (1, H, W) (where H and W are from `image_size`) and `float32` data type.

    Args:
        None.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If the returned type is not `T.Compose`, or if the transformed
                        image has an incorrect type, shape, or data type.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the `get_transforms` function from the `transforms` module.

    Explanation of the Theory:
        This test verifies the foundational image preprocessing pipeline.
        Correct transformations are vital to ensure that images are in the
        right format and range for neural network consumption, preventing
        issues during model training and inference.
    """
    image_size: Tuple[int, int] = (32, 32)
    transform: T.Compose = get_transforms(image_size)

    assert isinstance(transform, T.Compose), "Expected return type to be torchvision.transforms.Compose."

    # Create a dummy numpy image (H, W) with uint8 dtype, simulating raw image data.
    dummy_np_img: np.ndarray = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    transformed_img: torch.Tensor = transform(dummy_np_img)

    assert isinstance(transformed_img, torch.Tensor), "Transformed output should be a torch.Tensor."
    assert transformed_img.shape == (1, *image_size), f"Expected shape (1, {image_size}), but got {transformed_img.shape}."
    assert transformed_img.dtype == torch.float32, "Expected dtype float32."

def test_get_transforms_pixel_value_normalization() -> None:
    """
    Test that `get_transforms` correctly normalizes pixel values from [0, 255] (or similar)
    to the target range of [-1, 1].

    Given a dummy NumPy image with known pixel values (e.g., 0, 127, 255),
    When `get_transforms` is used to apply the transformation pipeline,
    Then the pixel values in the resulting `torch.Tensor` should be accurately
    normalized to the [-1, 1] range.

    Args:
        None.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If the normalized pixel values do not match the expected range
                        or specific values within that range.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the normalization step within the `get_transforms` pipeline.

    Explanation of the Theory:
        Normalization is a critical preprocessing step for neural networks,
        as it helps stabilize training by ensuring input features are on a
        consistent scale. This test confirms the correctness of that scaling.
    """
    image_size: Tuple[int, int] = (1, 1) # Smallest possible for a single pixel
    transform: T.Compose = get_transforms(image_size)

    # Test with specific values that map clearly after normalization
    # Raw value 0 -> PIL float (0.0) -> Tensor (0.0) -> Normalized (-1.0)
    # Raw value 255 -> PIL float (1.0) -> Tensor (1.0) -> Normalized (1.0)
    # Raw value 127.5 (approx) -> PIL float (0.5) -> Tensor (0.5) -> Normalized (0.0)
    
    # Create a dummy image with known values to test normalization
    dummy_np_img = np.array([[0, 255], [128, 64]], dtype=np.uint8) # 128 for 0.5 approx due to int conversion
    
    transformed_img: torch.Tensor = transform(dummy_np_img)

    # Expected values after ToPILImage (uint8 to float [0,1]), ToTensor (H,W to 1,H,W), Normalize (0,1 to -1,1)
    # x_norm = (x_float - 0.5) / 0.5 = 2 * x_float - 1
    # 0 (uint8) -> 0.0 (float) -> -1.0 (normalized)
    # 255 (uint8) -> 1.0 (float) -> 1.0 (normalized)
    # 128 (uint8) -> 128/255 approx 0.50196 -> 2*0.50196-1 approx 0.00392 (very close to 0)
    # 64 (uint8) -> 64/255 approx 0.25098 -> 2*0.25098-1 approx -0.498

    # Use torch.allclose for floating point comparisons
    assert torch.allclose(transformed_img[0, 0, 0], torch.tensor(-1.0)), "Pixel (0,0) should normalize to -1.0"
    assert torch.allclose(transformed_img[0, 0, 1], torch.tensor(1.0)), "Pixel (0,1) should normalize to 1.0"
    # Check values for 128 and 64, allowing for minor precision differences due to uint8 to float conversion.
    assert torch.allclose(transformed_img[0, 1, 0], torch.tensor(0.003921568627450979), atol=1e-5), "Pixel (1,0) should normalize close to 0."
    assert torch.allclose(transformed_img[0, 1, 1], torch.tensor(-0.49803921568627447), atol=1e-5), "Pixel (1,1) should normalize close to -0.5."


def test_get_fid_transforms_output_type_and_pil_conversion() -> None:
    """
    Test that `get_fid_transforms` returns a `torchvision.transforms.Compose` object
    and correctly transforms a tensor (normalized to [-1, 1]) back to a PIL Image.

    Given a dummy `torch.Tensor` representing a generated image (pixel values in [-1, 1]),
    When `get_fid_transforms` is called to create a transformation pipeline and this
    pipeline is applied to the dummy tensor,
    Then the output should be a `PIL.Image.Image` object, and its pixel values
    should be denormalized back to an appropriate range (implicitly [0, 255] for uint8 PIL).

    Args:
        None.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If the returned type is not `T.Compose`, or if the transformed
                        output is not a PIL Image or has incorrect pixel range.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the `get_fid_transforms` function from the `transforms` module.
        * Complements `get_transforms` by testing the reverse operation for evaluation.

    Explanation of the Theory:
        FID (Fréchet Inception Distance) calculation often requires images
        to be in a specific format (e.g., 8-bit, [0, 255]) before being
        fed to the pre-trained Inception model. This test ensures the
        post-processing pipeline for FID is correct.
    """
    fid_transform: T.Compose = get_fid_transforms()

    assert isinstance(fid_transform, T.Compose), "Expected return type to be torchvision.transforms.Compose."

    # Create a dummy torch.Tensor representing a generated image, normalized to [-1, 1].
    dummy_tensor_img: torch.Tensor = torch.tensor([[[[-1.0, 0.0], [0.5, 1.0]]]], dtype=torch.float32) # (1, C, H, W)
    
    transformed_img_pil: Image.Image = fid_transform(dummy_tensor_img)

    assert isinstance(transformed_img_pil, Image.Image), "Transformed output should be a PIL Image."

    # Denormalization: `x_float = x_norm * std + mean` => `x_float = x_norm * 0.5 + 0.5`
    # Our FID transform denormalizes by `T.Normalize(mean=[-1.0], std=[2.0])`
    # This means `x_denorm = x_input * 2.0 + (-1.0)`
    # -1.0 (input) -> -1.0 * 2.0 + (-1.0) = -3.0 (incorrect based on intended denorm)
    # Let's re-evaluate the get_fid_transforms normalization:
    # get_fid_transforms() applies T.Normalize(mean=[-1.0], std=[2.0])
    # This transform's purpose is actually to *reverse* an earlier normalization.
    # If the earlier normalization was (x - 0.5) / 0.5 => y = 2x - 1, then x = (y + 1) / 2.
    # So to go from [-1, 1] back to [0, 1], we need (x_tensor + 1) / 2.
    # T.Normalize(mean=[-1.0], std=[2.0]) does `(x - (-1.0)) / 2.0 = (x + 1) / 2`.
    # This is correct. So:
    # -1.0 (input) -> (-1.0 + 1) / 2 = 0.0
    # 0.0  (input) -> (0.0 + 1) / 2 = 0.5
    # 0.5  (input) -> (0.5 + 1) / 2 = 0.75
    # 1.0  (input) -> (1.0 + 1) / 2 = 1.0
    # Then ToPILImage converts float [0,1] to uint8 [0,255].
    # So, check pixel values in the PIL image (converted to numpy for easier checking).
    transformed_img_np: np.ndarray = np.array(transformed_img_pil)

    # Expected denormalization values for original tensor:
    # -1.0  -> 0.0 (float) -> 0 (uint8)
    # 0.0   -> 0.5 (float) -> 127 or 128 (uint8, depends on rounding)
    # 0.5   -> 0.75 (float) -> 191 or 192 (uint8)
    # 1.0   -> 1.0 (float) -> 255 (uint8)

    assert transformed_img_np.min() >= 0, f"PIL image min pixel value should be >= 0, but got {transformed_img_np.min()}."
    assert transformed_img_np.max() <= 255, f"PIL image max pixel value should be <= 255, but got {transformed_img_np.max()}."
    
    assert transformed_img_np[0, 0] == 0, "Expected pixel (0,0) to be 0 after FID transform."
    # Allow for rounding differences for 0.5 -> 127/128 and 0.75 -> 191/192
    assert transformed_img_np[0, 1] in [127, 128], f"Expected pixel (0,1) to be 127 or 128, but got {transformed_img_np[0, 1]}."
    assert transformed_img_np[1, 0] in [191, 192], f"Expected pixel (1,0) to be 191 or 192, but got {transformed_img_np[1, 0]}."
    assert transformed_img_np[1, 1] == 255, "Expected pixel (1,1) to be 255 after FID transform."