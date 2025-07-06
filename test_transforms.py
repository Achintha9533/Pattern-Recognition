# tests/test_transforms.py

import pytest
import torch
import numpy as np
from PIL import Image
from Synthetic_Image_Generator.transforms import get_transforms, get_fid_transforms

def test_get_transforms_output_type_and_shape():
    """
    Test that get_transforms returns a torchvision.transforms.Compose object
    and that it correctly transforms a dummy image to the expected shape and type.
    """
    image_size = (32, 32)
    transform = get_transforms(image_size)

    assert isinstance(transform, T.Compose)

    # Create a dummy numpy image (H, W)
    dummy_np_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    transformed_img = transform(dummy_np_img)

    assert isinstance(transformed_img, torch.Tensor)
    assert transformed_img.shape == (1, image_size[0], image_size[1]) # C, H, W
    assert transformed_img.dtype == torch.float32

def test_get_transforms_normalization_range():
    """
    Test that get_transforms correctly normalizes pixel values to [-1, 1].
    """
    transform = get_transforms(image_size=(10, 10))

    # Create a dummy numpy image with min/max values
    dummy_np_img = np.array([[0, 255], [127, 50]], dtype=np.uint8)
    transformed_img = transform(dummy_np_img)

    # After ToTensor, values are [0, 1]. After Normalize(mean=0.5, std=0.5),
    # 0 -> (0-0.5)/0.5 = -1
    # 1 -> (1-0.5)/0.5 = 1
    # Check approximate values due to floating point precision
    assert torch.isclose(transformed_img.min(), torch.tensor(-1.0)), "Min pixel value should be -1.0 after normalization."
    assert torch.isclose(transformed_img.max(), torch.tensor(1.0)), "Max pixel value should be 1.0 after normalization."

def test_get_fid_transforms_output_type_and_range():
    """
    Test that get_fid_transforms returns a torchvision.transforms.Compose object
    and correctly denormalizes a tensor from [-1, 1] to [0, 1] and converts to PIL Image.
    """
    fid_transform = get_fid_transforms()

    assert isinstance(fid_transform, T.Compose)

    # Create a dummy tensor normalized to [-1, 1]
    dummy_tensor = torch.tensor([[-1.0, 1.0], [0.0, 0.5]], dtype=torch.float32).unsqueeze(0) # Add channel dim

    transformed_img_pil = fid_transform(dummy_tensor)

    assert isinstance(transformed_img_pil, Image.Image), "Output should be a PIL Image."
    # When PIL converts float [0,1] to uint8, 0 -> 0, 1 -> 255.
    # So, check pixel values in the PIL image (converted to numpy for easier checking)
    transformed_img_np = np.array(transformed_img_pil)

    # Expecting 0 -> 0, 1 -> 255, 0.5 -> 127/128, -1 -> 0
    # The denormalization is (x + 1) / 2
    # -1 -> (-1+1)/2 = 0
    # 1 -> (1+1)/2 = 1
    # 0 -> (0+1)/2 = 0.5
    # 0.5 -> (0.5+1)/2 = 0.75
    # Then PIL converts [0,1] float to [0,255] uint8
    # So, expected values: 0, 255, 127/128, 191/192 (approx)
    assert transformed_img_np.min() >= 0 and transformed_img_np.max() <= 255, "PIL image pixel values should be in [0, 255]."
    assert transformed_img_np[0,0] == 0 # -1.0 -> 0
    assert transformed_img_np[0,1] == 255 # 1.0 -> 255
    # Check approximate values for intermediate points due to float to uint8 conversion
    assert np.isclose(transformed_img_np[1,0], 127.5, atol=1), "0.0 should map to ~127.5 (halfway)"
    assert np.isclose(transformed_img_np[1,1], 191.25, atol=1), "0.5 should map to ~191.25 (three-quarters)"