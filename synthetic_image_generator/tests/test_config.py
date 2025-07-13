# tests/test_config.py
import torch
import torchvision.transforms as T
from pathlib import Path

# Adjust import path based on your project structure
from config import (
    base_dir, image_size, G_LR, checkpoint_dir, 
    generator_checkpoint_path, GOOGLE_DRIVE_FILE_ID,
    transform, fid_transform, device, time_embed_dim
)

def test_config_types():
    """Verify the types of all configuration variables."""
    assert isinstance(base_dir, Path)
    assert isinstance(image_size, tuple)
    assert all(isinstance(dim, int) for dim in image_size)
    assert isinstance(G_LR, float)
    assert isinstance(checkpoint_dir, Path)
    assert isinstance(generator_checkpoint_path, Path)
    assert isinstance(GOOGLE_DRIVE_FILE_ID, str)
    assert isinstance(transform, T.Compose)
    assert isinstance(fid_transform, T.Compose)
    assert isinstance(device, torch.device)
    assert isinstance(time_embed_dim, int)

def test_config_values():
    """Verify specific values of key configuration variables."""
    assert image_size == (96, 96)
    assert G_LR == 1e-4
    assert time_embed_dim == 256
    assert str(checkpoint_dir) == "checkpoints"
    assert generator_checkpoint_path.name == "generator_final.pth"

def test_device_selection():
    """Check if the device is correctly set."""
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.type == expected_device