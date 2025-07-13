"""
Configuration settings for the Synthetic Image Generator project.

This module defines all static configuration parameters used throughout the
project. These include data paths, image dimensions, model hyper-parameters,
optimizer settings, checkpoint management, Google Drive integration for
pretrained weights, and image transformation pipelines.

Configuration Variables
-----------------------
base_dir : pathlib.Path
    The base directory containing patient subfolders with DICOM files.
    **IMPORTANT**: This path must be adjusted to your actual data directory
    for the project to run correctly.

image_size : tuple of int
    Desired output image size (height, width) for all image processing steps,
    including resizing inputs and generating outputs.

G_LR : float
    The learning rate for the generator's optimizer during training.

checkpoint_dir : pathlib.Path
    The local directory designated for saving and loading model weights (checkpoints).

generator_checkpoint_path : pathlib.Path
    The full path, including filename, where the final generator model's weights
    are expected to be saved or loaded from. It's constructed relative to `checkpoint_dir`.

GOOGLE_DRIVE_FILE_ID : str
    The Google Drive file ID used to download pretrained generator weights
    if they are not found locally.

transform : torchvision.transforms.Compose
    The comprehensive image preprocessing pipeline applied to input images.
    It includes converting to PIL Image, resizing to `image_size`, converting
    to a PyTorch tensor, and normalizing pixel values to the range [-1, 1].

fid_transform : torchvision.transforms.Compose
    A specialized image transformation pipeline used specifically for the
    Frechet Inception Distance (FID) calculation. This transform first
    denormalizes images from [-1, 1] back to [0, 1], and then converts them
    to a PIL Image, which handles conversion to the [0, 255] range for FID.

device : torch.device
    The computational device (e.g., 'cuda' for GPU or 'cpu') on which PyTorch
    operations and models will be run. It automatically detects CUDA availability.

time_embed_dim : int
    The dimensionality of the time embedding feature used within the model's
    architecture, typically for diffusion models or models sensitive to time steps.

Notes
-----
- Ensure `base_dir` is correctly configured for your local environment.
- This module is designed to be imported directly for accessing these global
  configuration settings throughout the project.
"""

from pathlib import Path
import torch
import torchvision.transforms as T

# Base directory containing patient subfolders with DICOM files
# IMPORTANT: Adjust this path to your actual data directory
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")

# Desired output image size (height, width) for all image operations.
image_size = (96, 96)

# Optimizer learning rate for the Generator model.
G_LR = 1e-4

# === Checkpoint settings ===
# Directory to save trained model weights (and load pre-trained weights from).
checkpoint_dir = Path("./checkpoints")
# Full path to the final generator model checkpoint file.
generator_checkpoint_path = checkpoint_dir / "generator_final.pth"

# Google Drive file ID for the pre-trained weights archive.
GOOGLE_DRIVE_FILE_ID = '1TzXuOGzpt1eR4wxE6GahCX8Ig9ia0ErN'

# === Image preprocessing transform ===
# Defines the sequence of transformations for input images:
# - ToPILImage(): Converts the input (e.g., NumPy array or tensor) to a PIL Image.
# - Resize(image_size): Resizes the image to the specified (height, width).
# - ToTensor(): Converts the PIL Image or NumPy array to a PyTorch FloatTensor.
# - Normalize(mean=[0.5], std=[0.5]): Normalizes pixel values from [0, 1] to [-1, 1].
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Transform for FID calculation:
# This denormalizes images from [-1, 1] to [0, 1] first,
# then ToPILImage() converts the FloatTensor to a PIL Image, implicitly scaling to [0, 255].
fid_transform = T.Compose([
    T.Normalize(mean=[-1.0], std=[2.0]), # Denormalizes from [-1, 1] to [0, 1]
    T.ToPILImage(),
])

# Device setup: Automatically selects CUDA (GPU) if available, otherwise defaults to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyper-parameters: Dimension for time embeddings within the model.
time_embed_dim = 256