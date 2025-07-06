# Synthetic Image Generator/config.py

from pathlib import Path
from typing import Tuple

"""
Configuration module for the Synthetic Image Generator project.

This module centralizes all global parameters and settings used across the application,
including image dimensions, learning rates, dataset paths, and training/generation
parameters. This approach enhances maintainability, allows for easy modification of
experiment settings, and minimizes reliance on hardcoded values within the logic.
"""

# === General Configuration ===
# Desired output image size (height, width)
IMAGE_SIZE: Tuple[int, int] = (64, 64)

# Optimizer learning rates for the generator
G_LR: float = 1e-4

# === Dataset Configuration ===
# Base directory containing patient subfolders with DICOM files.
# IMPORTANT: Adjust this path to your actual data directory on your system.
# Example: BASE_DIR = Path("/path/to/your/QIN LUNG CT")
BASE_DIR: Path = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")

# Number of middle images to select from each patient folder.
# This helps manage dataset size and focuses on central slices.
NUM_IMAGES_PER_FOLDER: int = 5

# === Checkpoint Settings ===
# Directory to save trained model weights.
CHECKPOINT_DIR: Path = Path("./checkpoints")
# Ensure the directory exists. It will be created if it doesn't already.
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GENERATOR_CHECKPOINT_PATH: Path = CHECKPOINT_DIR / "generator_final.pth"

# === Training Settings ===
# Number of training epochs.
EPOCHS: int = 150
# Batch size for training.
BATCH_SIZE: int = 64
# Number of worker processes for data loading. Set to 0 for macOS compatibility
# or if encountering multiprocessing issues.
NUM_WORKERS: int = 0

# === Generation Settings ===
# Number of steps for the image generation process (Euler integration steps).
# Higher values lead to more accurate generation but are computationally more expensive.
GENERATION_STEPS: int = 200
# Number of synthetic samples to generate for evaluation and visualization.
NUM_GENERATED_SAMPLES: int = 256

# === Evaluation Settings ===
# Number of batches to sample from the real dataset for pixel distribution plots.
# This provides a representative sample of real image pixel values.
NUM_BATCHES_FOR_DIST_PLOT: int = 5
