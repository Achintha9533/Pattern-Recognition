# Synthetic Image Generator/load_model.py

from typing import Dict
import torch
import gdown
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Any

"""
Module for loading pre-trained Conditional Normalizing Flow (CNF) model weights
from Google Drive.

This module provides a robust function, `load_model_from_drive`, to facilitate
the retrieval and initialization of a `CNF_UNet` model with pre-trained weights.
It handles downloading the weights file from a specified Google Drive URL,
loading it onto the correct device (CPU or CUDA), and setting the model to
evaluation mode.

It is designed to seamlessly integrate with a PyTorch project structure where
the `CNF_UNet` model and a `config` module (defining parameters like `IMAGE_SIZE`)
are available. For environments where these imports might fail (e.g., standalone
script execution without the full project structure), dummy classes are provided
as fallbacks to allow the script to run for demonstration purposes,
though they will not provide functional model behavior.
"""

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import CNF_UNet model and config from your project structure.
# This assumes 'synthetic_image_generator' is a package in your Python path.
try:
    # Use absolute imports for clarity and consistency
    from synthetic_image_generator.model import CNF_UNet
    from synthetic_image_generator import config # Direct import of config.py
except ImportError as e:
    logger.warning(f"Could not import CNF_UNet or config directly: {e}. Using dummy classes.")
    # Define dummy classes for standalone execution demonstration
    class CNF_UNet(torch.nn.Module):
        # This dummy class signature MUST match the real CNF_UNet signature
        def __init__(self, image_size: Tuple[int, int], input_channels: int, base_channels: int, time_embed_dim: int):
            super().__init__()
            logger.warning("Using dummy CNF_UNet. Model will not be functional.")
            self.image_size = image_size
            self.input_channels = input_channels
            self.linear = torch.nn.Linear(1, 1) # Minimal placeholder layer
        def forward(self, x, t):
            logger.warning("Dummy CNF_UNet forward pass called. Returning zeros.")
            return torch.zeros_like(x)

    class DummyConfig:
        IMAGE_SIZE = (64, 64)
        IMAGE_CHANNELS = 1
        BASE_CHANNELS = 32
        TIME_EMBED_DIM = 256
        GENERATOR_CHECKPOINT_PATH = Path("./dummy_checkpoint.pth") # Placeholder for demo
    config = DummyConfig()


def load_model_from_drive(
    drive_url: str,
    output_path: Path,
    image_size: Tuple[int, int],
    device: torch.device,
    image_channels: int,
    base_channels: int,
    time_embed_dim: int
) -> CNF_UNet:
    """
    Downloads model weights from Google Drive, loads them, and initializes the CNF_UNet model.

    Args:
        drive_url (str): The Google Drive URL (e.g., 'https://drive.google.com/uc?id=FILE_ID').
        output_path (Path): Local path where the downloaded weights file will be saved.
        image_size (Tuple[int, int]): Size of the images the model was trained on (H, W).
        device (torch.device): The device (CPU or CUDA) to load the model onto.
        image_channels (int): Number of input channels for the model (e.g., 1 for grayscale).
        base_channels (int): Base number of channels for the U-Net architecture.
        time_embed_dim (int): Dimension of the time embedding.

    Returns:
        CNF_UNet: An initialized CNF_UNet model with loaded weights.

    Raises:
        RuntimeError: If the model weights cannot be downloaded or loaded.
    """
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory for checkpoints: {output_path.parent}")

    if not output_path.exists():
        logger.info(f"Downloading model weights from {drive_url} to {output_path}...")
        try:
            gdown.download(drive_url, str(output_path), fuzzy=True, quiet=False)
            logger.info("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download model weights from Google Drive: {e}")
    else:
        logger.info(f"Model weights already exist at {output_path}. Skipping download.")

    logger.info(f"Loading model state dictionary from {output_path}...")
    try:
        state_dict = torch.load(output_path, map_location=device)

        # Initialize the model using parameters from config (now passed as arguments)
        # Ensure that the parameters passed match the CNF_UNet constructor exactly.
        model = CNF_UNet(
            image_size=image_size,
            input_channels=image_channels, # This correctly maps to 'input_channels' in CNF_UNet
            base_channels=base_channels,
            time_embed_dim=time_embed_dim
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval() # Set the model to evaluation mode
        logger.info("Model state dictionary loaded successfully.")
        return model

    except FileNotFoundError:
        raise RuntimeError(f"Model checkpoint not found at {output_path}. Please check the path.")
    except Exception as e:
        raise RuntimeError(f"Error loading model state dictionary or initializing model: {e}")

if __name__ == '__main__':
    logger.info("Running load_model.py directly for demonstration.")
    dummy_drive_url = "https://drive.google.com/uc?id=1BWrRqSEY2KSE-u3TI8c2JYucm69g6oo7"
    dummy_output_path = config.GENERATOR_CHECKPOINT_PATH
    dummy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        loaded_model = load_model_from_drive(
            drive_url=dummy_drive_url,
            output_path=dummy_output_path,
            image_size=config.IMAGE_SIZE,
            device=dummy_device,
            image_channels=config.IMAGE_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            time_embed_dim=config.TIME_EMBED_DIM
        )
        logger.info(f"Model loaded successfully on {dummy_device}!")
    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.error(f"An error occurred during model loading demonstration: {e}")
    finally:
        pass