# Synthetic Image Generator/dataset.py

import os
from pathlib import Path
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def load_dicom_image(file_path):
    """
    Loads a single DICOM image from the given file path, extracts pixel data,
    and normalizes it to a [0, 1] range.

    Args:
        file_path (Path): The path to the DICOM file.

    Returns:
        np.ndarray or None: The normalized pixel array (float32, [0, 1])
                            or None if loading fails.
    """
    try:
        # Read DICOM file using pydicom
        ds = pydicom.dcmread(file_path)
        # Convert pixel data to float32 numpy array
        img = ds.pixel_array.astype(np.float32)

        # Example range for lung CT images, adjust based on your dataset
        # These values correspond to Hounsfield Units (HU) windowing
        img_min = -1000
        img_max = 400

        if img_max == img_min:
            # Handle cases of uniform images to prevent division by zero
            img = np.zeros_like(img)
            logger.warning(f"Image {file_path} has uniform pixel values (max == min). Returning zeros.")
        else:
            # Normalize to [0, 1] directly as float32
            img = (img - img_min) / (img_max - img_min)

        return img

    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}. Returning None.")
        return None

class LungCTWithGaussianDataset(Dataset):
    """
    A custom PyTorch Dataset for loading Lung CT DICOM images and generating
    corresponding Gaussian noise.

    It iterates over patient folders (QIN LUNG CT 1 to 47) and selects
    a specified number of middle images from each folder.
    """
    def __init__(self, base_dir, transform=None, num_images_per_folder=5, image_size=(64, 64)):
        """
        Initializes the LungCTWithGaussianDataset.

        Args:
            base_dir (Path): The base directory containing patient subfolders with DICOM files.
            transform (torchvision.transforms.Compose, optional): Image transformations to apply. Defaults to None.
            num_images_per_folder (int): Number of middle images to select from each patient folder.
            image_size (tuple): Desired output image size (height, width) for black image fallback.
        """
        self.transform = transform
        self.image_paths = []
        self.image_size = image_size

        # Iterate over patient folders (QIN LUNG CT 1 to 47)
        # This range (1, 48) is hardcoded based on the original script's logic.
        for i in range(1, 48):
            folder = base_dir / f"QIN LUNG CT {i}"
            if not folder.is_dir():
                logger.warning(f"Patient folder not found: {folder}. Skipping.")
                continue

            current_folder_dicom_files = []
            for root, _, files in os.walk(folder):
                dicom_files = sorted([
                    os.path.join(root, f) for f in files if f.lower().endswith(".dcm")
                ])
                for f_path in dicom_files:
                    try:
                        # Only check if pydicom can read the file header, not pixel_array
                        # Pixel array reading is done in load_dicom_image for efficiency
                        pydicom.dcmread(f_path) # Just to check if it's a valid DICOM header
                        current_folder_dicom_files.append(f_path)
                    except Exception as e:
                        logger.warning(f"Skipping unreadable DICOM file: {f_path} - {e}")
                        continue

            # --- Select middle images from each folder ---
            num_files_in_folder = len(current_folder_dicom_files)

            if num_files_in_folder <= num_images_per_folder:
                # If there are fewer or equal files than desired, take all of them
                self.image_paths.extend(current_folder_dicom_files)
            else:
                # Calculate start and end index to select middle images
                start_index = (num_files_in_folder - num_images_per_folder) // 2
                end_index = start_index + num_images_per_folder
                self.image_paths.extend(current_folder_dicom_files[start_index:end_index])

        if not self.image_paths:
            raise ValueError(f"No DICOM images found or loaded from {base_dir}. "
                             "Check path, file permissions, and DICOM validity.")
        logger.info(f"Initialized dataset with {len(self.image_paths)} images.")

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and a corresponding noise tensor at the given index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing (noise_tensor, image_tensor).
                   noise_tensor: A random tensor of the same shape as the image.
                   image_tensor: The loaded and transformed image tensor.
        """
        img_path = self.image_paths[idx]
        img = load_dicom_image(img_path)

        if img is None:
            # Return a black image if loading failed to maintain batch consistency
            image = torch.zeros(1, *self.image_size)
            logger.warning(f"Using black placeholder for {img_path} due to loading failure.")
        else:
            if img.ndim == 2:
                pass # Already 2D
            elif img.ndim == 3 and img.shape[0] == 1:
                img = img.squeeze(0) # Remove singleton channel dimension if present (e.g., (1, H, W) -> (H, W))
            else:
                logger.warning(f"Unexpected image dimensions for {img_path}: {img.shape}. Attempting to squeeze.")
                img = img.squeeze() # Remove all singleton dimensions

            # Apply transform. load_dicom_image now returns float32 [0,1]
            image = self.transform(img)

        # Generate noise of the same shape as the image
        noise = torch.randn_like(image)
        return noise, image