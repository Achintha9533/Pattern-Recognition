"""
DICOM Image Loading and Custom Dataset for Lung CT Scans.

This module provides utilities for loading and preprocessing DICOM images,
specifically focusing on Lung CT scans. It includes a function to read and
normalize individual DICOM files into Hounsfield Units (HU) and a custom
PyTorch Dataset class to manage the loading and transformation of a subset
of these images for use in machine learning models.

Functions
---------
load_dicom_image(file_path)
    Loads a single DICOM image, applies rescaling, and normalizes
    Hounsfield Units (HU) to a [0, 1] range.

Classes
-------
LungCTWithGaussianDataset
    A PyTorch `Dataset` implementation for handling a collection of
    DICOM Lung CT images, selecting a subset from each patient folder,
    and applying specified transformations.

Notes
-----
- The `load_dicom_image` function includes specific Hounsfield Unit (HU)
  windowing assumptions (-1000 to 400 HU) suitable for lung tissue. Adjust
  these values if your dataset requires a different window.
- The `LungCTWithGaussianDataset` selects 5 middle images from each patient
  folder to create the dataset.
"""

from typing import Union
import os
from pathlib import Path
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

# === Load and normalize a single DICOM image ===
def load_dicom_image(file_path: Path) -> Union[np.ndarray, None]:
    """
    Loads a single DICOM image, applies rescaling, and normalizes
    Hounsfield Units (HU) to a [0, 1] range.

    This function reads a DICOM file, extracts pixel data, applies
    `RescaleSlope` and `RescaleIntercept` if present, and then normalizes
    the Hounsfield Units (HU) into a [0, 1] range based on a predefined
    CT window for lung tissue. Values are clipped to ensure they stay
    within this range.

    Parameters
    ----------
    file_path : pathlib.Path
        The full path to the DICOM (.dcm) file to be loaded.

    Returns
    -------
    numpy.ndarray or None
        A 2D NumPy array (float32) representing the normalized image with
        pixel values in the range [0, 1]. Returns `None` if loading fails.

    Warns
    -----
    UserWarning
        If the DICOM file fails to load or process due to an exception.

    Notes
    -----
    - Assumes a common Hounsfield Unit (HU) window for lung CT scans of
      [-1000, 400]. Modify `hu_min` and `hu_max` if your specific dataset
      requires a different windowing for normalization.
    - `pydicom.dcmread` is used with `force=True` to handle potentially
      non-compliant DICOM headers.
    """
    try:
        # Read DICOM file using pydicom, force=True to handle non-standard headers
        ds = pydicom.dcmread(file_path, force=True)

        # Convert pixel data to float32 numpy array
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if they exist in the DICOM header
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Normalize Hounsfield Units (HU) to [0, 1] based on common CT window.
        # Assuming a window of [-1000, 400] HU for lung CT; adjust as needed.
        hu_min = -1000.0
        hu_max = 400.0
        
        if hu_max == hu_min: # Handle cases of uniform images to prevent division by zero
            normalized_image = np.zeros_like(pixel_array)
        else:
            normalized_image = (pixel_array - hu_min) / (hu_max - hu_min)
        
        # Clip values to ensure they are within [0, 1] after normalization
        normalized_image = np.clip(normalized_image, 0.0, 1.0)
        
        return normalized_image

    except Exception as e:
        warnings.warn(f"Failed to load DICOM file '{file_path}': {e}", UserWarning)
        return None

# === Custom Dataset class ===
class LungCTWithGaussianDataset(Dataset):
    """
    PyTorch Dataset for loading and preprocessing a subset of Lung CT DICOM images.

    This dataset iterates through specified patient folders, selects a fixed
    number of middle DICOM images from each, loads them using `load_dicom_image`,
    and applies a provided transformation. For each retrieved item, it also
    generates a corresponding random noise tensor, which is common in diffusion
    model training.

    Parameters
    ----------
    base_dir : pathlib.Path
        The root directory containing subfolders for each patient (e.g., "QIN LUNG CT 1").
    transform : callable, optional
        A transformation function (e.g., `torchvision.transforms.Compose`) to be
        applied to the loaded images. Defaults to `None`.
    image_size : tuple of int, optional
        The target size (height, width) for images if `transform` requires it,
        or for creating default black images if loading fails. Defaults to (96, 96).

    Raises
    ------
    ValueError
        If no readable DICOM images are found in the specified `base_dir`.
    """
    def __init__(self, base_dir: Path, transform=None, image_size: tuple[int, int] = (96, 96)):
        self.transform = transform
        self.image_paths = []
        self.image_size = image_size
        
        # Define the range of patient folders to iterate through.
        # This iterates from 'QIN LUNG CT 1' to 'QIN LUNG CT 47'.
        for i in range(1, 48):
            folder = base_dir / f"QIN LUNG CT {i}"
            current_folder_dicom_files = []
            
            # Walk through subdirectories to find all .dcm files in the current patient folder.
            for root, _, files in os.walk(folder):
                dicom_files = sorted([
                    os.path.join(root, f) for f in files if f.lower().endswith(".dcm")
                ])
                for f_path in dicom_files:
                    try:
                        # Attempt to read the DICOM file to confirm it's readable
                        pydicom.dcmread(f_path, force=True)
                        current_folder_dicom_files.append(f_path)
                    except Exception as e:
                        warnings.warn(f"Skipping unreadable DICOM file: '{f_path}' - {e}", UserWarning)
                        continue
            
            # --- Select 5 middle images from each folder ---
            num_files_in_folder = len(current_folder_dicom_files)
            num_to_select = 5 

            if num_files_in_folder <= num_to_select:
                # If there are 5 or fewer files, take all of them.
                self.image_paths.extend(current_folder_dicom_files)
            else:
                # Calculate start and end index to select the 5 middle images.
                start_index = (num_files_in_folder - num_to_select) // 2
                end_index = start_index + num_to_select
                self.image_paths.extend(current_folder_dicom_files[start_index:end_index])

        if not self.image_paths:
            raise ValueError(
                f"No DICOM images found or loaded from '{base_dir}'. "
                "Please check the base directory path and file permissions."
            )

    def __len__(self) -> int:
        """
        Returns the total number of images (DICOM files) in the dataset.

        Returns
        -------
        int
            The number of image paths collected in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a pair of (noise_tensor, image_tensor) for a given index.

        Loads the DICOM image at the specified index, applies transformations,
        and generates a random noise tensor of the same shape as the processed image.
        Handles cases where image loading fails by returning a black image.

        Parameters
        ----------
        idx : int
            The index of the image to retrieve from the dataset.

        Returns
        -------
        tuple of torch.Tensor
            A tuple containing:
            - noise : torch.Tensor
                A random noise tensor of shape (C, H, W), suitable for diffusion models.
            - image : torch.Tensor
                The loaded, preprocessed, and transformed image tensor of shape (C, H, W).
        """
        img_path = self.image_paths[idx]
        img = load_dicom_image(img_path)

        if img is None:
            # If DICOM loading failed, return a black image tensor with a channel dimension.
            image = torch.zeros(1, *self.image_size, dtype=torch.float32)
        else:
            if img.ndim == 2:
                # Add a channel dimension if the image is just HxW (assuming grayscale).
                image = torch.from_numpy(img).unsqueeze(0) 
            else:
                # If it has more than 2 dimensions, squeeze to ensure it's HxW or CxHxW.
                image = torch.from_numpy(img).squeeze() 
                if image.ndim == 2: # After squeeze, ensure it's HxW, then add channel
                    image = image.unsqueeze(0)
            
            # Apply the transform to the image.
            # load_dicom_image now returns float32 [0,1] numpy array.
            # The `transform` (e.g., from config.py) expects a PIL Image or Tensor,
            # and will typically convert to [-1, 1] and ensure correct shape.
            image = self.transform(image) 
            
            # Final check to ensure the output image has a channel dimension (C, H, W).
            if image.ndim == 2:
                image = image.unsqueeze(0)

        # Generate a random noise tensor with the same shape as the processed image.
        noise = torch.randn_like(image)
        return noise, image