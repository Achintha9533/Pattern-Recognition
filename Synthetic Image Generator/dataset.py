# Synthetic Image Generator/dataset.py

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T # Imported here for type hinting clarity
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

def load_dicom_image(file_path: Path) -> Optional[np.ndarray]:
    """
    Loads a single DICOM image from the given file path, extracts pixel data,
    and normalizes it to a [0, 1] range based on typical CT Hounsfield Units.

    Args:
        file_path (Path): The path to the DICOM file.

    Returns:
        np.ndarray or None: The normalized pixel array (float32, [0, 1])
                            or None if loading or processing fails.
    """
    try:
        # Read DICOM file using pydicom
        ds = pydicom.dcmread(file_path)
        # Convert pixel data to float32 numpy array
        img: np.ndarray = ds.pixel_array.astype(np.float32)

        # Define Hounsfield Unit (HU) window for normalization.
        # These values are typical for lung CT images and may need adjustment
        # based on the specific dataset's characteristics.
        img_min: float = -1000.0  # Lower bound for HU window
        img_max: float = 400.0   # Upper bound for HU window

        if img_max == img_min:
            # Handle cases of uniform images to prevent division by zero.
            # This ensures that even if all pixels are the same, the normalization
            # doesn't break, and a zero array is returned.
            img = np.zeros_like(img)
            logger.warning(f"Image {file_path} has uniform pixel values (max == min). Returning zeros.")
        else:
            # Normalize pixel values to the [0, 1] range.
            # Pixels outside the [img_min, img_max] range will be clipped implicitly
            # by the division and subsequent clamping in the transform if applied.
            img = (img - img_min) / (img_max - img_min)

        return img

    except Exception as e:
        # Log a warning if DICOM loading or processing fails for a specific file.
        # This allows the dataset loading to continue for other valid files.
        logger.warning(f"Failed to load DICOM image from {file_path}: {e}. Returning None.")
        return None

class LungCTWithGaussianDataset(Dataset):
    """
    A custom PyTorch Dataset for loading Lung CT DICOM images and generating
    corresponding Gaussian noise.

    This dataset iterates over specified patient folders (e.g., QIN LUNG CT 1 to 47)
    and selects a predefined number of middle images from each folder to ensure
    a representative sample and manage dataset size.
    """
    def __init__(self, base_dir: Path, transform: Optional[T.Compose] = None,
                 num_images_per_folder: int = 5, image_size: Tuple[int, int] = (64, 64)):
        """
        Initializes the LungCTWithGaussianDataset.

        Args:
            base_dir (Path): The base directory containing patient subfolders with DICOM files.
                             Each subfolder is expected to represent a patient or series.
            transform (Optional[T.Compose]): Image transformations to apply to the loaded images.
                                             Defaults to None, but typically a torchvision.transforms.Compose
                                             object is provided for resizing, tensor conversion, and normalization.
            num_images_per_folder (int): The number of middle images to select from each
                                         patient folder. If a folder has fewer images than this number,
                                         all available images from that folder will be included.
            image_size (Tuple[int, int]): Desired output image size (height, width). Used for
                                          creating black placeholder images if DICOM loading fails,
                                          to maintain consistent batch dimensions.

        Raises:
            ValueError: If no valid DICOM images are found or successfully loaded from the
                        specified base directory after scanning all patient folders.
        """
        self.transform = transform
        self.image_paths: list[str] = []
        self.image_size = image_size

        # Iterate over patient folders (hardcoded range 1 to 47 based on dataset structure)
        for i in range(1, 48):
            folder: Path = base_dir / f"QIN LUNG CT {i}"
            if not folder.is_dir():
                logger.warning(f"Patient folder not found: {folder}. Skipping this folder.")
                continue

            current_folder_dicom_files: list[str] = []
            # Walk through the folder to find all DICOM files
            for root, _, files in os.walk(folder):
                dicom_files = sorted([
                    os.path.join(root, f) for f in files if f.lower().endswith(".dcm")
                ])
                for f_path in dicom_files:
                    try:
                        # Attempt to read just the DICOM header to confirm it's a valid DICOM file.
                        # This is quicker than loading pixel data for all files during initialization.
                        pydicom.dcmread(f_path)
                        current_folder_dicom_files.append(f_path)
                    except Exception as e:
                        logger.warning(f"Skipping unreadable or invalid DICOM file: {f_path} - {e}")
                        continue

            # --- Select a specified number of middle images from each folder ---
            num_files_in_folder: int = len(current_folder_dicom_files)

            if num_files_in_folder <= num_images_per_folder:
                # If the folder has fewer or equal files than desired, include all of them.
                self.image_paths.extend(current_folder_dicom_files)
            else:
                # Calculate the start and end indices to select the middle images.
                start_index: int = (num_files_in_folder - num_images_per_folder) // 2
                end_index: int = start_index + num_images_per_folder
                self.image_paths.extend(current_folder_dicom_files[start_index:end_index])

        if not self.image_paths:
            # Raise an error if no DICOM images were found or successfully processed.
            # This indicates a potential issue with the base_dir or data format.
            raise ValueError(f"No DICOM images found or loaded from '{base_dir}'. "
                             "Please check the base directory path, file permissions, "
                             "and ensure valid DICOM files are present.")
        logger.info(f"Initialized dataset with {len(self.image_paths)} images from '{base_dir}'.")

    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a pair of tensors: a Gaussian noise tensor and the corresponding
        loaded and transformed image tensor at the given index.

        Args:
            idx (int): The index of the item to retrieve from the dataset.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - noise_tensor (torch.Tensor): A random tensor of the same shape as the image,
                                               representing the initial state for generation.
                - image_tensor (torch.Tensor): The loaded, preprocessed, and transformed
                                               image tensor, ready for model input.
        """
        img_path: str = self.image_paths[idx]
        img_np: Optional[np.ndarray] = load_dicom_image(Path(img_path))

        if img_np is None:
            # If DICOM loading failed, return a black image placeholder to maintain
            # batch consistency and prevent downstream errors.
            image_tensor: torch.Tensor = torch.zeros(1, *self.image_size, dtype=torch.float32)
            logger.warning(f"Using black placeholder for {img_path} due to loading failure.")
        else:
            # Ensure the image is 2D (remove singleton channel dimensions if present).
            if img_np.ndim == 2:
                pass # Already 2D
            elif img_np.ndim == 3 and img_np.shape[0] == 1:
                img_np = img_np.squeeze(0) # Remove singleton channel dimension (e.g., (1, H, W) -> (H, W))
            else:
                # Log a warning if dimensions are unexpected but attempt to squeeze.
                logger.warning(f"Unexpected image dimensions for {img_path}: {img_np.shape}. Attempting to squeeze.")
                img_np = img_np.squeeze() # Remove all singleton dimensions

            # Apply the specified transformations (e.g., resize, to tensor, normalize).
            # load_dicom_image already returns float32 in [0,1] range.
            image_tensor = self.transform(img_np) if self.transform else torch.from_numpy(img_np).unsqueeze(0)

        # Generate Gaussian noise of the same shape as the image tensor.
        noise_tensor: torch.Tensor = torch.randn_like(image_tensor)
        return noise_tensor, image_tensor
