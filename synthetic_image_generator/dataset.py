# Synthetic Image Generator/dataset.py

import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T # Imported here for type hinting clarity
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

"""
This module provides functionalities for loading and preprocessing medical imaging data,
specifically DICOM files from Lung CT scans, and integrating them with Gaussian noise
for training a Conditional Normalizing Flow (CNF) model. It defines a custom PyTorch
Dataset for efficient data handling during the training process.

The module focuses on robust DICOM image loading, Hounsfield Unit (HU) normalization,
and structured data retrieval from a directory hierarchy.
"""

def load_dicom_image(file_path: Path) -> Optional[np.ndarray]:
    """
    Loads a single DICOM image from the given file path, extracts pixel data,
    and normalizes it to a [0, 1] range based on typical CT Hounsfield Units.

    This function is designed to handle common DICOM pixel data types and applies
    a fixed Hounsfield Unit (HU) window for intensity normalization, which is
    crucial for consistent input to machine learning models. It also handles
    potential errors during file reading or processing, returning None in case of failure.

    Args:
        file_path (Path): The path to the DICOM file. Expected to be a valid
                          path to a .dcm file.

    Returns:
        np.ndarray or None: The normalized pixel array (float32, [0, 1]) if successful,
                            or None if loading or processing fails. The image will be 2D.

    Potential Exceptions Raised:
        - pydicom.errors.InvalidDicomError: If the file is not a valid DICOM file.
        - KeyError: If expected DICOM tags like 'PixelData' are missing.
        - AttributeError: If `ds.pixel_array` is not available.
        - ValueError: If `img_max` and `img_min` are identical, preventing division by zero.
                      (Handled internally to return a black image).
        - OSError: If the file_path is inaccessible.

    Example of Usage:
    ```python
    from pathlib import Path
    # Assuming 'sample.dcm' is a valid DICOM file in the current directory
    dicom_file = Path("path/to/your/sample.dcm")
    image_data = load_dicom_image(dicom_file)
    if image_data is not None:
        print(f"Loaded DICOM image with shape: {image_data.shape} and pixel range: [{image_data.min()}, {image_data.max()}]")
    else:
        print(f"Failed to load DICOM image from {dicom_file}")
    ```

    Relationships with other functions:
    - Called by `LungCTWithGaussianDataset.__getitem__` to load individual images.

    Explanation of the theory:
    - **DICOM (Digital Imaging and Communications in Medicine):** A standard for
      handling, storing, printing, and transmitting information in medical imaging.
      DICOM files contain both image data and metadata (patient info, scan parameters).
    - **Hounsfield Unit (HU):** A quantitative scale used in CT scans to describe
      radiodensity. Normalization to a fixed HU window maps the relevant intensity
      range (e.g., lung tissue) to a standard [0, 1] range, making the data consistent
      across different scans and suitable for neural network input.
      The chosen window (-1000 HU to 400 HU) typically covers air to soft tissue,
      which is common for lung CT analysis.

    References for the theory:
    - Bushberg, J. T., Seibert, J. A., Leidholdt Jr, E. M., & Boone, J. M. (2011).
      The essential physics of medical imaging. Lippincott Williams & Wilkins.
    - DICOM Standard documentation (dicom.nema.org).
    """
    try:
        ds = pydicom.dcmread(file_path)
        img: np.ndarray = ds.pixel_array.astype(np.float32)

        # Define Hounsfield Unit (HU) window for normalization.
        img_min: float = -1000.0  # Lower bound for HU window (e.g., air)
        img_max: float = 400.0   # Upper bound for HU window (e.g., soft tissue)

        if img_max == img_min:
            # Handle cases of uniform images to prevent division by zero.
            logger.warning(f"Image max and min HU are identical for {file_path}. Returning a zero array.")
            return np.zeros_like(img)

        # Normalize pixel values from HU to [0, 1] range.
        img = (img - img_min) / (img_max - img_min)
        img = np.clip(img, 0.0, 1.0) # Clip values to ensure they are strictly within [0, 1]
        return img
    except Exception as e:
        logger.error(f"Failed to load or process DICOM image {file_path}: {e}")
        return None


class LungCTWithGaussianDataset(Dataset):
    """
    A PyTorch Dataset for loading Lung CT images and generating corresponding
    Gaussian noise samples.

    This dataset iterates through a directory structure where each patient has
    a subfolder containing DICOM images. It selects a specified number of
    "middle" images from each patient's scan to form the dataset, pairs them
    with random Gaussian noise, and applies image transformations. This setup
    is designed for training generative models like Conditional Normalizing Flows
    that learn to transform noise into realistic images.
    """

    def __init__(
        self,
        base_dir: Path,
        num_images_per_folder: int = 5,
        image_size: Tuple[int, int] = (64, 64),
        transform: Optional[T.Compose] = None
    ):
        """
        Initializes the LungCTWithGaussianDataset.

        Args:
            base_dir (Path): The base directory containing patient subfolders.
                             Each subfolder is expected to contain DICOM (.dcm) files.
            num_images_per_folder (int): The number of DICOM images to select from
                                         the middle of each patient's scan.
                                         Defaults to 5.
            image_size (Tuple[int, int]): The target (height, width) for image
                                          resizing. Used for generating noise of
                                          the correct shape. Defaults to (64, 64).
            transform (Optional[T.Compose]): A torchvision.transforms.Compose object
                                             to apply to the loaded images. This
                                             typically includes resizing, conversion
                                             to tensor, and normalization.
                                             Defaults to None (no transformation).

        Returns:
            None: The constructor initializes the dataset object.

        Potential Exceptions Raised:
            - FileNotFoundError: If `base_dir` does not exist or contains no patient folders.
            - ValueError: If `num_images_per_folder` is non-positive.
            - Exception during file listing (e.g., permission errors).

        Example of Usage:
        ```python
        from pathlib import Path
        import torchvision.transforms as T
        from .transforms import get_transforms # Assuming this is available
        # from . import config # If using config for BASE_DIR, IMAGE_SIZE

        # Set up transformations
        image_transform = get_transforms(image_size=(128, 128))

        # Initialize dataset
        # base_data_dir = Path("/path/to/your/DICOM_data")
        # dataset = LungCTWithGaussianDataset(
        #     base_dir=base_data_dir,
        #     num_images_per_folder=10,
        #     image_size=(128, 128),
        #     transform=image_transform
        # )
        # print(f"Dataset has {len(dataset)} images.")
        # sample_noise, sample_image = dataset[0]
        # print(f"Sample image shape: {sample_image.shape}, noise shape: {sample_noise.shape}")
        ```

        Relationships with other functions/modules:
        - Uses `load_dicom_image` for loading individual DICOM files.
        - Utilizes `torch.utils.data.Dataset` as its base class.
        - Relies on `torchvision.transforms` for image preprocessing.
        - `config.py` typically provides `base_dir`, `num_images_per_folder`, and `image_size`.
        - Consumed by `torch.utils.data.DataLoader` in `main.py` or `train.py`.

        Explanation of the theory:
        - **PyTorch Dataset:** An abstract class representing a dataset. Custom datasets
          inherit from `torch.utils.data.Dataset` and must override `__len__` and
          `__getitem__`. This abstraction allows PyTorch's `DataLoader` to efficiently
          batch and shuffle data for training.
        - **Data Sampling (Middle Images):** For 3D medical scans (like CT), selecting
          a subset of images from the middle of the scan (e.g., `num_images_per_folder`)
          is a common practice. This is because slices at the beginning and end of a
          scan often contain less relevant anatomical information or more artifacts.
        """
        self.base_dir = Path(base_dir)
        self.num_images_per_folder = num_images_per_folder
        self.image_size = image_size
        self.transform = transform
        self.image_paths: List[Path] = []

        if not self.base_dir.is_dir():
            logger.error(f"Base directory not found: {self.base_dir}")
            raise FileNotFoundError(f"Base directory not found: {self.base_dir}")

        self._load_image_paths()
        if not self.image_paths:
            logger.warning(f"No valid images found in {self.base_dir}. Dataset will be empty.")

    def _load_image_paths(self) -> None:
        """
        (Private Method) Populates `self.image_paths` by scanning the base directory
        for patient subfolders and selecting a specified number of DICOM images
        from the middle of each folder.

        This method iterates through subdirectories of `base_dir`, assumes each
        subdirectory represents a patient, and then sorts the DICOM files within
        to consistently pick 'middle' slices. This ensures that the dataset
        primarily consists of relevant anatomical views.
        """
        logger.info(f"Scanning base directory: {self.base_dir} for DICOM files...")
        patient_folders = [f for f in self.base_dir.iterdir() if f.is_dir()]
        total_found = 0

        for patient_folder in tqdm(patient_folders, desc="Collecting image paths"):
            dicom_files = sorted(list(patient_folder.glob("*.dcm")))
            if not dicom_files:
                logger.debug(f"No DICOM files found in {patient_folder}")
                continue

            # Select N images from the middle of the sorted list
            if len(dicom_files) > self.num_images_per_folder:
                start_index = (len(dicom_files) - self.num_images_per_folder) // 2
                selected_files = dicom_files[start_index : start_index + self.num_images_per_folder]
            else:
                selected_files = dicom_files # Take all if fewer than desired

            self.image_paths.extend(selected_files)
            total_found += len(selected_files)

        logger.info(f"Finished scanning. Found {total_found} DICOM images across all folders.")

    def __len__(self) -> int:
        """
        (Magic Method) Returns the total number of images in the dataset.

        Returns:
            int: The number of images available in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (Magic Method) Retrieves a single data sample (Gaussian noise, real image)
        from the dataset at the given index.

        This method is critical for PyTorch's DataLoader. It loads a DICOM image,
        applies transformations, and generates a corresponding Gaussian noise tensor
        of the same shape, returning both as PyTorch tensors.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - noise_tensor (torch.Tensor): A randomly generated Gaussian noise tensor
                                               with shape (1, H, W).
                - image_tensor (torch.Tensor): The preprocessed real image tensor
                                               with shape (1, H, W).
                                               Pixel values are typically normalized to [-1, 1].

        Potential Exceptions Raised:
            - IndexError: If `idx` is out of bounds for `self.image_paths`.
            - Exception from `load_dicom_image`: If the DICOM file fails to load.
            - Exception from `self.transform`: If the transformation pipeline fails.

        Example of Usage:
        ```python
        # Assuming `dataset` is an initialized LungCTWithGaussianDataset object
        # noise_sample, image_sample = dataset[0]
        # print(f"Noise tensor shape: {noise_sample.shape}")
        # print(f"Image tensor shape: {image_sample.shape}")
        ```

        Relationships with other functions/modules:
        - Calls `load_dicom_image`.
        - Uses `self.transform` (from `torchvision.transforms`).
        - Called by `torch.utils.data.DataLoader` during training/evaluation loops.

        Explanation of the theory:
        - **Gaussian Noise:** A random variable that follows a normal (Gaussian) distribution.
          It serves as the starting point for generative models like CNFs, which learn
          to deterministically transform this noise into target data.
        - **Data Loading Pipeline:** `__getitem__` defines the critical steps for
          preparing a single data point: loading, transforming (resizing, normalization),
          and pairing with noise. This prepares the input for the generative model.
        """
        img_path: Path = self.image_paths[idx]
        img_np: Optional[np.ndarray] = load_dicom_image(img_path)

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