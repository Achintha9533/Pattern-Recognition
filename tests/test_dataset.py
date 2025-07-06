# tests/test_dataset.py

import pytest
from pathlib import Path
import torch
import numpy as np
import pydicom
from pydicom.filewriter import dcmwrite
from pydicom.dataset import Dataset, FileDataset
import os
import shutil
from typing import Generator, Tuple, Any, Optional

# Import modules from your package
from Synthetic_Image_Generator.dataset import load_dicom_image, LungCTWithGaussianDataset
from Synthetic_Image_Generator.transforms import get_transforms # Needed for dataset init

"""
Test suite for the dataset module.

This module contains unit tests for the `load_dicom_image` function and the
`LungCTWithGaussianDataset` class, ensuring correct DICOM file loading,
image preprocessing, and dataset behavior.
"""

# --- Fixtures for creating dummy DICOM data ---
@pytest.fixture(scope="function")
def dummy_dicom_data_path(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Fixture to create a temporary directory with dummy DICOM files for testing.

    Given a temporary path, this fixture creates a simulated dataset structure
    with patient folders and DICOM files, mimicking the expected input for
    `LungCTWithGaussianDataset`. It generates files with varying counts to
    test image selection logic.

    Args:
        tmp_path (Path): pytest's built-in fixture for a temporary directory.

    Yields:
        Path: The path to the base directory containing the dummy DICOM data.
    """
    base_dir: Path = tmp_path / "Data"
    base_dir.mkdir()

    # Patient 1: More than 5 files (to test middle image selection)
    patient1_folder: Path = base_dir / "QIN LUNG CT 1"
    patient1_series_folder: Path = patient1_folder / "series1"
    patient1_series_folder.mkdir(parents=True)
    for i in range(7): # Create 7 DICOM files
        filename: Path = patient1_series_folder / f"image{i:03d}.dcm"
        ds = FileDataset(str(filename), {}, file_meta=pydicom.FileMetaDataset())
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.PatientName = f"Patient1_{i}"
        ds.Rows = 10
        ds.Columns = 10
        # Create a simple gradient image for pixel_array
        pixel_data: np.ndarray = np.arange(100, dtype=np.uint16).reshape(10, 10)
        ds.PixelData = pixel_data.tobytes()
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0 # Typical for CT
        ds.WindowCenter = -500 # Example window
        ds.WindowWidth = 1000 # Example window
        dcmwrite(str(filename), ds)

    # Patient 2: Less than 5 files (to test selection of all available images)
    patient2_folder: Path = base_dir / "QIN LUNG CT 2"
    patient2_series_folder: Path = patient2_folder / "series1"
    patient2_series_folder.mkdir(parents=True)
    for i in range(3): # Create 3 DICOM files
        filename = patient2_series_folder / f"image{i:03d}.dcm"
        ds = FileDataset(str(filename), {}, file_meta=pydicom.FileMetaDataset())
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        ds.PatientName = f"Patient2_{i}"
        ds.Rows = 10
        ds.Columns = 10
        pixel_data = np.full((10, 10), i * 100, dtype=np.uint16) # Different constant values
        ds.PixelData = pixel_data.tobytes()
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        ds.WindowCenter = -500
        ds.WindowWidth = 1000
        dcmwrite(str(filename), ds)

    # Patient 3: No DICOM files (to test handling of empty folders)
    patient3_folder: Path = base_dir / "QIN LUNG CT 3"
    patient3_folder.mkdir()

    yield base_dir # Yield the base directory path for tests to use

    # Cleanup is handled automatically by pytest's tmp_path fixture.


# --- Tests for load_dicom_image function ---
def test_load_dicom_image_success(dummy_dicom_data_path: Path) -> None:
    """
    Test that `load_dicom_image` successfully loads a valid DICOM file
    and normalizes pixel values to the [0, 1] range.

    Given a path to a valid DICOM file,
    When `load_dicom_image` is called,
    Then it should return a NumPy array with float32 dtype, correct shape,
    and pixel values normalized between 0 and 1.
    """
    file_path: Path = dummy_dicom_data_path / "QIN LUNG CT 1" / "series1" / "image001.dcm"
    img: Optional[np.ndarray] = load_dicom_image(file_path)

    assert img is not None, f"Image should be loaded successfully, but got None for {file_path}."
    assert isinstance(img, np.ndarray), "Loaded image should be a numpy array."
    assert img.dtype == np.float32, f"Image dtype should be float32, but got {img.dtype}."
    assert img.shape == (10, 10), f"Image shape should be (10, 10), but got {img.shape}."
    assert np.min(img) >= 0.0, f"Minimum pixel value should be >= 0.0, but got {np.min(img)}."
    assert np.max(img) <= 1.0, f"Maximum pixel value should be <= 1.0, but got {np.max(img)}."

def test_load_dicom_image_non_dicom_file(tmp_path: Path) -> None:
    """
    Test that `load_dicom_image` returns None for a non-DICOM file.

    Given a path to a file that is not a valid DICOM format,
    When `load_dicom_image` is called,
    Then it should return None.
    """
    non_dicom_file: Path = tmp_path / "not_a_dicom.txt"
    non_dicom_file.write_text("This is not a DICOM file.")
    img: Optional[np.ndarray] = load_dicom_image(non_dicom_file)
    assert img is None, f"Should return None for a non-DICOM file, but got {type(img)}."

def test_load_dicom_image_missing_file() -> None:
    """
    Test that `load_dicom_image` returns None for a non-existent file path.

    Given a path to a file that does not exist,
    When `load_dicom_image` is called,
    Then it should return None.
    """
    img: Optional[np.ndarray] = load_dicom_image(Path("non_existent_file.dcm"))
    assert img is None, "Should return None for a missing file."

def test_load_dicom_image_uniform_pixels(tmp_path: Path) -> None:
    """
    Test that `load_dicom_image` handles uniform pixel data (where img_max == img_min
    after HU windowing logic) by returning an array of zeros.

    Given a DICOM file with uniform pixel values that fall into a zero-width HU window,
    When `load_dicom_image` is called,
    Then it should return a NumPy array filled with zeros.
    """
    file_path: Path = tmp_path / "uniform.dcm"
    ds = FileDataset(str(file_path), {}, file_meta=pydicom.FileMetaDataset())
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.PatientName = "Uniform^Test"
    ds.Rows = 5
    ds.Columns = 5
    pixel_data: np.ndarray = np.full((5, 5), 500, dtype=np.uint16) # All pixels same value
    ds.PixelData = pixel_data.tobytes()
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.WindowCenter = 500
    ds.WindowWidth = 1 # Small window to force img_max == img_min after windowing logic
    dcmwrite(str(file_path), ds)

    img: Optional[np.ndarray] = load_dicom_image(file_path)
    assert img is not None, "Image should not be None even for uniform pixels."
    assert np.all(img == 0.0), f"Uniform image should be normalized to all zeros, but got min={np.min(img)}, max={np.max(img)}."


# --- Tests for LungCTWithGaussianDataset class ---
def test_dataset_initialization_success(dummy_dicom_data_path: Path) -> None:
    """
    Test that `LungCTWithGaussianDataset` initializes successfully with valid data,
    and correctly selects the middle 5 images from folders with more than 5 files,
    and all images from folders with 5 or fewer files.

    Given a dummy dataset with patient folders containing varying numbers of DICOM files,
    When `LungCTWithGaussianDataset` is initialized,
    Then it should correctly identify and store the paths to the selected images,
    resulting in the expected total count.
    """
    transform = get_transforms() # Dummy transform for init
    dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, transform=transform, num_images_per_folder=5)

    # Patient 1 had 7 files, should select 5 middle ones (indices 2,3,4,5,6).
    # Patient 2 had 3 files, should select all 3 (indices 0,1,2).
    # Total expected images: 5 + 3 = 8
    expected_total_images: int = 8
    assert len(dataset) == expected_total_images, \
        f"Dataset should contain {expected_total_images} images, but found {len(dataset)}."

    # Check that paths are correctly stored and correspond to the middle selection logic
    assert all(isinstance(p, str) for p in dataset.image_paths), "All image paths should be strings."
    # Verify specific files are included based on the selection logic
    assert "image002.dcm" in dataset.image_paths[0], "First selected image from Patient 1 should be image002.dcm."
    assert "image006.dcm" in dataset.image_paths[4], "Fifth selected image from Patient 1 should be image006.dcm."
    assert "image000.dcm" in dataset.image_paths[5], "First selected image from Patient 2 should be image000.dcm."
    assert "image002.dcm" in dataset.image_paths[7], "Last selected image from Patient 2 should be image002.dcm."

def test_dataset_initialization_no_dicom_files(tmp_path: Path) -> None:
    """
    Test that `LungCTWithGaussianDataset` raises a ValueError if no valid DICOM files
    are found in the specified base directory.

    Given a base directory that contains no DICOM files or only unreadable ones,
    When `LungCTWithGaussianDataset` is initialized,
    Then it should raise a ValueError with a specific message.
    """
    empty_base_dir: Path = tmp_path / "empty_data"
    empty_base_dir.mkdir()
    (empty_base_dir / "QIN LUNG CT 1").mkdir() # Create an empty patient folder

    transform = get_transforms()
    with pytest.raises(ValueError, match="No DICOM images found or loaded"):
        LungCTWithGaussianDataset(empty_base_dir, transform=transform)

def test_dataset_getitem_valid_image(dummy_dicom_data_path: Path) -> None:
    """
    Test that `__getitem__` returns a Gaussian noise tensor and a correctly
    transformed image tensor for a valid DICOM file.

    Given a dataset initialized with valid DICOM files,
    When `__getitem__` is called for a specific index,
    Then it should return two float32 tensors (noise and image) of the expected shape
    and with the image normalized to the [-1, 1] range.
    """
    image_size: Tuple[int, int] = (64, 64)
    transform = get_transforms(image_size=image_size)
    # Select 1 image for simplicity to ensure __getitem__ works on a known single path
    dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, transform=transform, num_images_per_folder=1)
    
    noise: torch.Tensor
    image: torch.Tensor
    noise, image = dataset[0]

    assert isinstance(noise, torch.Tensor), "Returned noise should be a torch.Tensor."
    assert isinstance(image, torch.Tensor), "Returned image should be a torch.Tensor."
    assert noise.shape == (1, *image_size), f"Noise shape should be (1, {image_size[0]}, {image_size[1]}), but got {noise.shape}."
    assert image.shape == (1, *image_size), f"Image shape should be (1, {image_size[0]}, {image_size[1]}), but got {image.shape}."
    assert noise.dtype == torch.float32, f"Noise dtype should be float32, but got {noise.dtype}."
    assert image.dtype == torch.float32, f"Image dtype should be float32, but got {image.dtype}."
    
    # Check image normalization range using torch.min and torch.max
    assert torch.min(image) >= -1.0, f"Minimum pixel value for image should be >= -1.0, but got {torch.min(image)}."
    assert torch.max(image) <= 1.0, f"Maximum pixel value for image should be <= 1.0, but got {torch.max(image)}."

def test_dataset_getitem_failed_image_loading(mocker: Any) -> None:
    """
    Test that `__getitem__` returns a black image tensor when `load_dicom_image` fails
    for a specific file, ensuring batch consistency.

    Given a mocked `load_dicom_image` that always returns None,
    When `__getitem__` is called,
    Then it should return a noise tensor and an all-black (zero-filled) image tensor
    of the expected shape.
    """
    # Mock `load_dicom_image` to always return None, simulating a loading failure.
    mocker.patch('Synthetic_Image_Generator.dataset.load_dicom_image', return_value=None)

    # Create a dummy base_dir and a dummy file to ensure the dataset finds a path,
    # even though loading will be mocked to fail.
    dummy_base_dir: Path = Path("dummy_data_for_mock")
    dummy_base_dir.mkdir(exist_ok=True)
    (dummy_base_dir / "QIN LUNG CT 1").mkdir(exist_ok=True)
    dummy_file: Path = dummy_base_dir / "QIN LUNG CT 1" / "dummy.dcm"
    dummy_file.touch() # Create a dummy file so dataset initialization finds a path

    image_size: Tuple[int, int] = (64, 64)
    transform = get_transforms(image_size=image_size)
    
    # Initialize dataset. We force `image_paths` to include the dummy file
    # because the dataset's `__init__` tries to read DICOM headers, which isn't mocked here.
    dataset = LungCTWithGaussianDataset(dummy_base_dir, transform=transform, num_images_per_folder=1, image_size=image_size)
    dataset.image_paths = [str(dummy_file)] # Force it to have one path for __getitem__

    noise: torch.Tensor
    image: torch.Tensor
    noise, image = dataset[0]

    assert isinstance(noise, torch.Tensor), "Returned noise should be a torch.Tensor."
    assert isinstance(image, torch.Tensor), "Returned image should be a torch.Tensor."
    assert image.shape == (1, *image_size), f"Image shape should be (1, {image_size[0]}, {image_size[1]}), but got {image.shape}."
    assert torch.all(image == 0.0), "Image should be an all-black tensor (zeros) when loading fails."

    # Clean up the dummy directory created for this test.
    shutil.rmtree(dummy_base_dir)
