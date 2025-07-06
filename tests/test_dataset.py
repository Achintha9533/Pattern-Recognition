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

# Import modules from your package
from Synthetic_Image_Generator.dataset import load_dicom_image, LungCTWithGaussianDataset
from Synthetic_Image_Generator.transforms import get_transforms # Need this for dataset init

# --- Fixtures for creating dummy DICOM data ---
@pytest.fixture(scope="function")
def dummy_dicom_data_path(tmp_path):
    """
    Fixture to create a temporary directory with dummy DICOM files for testing.
    Creates a structure like:
    tmp_path/
        QIN LUNG CT 1/
            series1/
                image001.dcm
                image002.dcm
                ... (7 files)
        QIN LUNG CT 2/
            series1/
                image001.dcm
                ... (3 files)
    """
    base_dir = tmp_path / "Data"
    base_dir.mkdir()

    # Patient 1: More than 5 files
    patient1_folder = base_dir / "QIN LUNG CT 1"
    patient1_series_folder = patient1_folder / "series1"
    patient1_series_folder.mkdir(parents=True)
    for i in range(7): # Create 7 DICOM files
        filename = patient1_series_folder / f"image{i:03d}.dcm"
        ds = FileDataset(filename, {}, file_meta=pydicom.FileMetaDataset())
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'  # CT Image Storage
        ds.PatientName = f"Patient1_{i}"
        ds.Rows = 10
        ds.Columns = 10
        # Create a simple gradient image for pixel_array
        pixel_data = np.arange(100, dtype=np.uint16).reshape(10, 10)
        ds.PixelData = pixel_data.tobytes()
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0 # Typical for CT
        ds.WindowCenter = -500 # Example window
        ds.WindowWidth = 1000 # Example window
        dcmwrite(filename, ds)

    # Patient 2: Less than 5 files
    patient2_folder = base_dir / "QIN LUNG CT 2"
    patient2_series_folder = patient2_folder / "series1"
    patient2_series_folder.mkdir(parents=True)
    for i in range(3): # Create 3 DICOM files
        filename = patient2_series_folder / f"image{i:03d}.dcm"
        ds = FileDataset(filename, {}, file_meta=pydicom.FileMetaDataset())
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
        dcmwrite(filename, ds)

    # Patient 3: No DICOM files
    patient3_folder = base_dir / "QIN LUNG CT 3"
    patient3_folder.mkdir()

    yield base_dir # Yield the base directory path

    # Cleanup (pytest tmp_path handles this, but explicit for clarity)
    # shutil.rmtree(tmp_path)


# --- Tests for load_dicom_image function ---
def test_load_dicom_image_success(dummy_dicom_data_path):
    """
    Test that load_dicom_image successfully loads a valid DICOM file
    and normalizes pixel values to [0, 1].
    """
    file_path = dummy_dicom_data_path / "QIN LUNG CT 1" / "series1" / "image001.dcm"
    img = load_dicom_image(file_path)

    assert img is not None, "Image should be loaded successfully."
    assert isinstance(img, np.ndarray), "Loaded image should be a numpy array."
    assert img.dtype == np.float32, "Image dtype should be float32."
    assert img.shape == (10, 10), "Image shape should match original pixel array."
    assert np.min(img) >= 0.0 and np.max(img) <= 1.0, "Pixel values should be normalized to [0, 1]."

def test_load_dicom_image_non_dicom_file(tmp_path):
    """
    Test that load_dicom_image returns None for a non-DICOM file.
    """
    non_dicom_file = tmp_path / "not_a_dicom.txt"
    non_dicom_file.write_text("This is not a DICOM file.")
    img = load_dicom_image(non_dicom_file)
    assert img is None, "Should return None for a non-DICOM file."

def test_load_dicom_image_missing_file():
    """
    Test that load_dicom_image returns None for a non-existent file.
    """
    img = load_dicom_image(Path("non_existent_file.dcm"))
    assert img is None, "Should return None for a missing file."

def test_load_dicom_image_uniform_pixels(tmp_path):
    """
    Test that load_dicom_image handles uniform pixel data (img_max == img_min)
    by returning an array of zeros.
    """
    file_path = tmp_path / "uniform.dcm"
    ds = FileDataset(file_path, {}, file_meta=pydicom.FileMetaDataset())
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    ds.PatientName = "Uniform^Test"
    ds.Rows = 5
    ds.Columns = 5
    pixel_data = np.full((5, 5), 500, dtype=np.uint16) # All pixels same value
    ds.PixelData = pixel_data.tobytes()
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    ds.WindowCenter = 500
    ds.WindowWidth = 1 # Small window to force img_max == img_min after windowing logic
    dcmwrite(file_path, ds)

    img = load_dicom_image(file_path)
    assert img is not None
    assert np.all(img == 0.0), "Uniform image should be normalized to all zeros."


# --- Tests for LungCTWithGaussianDataset class ---
def test_dataset_initialization_success(dummy_dicom_data_path):
    """
    Test that LungCTWithGaussianDataset initializes successfully with valid data,
    and correctly selects the middle 5 images from folders with more than 5 files,
    and all images from folders with 5 or fewer files.
    """
    transform = get_transforms() # Dummy transform for init
    dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, transform=transform, num_images_per_folder=5)

    # Patient 1 had 7 files, should select 5 middle ones.
    # Patient 2 had 3 files, should select all 3.
    # Total expected: 5 + 3 = 8
    assert len(dataset) == 8, "Dataset should contain the correct number of selected images."
    # Check that paths are correctly stored
    assert all(isinstance(p, str) for p in dataset.image_paths)
    assert "image002.dcm" in dataset.image_paths[0] # Middle 5 from 7 files (0-6) are 2,3,4,5,6
    assert "image006.dcm" in dataset.image_paths[4]
    assert "image000.dcm" in dataset.image_paths[5] # First from 3 files (0-2) is 0
    assert "image002.dcm" in dataset.image_paths[7]

def test_dataset_initialization_no_dicom_files(tmp_path):
    """
    Test that LungCTWithGaussianDataset raises ValueError if no DICOM files are found.
    """
    empty_base_dir = tmp_path / "empty_data"
    empty_base_dir.mkdir()
    (empty_base_dir / "QIN LUNG CT 1").mkdir() # Create an empty patient folder

    transform = get_transforms()
    with pytest.raises(ValueError, match="No DICOM images found or loaded"):
        LungCTWithGaussianDataset(empty_base_dir, transform=transform)

def test_dataset_getitem_valid_image(dummy_dicom_data_path):
    """
    Test that __getitem__ returns noise and a correctly transformed image
    for a valid DICOM file.
    """
    transform = get_transforms(image_size=(64, 64))
    dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, transform=transform, num_images_per_folder=1) # Select 1 for simplicity
    
    noise, image = dataset[0]

    assert isinstance(noise, torch.Tensor)
    assert isinstance(image, torch.Tensor)
    assert noise.shape == (1, 64, 64) # 1 channel, 64x64
    assert image.shape == (1, 64, 64)
    assert noise.dtype == torch.float32
    assert image.dtype == torch.float32
    # Check image normalization range
    assert torch.min(image) >= -1.0 and torch.max(image) <= 1.0

def test_dataset_getitem_failed_image_loading(mocker):
    """
    Test that __getitem__ returns a black image when load_dicom_image fails.
    Mocks load_dicom_image to simulate failure.
    """
    # Mock load_dicom_image to always return None
    mocker.patch('Synthetic_Image_Generator.dataset.load_dicom_image', return_value=None)

    # Create a dummy base_dir with at least one path, even if it's invalid
    dummy_base_dir = Path("dummy_data_for_mock")
    dummy_base_dir.mkdir(exist_ok=True)
    (dummy_base_dir / "QIN LUNG CT 1").mkdir(exist_ok=True)
    dummy_file = dummy_base_dir / "QIN LUNG CT 1" / "dummy.dcm"
    dummy_file.touch() # Create a dummy file so dataset finds a path

    transform = get_transforms(image_size=(64, 64))
    # Temporarily modify image_paths to include the dummy file for the test
    # This is a bit hacky, but necessary since the dataset init tries to read DICOM headers.
    # A more robust solution for testing dataset init would be to mock pydicom.dcmread too.
    dataset = LungCTWithGaussianDataset(dummy_base_dir, transform=transform, num_images_per_folder=1)
    dataset.image_paths = [str(dummy_file)] # Force it to have one path

    noise, image = dataset[0]

    assert isinstance(noise, torch.Tensor)
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 64, 64)
    assert torch.all(image == 0.0), "Image should be an all-black tensor (zeros)."

    # Clean up dummy directory
    shutil.rmtree(dummy_base_dir)