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

    This fixture creates a simulated dataset structure within a temporary directory,
    mimicking the expected input for `LungCTWithGaussianDataset`. It generates patient
    folders containing dummy DICOM files with varying counts to thoroughly test
    image selection logic (e.g., `NUM_IMAGES_PER_FOLDER`). Each DICOM file contains
    a simple, predictable pixel array to facilitate verification of loading and
    normalization.

    Args:
        tmp_path (Path): pytest's built-in fixture providing a unique temporary directory
                         for the test function.

    Returns:
        Generator[Path, None, None]: A generator that yields the path to the base directory
                                     containing the dummy DICOM data. This path can then
                                     be used to initialize the `LungCTWithGaussianDataset`.

    Potential Exceptions Raised:
        None directly by this fixture, but underlying file operations could raise OSError.

    Example of Usage:
    ```python
    def test_something(dummy_dicom_data_path):
        dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, transform=None)
        # ... use dataset ...
    ```

    Relationships with Other Functions:
        * `dcmwrite`: Used to write the dummy DICOM files.
        * `FileDataset`: Used to construct the in-memory DICOM dataset before writing.

    Explanation of the Theory:
        This fixture adheres to the principle of test isolation by providing a
        self-contained and predictable test environment for dataset-related tests.
        It simulates a real-world data structure without relying on actual patient data,
        ensuring tests are repeatable and independent.

    References for the Theory:
        * Pytest fixtures: https://docs.pytest.org/en/stable/how-to/fixtures.html
        * DICOM standard: https://www.dicomstandard.org/
    """
    base_dir: Path = tmp_path / "Data"
    base_dir.mkdir()

    # Patient 1: 3 images
    (base_dir / "patient001").mkdir()
    for i in range(3):
        filepath = base_dir / "patient001" / f"image{i:03d}.dcm"
        # Create a dummy DICOM dataset
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        # Add required DICOM tags
        ds.PatientName = "Test^Patient"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.Rows = 10
        ds.Columns = 10
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        # Create a simple pixel array
        ds.PixelData = np.full((10, 10), i * 100, dtype=np.int16).tobytes()
        dcmwrite(str(filepath), ds)

    # Patient 2: 5 images
    (base_dir / "patient002").mkdir()
    for i in range(5):
        filepath = base_dir / "patient002" / f"image{i:03d}.dcm"
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
        ds.PatientName = "Test^Patient2"
        ds.StudyInstanceUID = pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.Modality = "CT"
        ds.Rows = 10
        ds.Columns = 10
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.RescaleIntercept = -1024
        ds.RescaleSlope = 1
        ds.PixelData = np.full((10, 10), (i + 1) * 50, dtype=np.int16).tobytes()
        dcmwrite(str(filepath), ds)

    # Patient 3: 1 image (to test folders with fewer than num_images_per_folder)
    (base_dir / "patient003").mkdir()
    filepath = base_dir / "patient003" / f"image001.dcm"
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2' # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(filepath, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "Test^Patient3"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows = 10
    ds.Columns = 10
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.RescaleIntercept = -1024
    ds.RescaleSlope = 1
    ds.PixelData = np.full((10, 10), 200, dtype=np.int16).tobytes()
    dcmwrite(str(filepath), ds)

    yield base_dir

    # Cleanup is handled automatically by tmp_path fixture

@pytest.fixture(scope="function")
def malformed_dicom_path(tmp_path: Path) -> Path:
    """
    Fixture to create a temporary directory with a malformed DICOM file for testing.

    This fixture creates a file with a `.dcm` extension but with invalid content
    (e.g., not a valid DICOM structure) to test the robustness of the `load_dicom_image`
    function when encountering corrupted or non-DICOM files.

    Args:
        tmp_path (Path): pytest's built-in fixture for a temporary directory.

    Returns:
        Path: The path to the malformed DICOM file.

    Potential Exceptions Raised:
        None directly by this fixture.

    Example of Usage:
    ```python
    def test_loading_malformed_dicom(malformed_dicom_path):
        result = load_dicom_image(malformed_dicom_path)
        assert result is None
    ```

    Relationships with Other Functions:
        * Used by `test_load_dicom_image_malformed_file` to provide an invalid input.

    Explanation of the Theory:
        This fixture supports negative testing, which is crucial for verifying that
        functions handle unexpected or invalid inputs gracefully, preventing crashes
        or undefined behavior.
    """
    malformed_dir = tmp_path / "malformed_data"
    malformed_dir.mkdir()
    malformed_file = malformed_dir / "malformed.dcm"
    malformed_file.write_text("This is not a DICOM file.") # Write invalid content
    return malformed_file

# --- Tests for load_dicom_image function ---

def test_load_dicom_image_valid_file(dummy_dicom_data_path: Path) -> None:
    """
    Test that `load_dicom_image` successfully loads a valid DICOM file and returns
    a normalized NumPy array.

    Given a path to a valid DICOM file created by `dummy_dicom_data_path`,
    When `load_dicom_image` is called with this path,
    Then it should return a NumPy array with the expected shape, data type (float32),
    and pixel values normalized within the [0, 1] range based on the configured HU window.

    Args:
        dummy_dicom_data_path (Path): Path to the temporary directory containing dummy DICOM data.

    Returns:
        None

    Potential Exceptions Raised:
        AssertionError: If the returned image is not a NumPy array, has incorrect shape/dtype,
                        or if pixel values are outside the expected normalized range.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_dicom_data_path` fixture for test data.
        * Tests the core functionality of `dataset.load_dicom_image`.

    Explanation of the Theory:
        This test verifies the image loading and normalization pipeline, ensuring
        that raw DICOM pixel data is correctly transformed into a format suitable
        for model input (normalized float32 array). Normalization is critical
        for stable neural network training.
    """
    file_path = dummy_dicom_data_path / "patient001" / "image000.dcm"
    img_np = load_dicom_image(file_path)

    assert img_np is not None, "Image should be loaded successfully."
    assert isinstance(img_np, np.ndarray), "Returned object should be a numpy array."
    assert img_np.dtype == np.float32, "Image data type should be float32."
    assert img_np.shape == (10, 10), "Image shape should match the DICOM dimensions."
    assert np.all(img_np >= 0.0) and np.all(img_np <= 1.0), "Pixel values should be normalized to [0, 1]."
    # More specific value check for the dummy data after normalization (HU -1024 -> 0)
    # The dummy image data is all 0, so after rescale_intercept (-1024) and slope (1), it's -1024 HU.
    # Given img_min=-1000, img_max=400, (HU - img_min) / (img_max - img_min) = (-1024 - (-1000)) / (400 - (-1000))
    # = -24 / 1400. This should be clipped to 0.
    assert np.isclose(img_np.min(), 0.0), "Minimum pixel value should be 0.0 after normalization and clipping."
    assert np.isclose(img_np.max(), 0.0), "Maximum pixel value should be 0.0 after normalization and clipping."


def test_load_dicom_image_malformed_file(malformed_dicom_path: Path, caplog: Any) -> None:
    """
    Test that `load_dicom_image` handles malformed DICOM files gracefully by returning `None`
    and logging a warning.

    Given a path to a malformed DICOM file created by `malformed_dicom_path`,
    When `load_dicom_image` is called with this path,
    Then it should return `None` and log a warning message indicating the loading failure.

    Args:
        malformed_dicom_path (Path): Path to the temporary malformed DICOM file.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None

    Potential Exceptions Raised:
        AssertionError: If the returned value is not `None` or if the expected
                        warning message is not logged.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `malformed_dicom_path` fixture for test data.
        * Tests the error handling of `dataset.load_dicom_image`.

    Explanation of the Theory:
        Robust error handling is vital for real-world applications where data
        can be corrupted or incomplete. This test ensures that the loading function
        fails gracefully for invalid inputs, preventing the entire application
        from crashing and providing informative feedback.
    """
    with caplog.at_level(logging.WARNING):
        img_np = load_dicom_image(malformed_dicom_path)

    assert img_np is None, "Malformed image loading should return None."
    assert "Failed to load DICOM file" in caplog.text, "Expected warning log not found."


def test_load_dicom_image_file_not_found(tmp_path: Path, caplog: Any) -> None:
    """
    Test that `load_dicom_image` handles non-existent file paths gracefully by returning `None`
    and logging a warning.

    Given a path to a non-existent file,
    When `load_dicom_image` is called with this path,
    Then it should return `None` and log a warning message indicating the file was not found.

    Args:
        tmp_path (Path): pytest's built-in fixture for a temporary directory.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None

    Potential Exceptions Raised:
        AssertionError: If the returned value is not `None` or if the expected
                        warning message is not logged.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the error handling of `dataset.load_dicom_image` for non-existent files.

    Explanation of the Theory:
        Similar to handling malformed files, ensuring proper behavior when files
        are missing is essential for application stability and user feedback.
        This test confirms that `load_dicom_image` doesn't crash but instead
        signals the failure appropriately.
    """
    non_existent_file = tmp_path / "non_existent.dcm"
    with caplog.at_level(logging.WARNING):
        img_np = load_dicom_image(non_existent_file)

    assert img_np is None, "Non-existent file loading should return None."
    assert "Failed to load DICOM file" in caplog.text, "Expected warning log for file not found not found."

# --- Tests for LungCTWithGaussianDataset class ---

def test_dataset_initialization_and_length(dummy_dicom_data_path: Path) -> None:
    """
    Test that `LungCTWithGaussianDataset` initializes correctly and reports the expected
    number of images based on `num_images_per_folder`.

    Given a dummy DICOM data directory with multiple patient folders and images,
    When `LungCTWithGaussianDataset` is initialized with a specific `num_images_per_folder`,
    Then the dataset's length (`__len__`) should be the sum of `num_images_per_folder`
    for each patient folder, or the actual number of images if fewer are available.

    Args:
        dummy_dicom_data_path (Path): Path to the temporary directory containing dummy DICOM data.

    Returns:
        None

    Potential Exceptions Raised:
        AssertionError: If the dataset length does not match the expected value.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_dicom_data_path` fixture for test data.
        * Tests the `__init__` and `__len__` methods of `LungCTWithGaussianDataset`.

    Explanation of the Theory:
        This test confirms that the dataset properly discovers and counts image paths
        according to the specified sampling strategy, which is fundamental for
        correct data loading during training or evaluation.
    """
    # Patient 1 has 3 images, Patient 2 has 5 images, Patient 3 has 1 image.
    # If num_images_per_folder = 2:
    # Patient 1: takes 2
    # Patient 2: takes 2
    # Patient 3: takes 1 (fewer than 2 available)
    # Total expected: 2 + 2 + 1 = 5
    dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, num_images_per_folder=2)
    assert len(dataset) == 5, f"Expected 5 images, but got {len(dataset)}."

    # Test with num_images_per_folder larger than any folder's content
    dataset_all = LungCTWithGaussianDataset(dummy_dicom_data_path, num_images_per_folder=100)
    # Total expected: 3 (patient1) + 5 (patient2) + 1 (patient3) = 9
    assert len(dataset_all) == 9, f"Expected 9 images, but got {len(dataset_all)}."

def test_dataset_getitem_output_types_and_shapes(dummy_dicom_data_path: Path, caplog: Any) -> None:
    """
    Test that `LungCTWithGaussianDataset.__getitem__` returns a noise tensor and an image tensor
    of the expected types, shapes, and pixel ranges, including handling of failed DICOM loads.

    Given a `LungCTWithGaussianDataset` initialized with dummy data,
    When an item is retrieved using `__getitem__`,
    Then it should return two `torch.Tensor` objects: a Gaussian noise tensor and
    a real image tensor. Both tensors should have the correct shape (1, H, W)
    and `float32` data type. The real image tensor's pixel values should be
    normalized to [-1, 1], and the noise tensor should be standard normal.
    Also, verifies graceful handling (black placeholder) when `load_dicom_image` fails.

    Args:
        dummy_dicom_data_path (Path): Path to the temporary directory containing dummy DICOM data.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None

    Potential Exceptions Raised:
        AssertionError: If output types, shapes, or pixel ranges are incorrect.
        RuntimeError: If `load_dicom_image` is mocked to raise an unexpected error.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_dicom_data_path` fixture for test data.
        * Uses `load_dicom_image` internally to load real images.
        * Uses `get_transforms` for image preprocessing.
        * Tests the `__getitem__` method of `LungCTWithGaussianDataset`.

    Explanation of the Theory:
        This test validates the core data retrieval mechanism of the dataset, ensuring
        that data is presented to the model in the correct format and range, and that
        it can gracefully handle data loading failures by providing a consistent
        placeholder, preventing downstream errors during training.
    """
    image_size: Tuple[int, int] = (64, 64)
    transform = get_transforms(image_size=image_size)

    dataset = LungCTWithGaussianDataset(dummy_dicom_data_path, transform=transform, num_images_per_folder=1)
    dataset.image_paths = [str(dummy_dicom_data_path / "patient001" / "image000.dcm")] # Force one known path

    noise, image = dataset[0]

    assert isinstance(noise, torch.Tensor), "Returned noise should be a torch.Tensor."
    assert isinstance(image, torch.Tensor), "Returned image should be a torch.Tensor."
    assert image.shape == (1, *image_size), f"Image shape should be (1, {image_size}), but got {image.shape}."
    assert noise.shape == (1, *image_size), f"Noise shape should be (1, {image_size}), but got {noise.shape}."
    assert image.dtype == torch.float32, "Image data type should be float32."
    assert noise.dtype == torch.float32, "Noise data type should be float32."

    # Check pixel value range for normalized image
    assert image.min() >= -1.0 and image.max() <= 1.0, "Image pixel values should be normalized to [-1, 1]."
    # Check noise distribution (approximately for random noise)
    assert noise.std() > 0.1, "Noise tensor should not be all zeros (standard deviation should be > 0.1)."

    # Test handling of DICOM loading failure:
    # Temporarily replace the load_dicom_image function with a mock that returns None
    original_load_dicom = dataset.load_dicom_image
    try:
        dataset.load_dicom_image = MagicMock(return_value=None)
        with caplog.at_level(logging.WARNING):
            noise_failed, image_failed = dataset[0]

        assert torch.all(image_failed == 0.0), "Failed DICOM load should result in a black placeholder image."
        assert "Using black placeholder" in caplog.text, "Expected warning for black placeholder not logged."
    finally:
        # Restore the original function
        dataset.load_dicom_image = original_load_dicom


def test_dataset_image_dimension_handling(tmp_path: Path, caplog: Any) -> None:
    """
    Test that `LungCTWithGaussianDataset.__getitem__` correctly handles 2D and 3D images
    (e.g., (1, H, W)) and logs warnings for unexpected dimensions.

    Given a dummy DICOM data directory with images of various dimensions (2D, 3D with singleton channel),
    When `__getitem__` is called,
    Then it should ensure the image is 2D (H, W) or (C, H, W) where C is 1 before transformations,
    and log warnings for unexpected input dimensions while attempting to squeeze.

    Args:
        tmp_path (Path): pytest's built-in fixture for a temporary directory.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None

    Potential Exceptions Raised:
        AssertionError: If the final image tensor shape is incorrect.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `load_dicom_image` to simulate different input image dimensions.
        * Tests the image dimension handling logic within `LungCTWithGaussianDataset.__getitem__`.

    Explanation of the Theory:
        This test addresses data heterogeneity, ensuring that the dataset can
        process images from various sources that might have different channel
        configurations (e.g., (H,W) vs (1,H,W)) and standardize them for the model.
    """
    dummy_base_dir: Path = tmp_path / "dummy_data_dim_test"
    dummy_base_dir.mkdir(exist_ok=True)
    (dummy_base_dir / "patient_2d").mkdir(exist_ok=True)
    (dummy_base_dir / "patient_3d_1ch").mkdir(exist_ok=True)
    (dummy_base_dir / "patient_3d_multi_ch").mkdir(exist_ok=True) # Will trigger warning

    # Create dummy DICOM files returning different array shapes from load_dicom_image
    def mock_load_dicom_2d(file_path):
        return np.random.rand(10, 10).astype(np.float32) # (H, W)

    def mock_load_dicom_3d_1ch(file_path):
        return np.random.rand(1, 10, 10).astype(np.float32) # (1, H, W)

    def mock_load_dicom_3d_multi_ch(file_path):
        return np.random.rand(3, 10, 10).astype(np.float32) # (C, H, W) - unexpected for CT

    image_size: Tuple[int, int] = (64, 64)
    transform = get_transforms(image_size=image_size)

    # Test 2D image
    dataset_2d = LungCTWithGaussianDataset(dummy_base_dir, transform=transform, num_images_per_folder=1)
    dataset_2d.image_paths = [str(dummy_base_dir / "patient_2d" / "img.dcm")]
    with patch('Synthetic_Image_Generator.dataset.load_dicom_image', side_effect=mock_load_dicom_2d):
        _, image_2d = dataset_2d[0]
        assert image_2d.shape == (1, *image_size), f"2D image incorrect shape: {image_2d.shape}"
        assert not caplog.records, "No warnings expected for 2D image."
        caplog.clear() # Clear logs for next part

    # Test 3D (1 channel) image
    dataset_3d_1ch = LungCTWithGaussianDataset(dummy_base_dir, transform=transform, num_images_per_folder=1)
    dataset_3d_1ch.image_paths = [str(dummy_base_dir / "patient_3d_1ch" / "img.dcm")]
    with patch('Synthetic_Image_Generator.dataset.load_dicom_image', side_effect=mock_load_dicom_3d_1ch):
        _, image_3d_1ch = dataset_3d_1ch[0]
        assert image_3d_1ch.shape == (1, *image_size), f"3D (1ch) image incorrect shape: {image_3d_1ch.shape}"
        assert not caplog.records, "No warnings expected for 3D (1 channel) image."
        caplog.clear()

    # Test 3D (multi-channel) image - should trigger warning
    dataset_3d_multi_ch = LungCTWithGaussianDataset(dummy_base_dir, transform=transform, num_images_per_folder=1)
    dataset_3d_multi_ch.image_paths = [str(dummy_base_dir / "patient_3d_multi_ch" / "img.dcm")]
    with patch('Synthetic_Image_Generator.dataset.load_dicom_image', side_effect=mock_load_dicom_3d_multi_ch):
        with caplog.at_level(logging.WARNING):
            _, image_3d_multi_ch = dataset_3d_multi_ch[0]
            assert image_3d_multi_ch.shape == (1, *image_size), f"3D (multich) image incorrect shape: {image_3d_multi_ch.shape}"
            assert "Unexpected image dimensions" in caplog.text, "Expected warning for unexpected dimensions not logged."
            caplog.clear()

def test_dataset_no_data_found(tmp_path: Path, caplog: Any) -> None:
    """
    Test that `LungCTWithGaussianDataset` handles cases where no DICOM data is found
    by raising a ValueError and logging a critical message.

    Given an empty base directory or a directory with no valid DICOM files,
    When `LungCTWithGaussianDataset` is initialized,
    Then it should raise a `ValueError` and log a critical message indicating
    that no image paths were found.

    Args:
        tmp_path (Path): pytest's built-in fixture for a temporary directory.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None

    Potential Exceptions Raised:
        ValueError: Expected exception if no data is found.
        AssertionError: If the ValueError is not raised or the critical log is missing.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the error handling during initialization of `LungCTWithGaussianDataset`.

    Explanation of the Theory:
        This test confirms that the dataset constructor provides clear feedback
        and prevents further execution if its fundamental requirement (data availability)
        is not met, improving debugging and application robustness.
    """
    empty_base_dir: Path = tmp_path / "empty_data"
    empty_base_dir.mkdir()

    with pytest.raises(ValueError) as excinfo:
        with caplog.at_level(logging.CRITICAL):
            LungCTWithGaussianDataset(empty_base_dir)

    assert "No image paths found" in str(excinfo.value), "Expected ValueError message not found."
    assert "No image paths found in the base directory" in caplog.text, "Expected critical log not found."