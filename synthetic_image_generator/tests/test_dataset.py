# tests/test_dataset.py
import torch
import numpy as np
import pydicom
# Import specific UIDs and FileMetaDataset for clarity and correctness
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage
from pydicom.dataset import FileMetaDataset # Explicitly import FileMetaDataset

# Adjust import path
from dataset import load_dicom_image, LungCTWithGaussianDataset
from config import transform


def test_load_dicom_image(tmp_path):
    """
    GIVEN: a temporary path and a dummy DICOM file created with known pixel data (HU value 400)
    WHEN: the `load_dicom_image` function is called with the path to the dummy DICOM file
    THEN: the function should return a NumPy array of float32,
          and the pixel values in the array should be correctly normalized to 1.0 (within a small tolerance)
    """
    file_path = tmp_path / "test.dcm"
    
    # Create a dummy DICOM file with a known pixel array
    ds = pydicom.Dataset()
    
    # CORRECTED: Initialize file_meta as a FileMetaDataset instance directly
    ds.file_meta = FileMetaDataset() 
    
    # Use ExplicitVRLittleEndian for TransferSyntaxUID as it's a common explicit VR syntax
    # This must be present and correctly set in the file_meta for the DICM prefix.
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian 
    
    # Add required File Meta Information Elements for proper DICOM parsing
    # These are crucial for pydicom to recognize it as a valid DICOM file header.
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage # Use a valid SOP Class UID for CT images
    ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid() # Generate a unique UID
    ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid() # Unique UID for the implementation
    ds.file_meta.ImplementationVersionName = "PYDICOM_TEST" # Name of the implementation
    ds.file_meta.SourceApplicationEntityTitle = "PYDICOMAPP" # Source Application
    
    # Add common DICOM image pixel module attributes
    # These are essential for pydicom to correctly interpret the PixelData.
    ds.Modality = "CT" # Modality of the image
    ds.SOPClassUID = CTImageStorage # Same as MediaStorageSOPClassUID for the main image
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID # Same as MediaStorageSOPInstanceUID
    ds.PatientName = "Test^Patient"
    ds.PatientID = "12345"
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyID = "1"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    
    # Pixel Data characteristics
    ds.SamplesPerPixel = 1 # For grayscale images (1 channel)
    ds.PhotometricInterpretation = "MONOCHROME2" # Standard for grayscale CT (black=0, white=max_pixel_value)
    ds.PixelRepresentation = 1 # 0 for unsigned integer, 1 for signed integer (CT values can be negative, e.g., air is -1000 HU)
    ds.BitsAllocated = 16 # Common for CT images (e.g., storing 12-bit or 16-bit raw data)
    ds.BitsStored = 12    # Number of bits actually used for pixel data (e.g., 12-bit CT data)
    ds.HighBit = ds.BitsStored - 1 # HighBit should be (BitsStored - 1)
    
    # Create pixel data as int16, as this is typical for medical images that use RescaleSlope/Intercept
    # The value 1400, with RescaleIntercept=-1000 and RescaleSlope=1, will result in a HU of 400.
    pixel_array_int16 = np.full((50, 50), 1400, dtype=np.int16) 
    ds.PixelData = pixel_array_int16.tobytes()

    # Rescale Slope and Intercept for converting raw pixel values to Hounsfield Units (HU)
    ds.RescaleIntercept = -1000.0 # Define as float for consistency (typical for CT)
    ds.RescaleSlope = 1.0         # Define as float for consistency (typical for CT)
    
    # Add Image Position/Orientation/Spacing for a more complete DICOM
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    
    ds.Rows, ds.Columns = 50, 50
    ds.is_little_endian = True
    ds.is_implicit_VR = False # ExplicitVRLittleEndian means it's not implicit VR
    
    ds.save_as(file_path)

    # Load and check normalization
    img = load_dicom_image(file_path)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    
    # Normalization check: (HU_value - HU_min) / (HU_max - HU_min)
    # The pixel array's HU value is 400 (from 1400 * 1.0 + (-1000)).
    # If the `load_dicom_image` function normalizes HU values from -1000 to 400 to [0, 1]:
    # Normalized value = (400 - (-1000)) / (400 - (-1000)) = 1400 / 1400 = 1.0
    assert np.isclose(img.max(), 1.0) 
    assert np.isclose(img.min(), 1.0) # Min should also be 1.0 if all pixels are 400

def test_lung_ct_dataset(mock_dicom_dir):
    """
    GIVEN: a mock DICOM directory containing dummy patient scan data
    WHEN: a LungCTWithGaussianDataset is initialized with this directory, a transform, and an image size,
          and an item is retrieved from the dataset
    THEN: the dataset should report the correct number of samples (10),
          and the retrieved noise and image tensors should have the expected shape (1, 96, 96)
          and the image values should be normalized within the [-1, 1] range
    """
    
    dataset = LungCTWithGaussianDataset(
        base_dir=mock_dicom_dir, 
        transform=transform, 
        image_size=(96, 96)
    )
    
    # We created 2 folders, selecting 5 images from each
    # The conftest.py fixture creates 10 images per scan folder, 2 patient folders.
    # If LungCTWithGaussianDataset selects 5 images from each patient, then 2*5 = 10.
    assert len(dataset) == 10
    
    # Test __getitem__
    noise, image = dataset[0]
    
    # Check output shapes and types
    assert isinstance(noise, torch.Tensor)
    assert isinstance(image, torch.Tensor)
    assert noise.shape == (1, 96, 96)
    assert image.shape == (1, 96, 96)
    
    # Check that image values are normalized to [-1, 1] by the transform
    assert image.min() >= -1.0
    assert image.max() <= 1.0