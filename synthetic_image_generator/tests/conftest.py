# tests/conftest.py
import pytest
import torch
import pydicom
from pydicom.dataset import Dataset as DicomDataset
from pydicom.filebase import DicomFileLike
import numpy as np
from pathlib import Path
from pydicom.uid import CTImageStorage # Import CTImageStorage

@pytest.fixture(scope="session")
def mock_dicom_dir(tmp_path_factory):
    """
    GIVEN: access to pytest's temporary path factory
    WHEN: a temporary directory structure mimicking the QIN LUNG CT dataset is created,
          populated with dummy DICOM files for two patients, each with 10 images
    THEN: a Path object to the base of this mock DICOM directory is returned
    """
    base_dir = tmp_path_factory.mktemp("mock_data")
    
    # Create 2 patient folders
    for i in range(1, 3):
        patient_folder = base_dir / f"QIN LUNG CT {i}"
        scan_folder = patient_folder / "scan_1"
        scan_folder.mkdir(parents=True, exist_ok=True)
        
        # Create 10 dummy DICOM files in each scan folder
        for j in range(10):
            file_path = scan_folder / f"image_{j}.dcm"
            
            # Create a basic pydicom dataset
            ds = DicomDataset()
            ds.file_meta = DicomDataset()
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            ds.file_meta.MediaStorageSOPClassUID = CTImageStorage # Add this for proper file meta
            ds.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid() # Ensure unique UID
            ds.file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
            ds.file_meta.ImplementationVersionName = "PYDICOM_TEST"
            ds.file_meta.SourceApplicationEntityTitle = "PYDICOMAPP"

            # Add basic DICOM image tags
            ds.Modality = "CT"
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
            ds.PatientName = f"Patient {i}"
            ds.PatientID = f"ID{i}"
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.StudyID = "1"
            ds.SeriesNumber = "1"
            ds.InstanceNumber = str(j+1)
            
            # Generate pixel data as uint16, then apply rescale later to get HU
            # This will create raw pixel values in range [0, 2000].
            # With RescaleIntercept = -1000 and RescaleSlope = 1, the HU will be [-1000, 1000].
            pixel_array = (np.random.randint(0, 2000, (96, 96), dtype=np.uint16)) 
            ds.PixelData = pixel_array.tobytes()
            ds.Rows, ds.Columns = 96, 96
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            # Add crucial Pixel Module Attributes for pydicom to interpret PixelData correctly
            ds.SamplesPerPixel = 1 # For grayscale
            ds.PhotometricInterpretation = 'MONOCHROME2' # For grayscale images (black=0, white=max)
            ds.BitsAllocated = 16
            ds.BitsStored = 12 # Assume 12-bit stored for typical CT
            ds.HighBit = 11 # HighBit should be BitsStored - 1
            ds.PixelRepresentation = 0 # Usually 0 for unsigned raw pixel data (input to rescale)

            # Define Rescale Intercept and Slope to convert pixel values to Hounsfield Units
            ds.RescaleIntercept = -1000.0 
            ds.RescaleSlope = 1.0 
            
            # Add Image Position/Orientation/Spacing
            ds.ImagePositionPatient = [0.0, 0.0, float(j)] # Vary Z position
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.PixelSpacing = [1.0, 1.0]

            # Save the dummy DICOM file
            ds.save_as(file_path, write_like_original=False)
            
    return base_dir

@pytest.fixture
def mock_generator():
    """
    GIVEN: a request for a mock generator
    WHEN: a simple PyTorch module is defined that mimics the CNF_UNet's forward pass behavior
    THEN: an instance of this mock generator, capable of returning a tensor of the same shape as its input, is provided
    """
    class MockCNF_UNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Define parameters that might be accessed by the training loop
            self.param1 = torch.nn.Parameter(torch.tensor(1.0))
        def forward(self, x, t):
            # Return a tensor of the expected shape for generated images
            # Assuming output is single channel, 96x96, normalized [-1, 1]
            return torch.randn_like(x) # Return noise of the same shape as input
    return MockCNF_UNet()