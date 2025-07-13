# dataset.py
import os
from pathlib import Path
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import warnings

# === Load and normalize a single DICOM image ===
def load_dicom_image(file_path):
    try:
        # Read DICOM file using pydicom, force=True to handle non-standard headers
        ds = pydicom.dcmread(file_path, force=True)

        # Convert pixel data to float32 numpy array
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply Rescale Slope and Intercept if they exist in the DICOM header
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Normalize Hounsfield Units (HU) to [0, 1] based on common CT window
        # Assuming a window of [-1000, 400] HU for lung CT, adjust based on your dataset
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
        warnings.warn(f"Failed to load {file_path}: {e}")
        return None

# === Custom Dataset class ===
class LungCTWithGaussianDataset(Dataset):
    def __init__(self, base_dir, transform=None, image_size=(96, 96)):
        self.transform = transform
        self.image_paths = []
        self.image_size = image_size

        # Iterate over patient folders (QIN LUNG CT 1 to 47)
        for i in range(1, 48):
            folder = base_dir / f"QIN LUNG CT {i}"
            current_folder_dicom_files = []
            for root, _, files in os.walk(folder):
                dicom_files = sorted([
                    os.path.join(root, f) for f in files if f.lower().endswith(".dcm")
                ])
                for f_path in dicom_files:
                    try:
                        # Also use force=True when scanning for readable files
                        pydicom.dcmread(f_path, force=True)
                        current_folder_dicom_files.append(f_path)
                    except Exception as e:
                        warnings.warn(f"Skipping unreadable DICOM file: {f_path} - {e}")
                        continue
            
            # --- Select 5 middle images from each folder ---
            num_files_in_folder = len(current_folder_dicom_files)
            num_to_select = 5 

            if num_files_in_folder <= num_to_select:
                # If there are 5 or fewer files, take all of them
                self.image_paths.extend(current_folder_dicom_files)
            else:
                # Calculate start and end index to select 5 middle images
                start_index = (num_files_in_folder - num_to_select) // 2
                end_index = start_index + num_to_select
                self.image_paths.extend(current_folder_dicom_files[start_index:end_index])

        if not self.image_paths:
            raise ValueError(f"No DICOM images found or loaded from {base_dir}. Check path and file permissions.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = load_dicom_image(img_path)

        if img is None:
            # Return a black image if loading failed
            image = torch.zeros(1, *self.image_size)
        else:
            if img.ndim == 2:
                # Add a channel dimension if it's just HxW
                image = torch.from_numpy(img).unsqueeze(0) 
            else:
                image = torch.from_numpy(img).squeeze() # Ensure it's 1-channel if it has other dimensions
                if image.ndim == 2: # After squeeze, ensure it's HxW, then add channel
                    image = image.unsqueeze(0)

            # Apply transform. load_dicom_image now returns float32 [0,1]
            # The transform expects a PIL Image or Tensor, and converts to [-1, 1]
            image = self.transform(image) # Already a tensor, transform expects it
            
            # Ensure the output image has a channel dimension (C, H, W)
            if image.ndim == 2:
                image = image.unsqueeze(0)


        # Generate noise batch_size, 1, H, W
        noise = torch.randn_like(image)
        return noise, image