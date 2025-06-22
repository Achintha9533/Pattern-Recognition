import os
from pathlib import Path
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import warnings

# === Load and normalize a single DICOM image (Improved with HU windowing) ===
def load_dicom_image(file_path, hu_window=(-1000, 400)):
    try:
        ds = pydicom.dcmread(file_path, force=True) # force=True to handle potential issues

        if not hasattr(ds, 'PixelData'):
            warnings.warn(f"Skipping {file_path}: No PixelData attribute.")
            return None
        
        img = ds.pixel_array.astype(np.float32)

        # Apply RescaleSlope and RescaleIntercept to convert to Hounsfield Units (HU)
        if hasattr(ds, 'RescaleIntercept') and hasattr(ds, 'RescaleSlope'):
            intercept = float(ds.RescaleIntercept)
            slope = float(ds.RescaleSlope)
            img = img * slope + intercept
        else:
            warnings.warn(f"DICOM {file_path} missing RescaleIntercept/Slope. Assuming HU.")

        # Apply HU windowing and clip values
        min_hu, max_hu = hu_window
        img = np.clip(img, min_hu, max_hu)

        # Normalize to [0, 1] based on HU window
        img = (img - min_hu) / (max_hu - min_hu + 1e-8) # Add epsilon for stability

        # Normalize to [-1, 1] (standard for many generative models)
        img = 2.0 * img - 1.0

        # Ensure image is 2D (H x W)
        if img.ndim == 3 and img.shape[0] == 1:
            img = img.squeeze(0)
        elif img.ndim != 2:
            warnings.warn(f"Skipping {file_path}: Unexpected image dimension {img.ndim}. Expected 2 or 3 (with channel first).")
            return None

        return img # Returns a numpy array in [-1, 1] range

    except Exception as e:
        warnings.warn(f"Failed to load or process {file_path}: {e}")
        return None

# === Custom Dataset class ===
class LungCTWithGaussianDataset(Dataset):
    def __init__(self, base_dir, transform=None, hu_window=(-1000, 400), num_patients_limit=47):
        self.transform = transform
        self.hu_window = hu_window
        self.image_paths = []

        # Iterate over patient folders
        for i in range(1, num_patients_limit + 1): # Use num_patients_limit here
            folder = base_dir / f"QIN LUNG CT {i}"
            if not folder.is_dir():
                warnings.warn(f"Skipping non-existent patient folder: {folder}")
                continue

            dicom_files = sorted([
                p.resolve() for p in folder.rglob("*.dcm") if p.is_file()
            ])
            # For demonstration, limit slices per patient to keep dataset manageable
            self.image_paths.extend(dicom_files[:40]) # Increased from 20 to 40 slices per patient

        if not self.image_paths:
            raise RuntimeError(f"No DICOM images found in {base_dir}. Please check path and data.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_np = load_dicom_image(img_path, self.hu_window)

        if img_np is None:
            # Fallback: return a zero tensor if image loading fails
            image = torch.zeros(1, 64, 64, dtype=torch.float32) # Assume image_size=(64,64)
        else:
            if self.transform:
                try:
                    image = self.transform(img_np)
                except Exception as e:
                    warnings.warn(f"Transform failed on {img_path}: {e}")
                    image = torch.zeros(1, 64, 64, dtype=torch.float32) # Assume image_size=(64,64)
            else:
                # If no transform, convert processed numpy array (already -1 to 1) to tensor
                image = torch.from_numpy(img_np).unsqueeze(0).float()

        # Ensure output is 1 channel and correct size even if transform fails
        if image.ndim == 2: # handle case where ToTensor might not add channel for 2D PIL
             image = image.unsqueeze(0)
        # Assuming image_size is available (e.g., passed or imported from config)
        # For this standalone file, let's hardcode 64x64 or pass it
        # For now, let's ensure it's resized if not already.
        if image.shape[2:] != (64, 64): # Assuming image_size is (64, 64)
            image = T.Resize((64, 64))(image) # Ensure final size even if transform was skipped/failed

        noise = torch.randn_like(image) # Generate Gaussian noise for the corresponding image

        return noise, image