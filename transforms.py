import numpy as np
import torch
import torchvision.transforms as T
import pydicom
import warnings

# Desired output image size (height, width) - Defined here for transform
image_size = (64, 64)

# === Image preprocessing transform ===
# Resize images, convert to tensor, normalize pixel values to [-1, 1]
transform = T.Compose([
    T.ToPILImage(),               # Convert numpy array to PIL Image for transformations
    T.Resize(image_size),         # Resize to 64x64
    T.ToTensor(),                 # Convert PIL Image to torch tensor (C x H x W), scales [0,255] to [0,1] for uint8, or [0,1] for float
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# === Load and normalize a single DICOM image ===
def load_dicom_image(file_path):
    try:
        # Read DICOM file using pydicom
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array.astype(np.float32)  # Convert pixel data to float32 numpy array

        img_min = img.min()
        img_max = img.max()
        if img_max == img_min: # Handle cases of uniform images to prevent division by zero
            img = np.zeros_like(img)
        else:
            # Normalize to [0, 1] directly as float32
            img = (img - img_min) / (img_max - img_min)

        return img

    except Exception as e:
        warnings.warn(f"Failed to load {file_path}: {e}")
        return None