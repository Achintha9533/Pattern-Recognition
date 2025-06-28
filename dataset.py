import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom

# Assuming image_size and transform are imported or defined globally if this were standalone
# from transform import transform, image_size, load_dicom_image

# Base directory containing patient subfolders with DICOM files
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")
image_size = (64, 64) # Redefined for clarity if dataset.py is standalone, otherwise import
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])
def load_dicom_image(file_path): # Redefined for clarity, otherwise import from transform
    try:
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / (img_max - img_min) if img_max != img_min else np.zeros_like(img)
        return img
    except Exception as e:
        warnings.warn(f"Failed to load {file_path}: {e}")
        return None


# === Custom Dataset class ===
class LungCTWithGaussianDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.image_paths = []

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
                        # Only check if pydicom can read the file header, not pixel_array
                        # Pixel array reading is done in load_dicom_image for efficiency
                        pydicom.dcmread(f_path)
                        current_folder_dicom_files.append(f_path)
                    except Exception as e:
                        warnings.warn(f"Skipping unreadable DICOM file: {f_path} - {e}")
                        continue
            
            # --- Select 58 middle images from each folder ---
            num_files_in_folder = len(current_folder_dicom_files)
            num_to_select = 58

            if num_files_in_folder <= num_to_select:
                # If there are 58 or fewer files, take all of them
                self.image_paths.extend(current_folder_dicom_files)
            else:
                # Calculate start and end index to select 58 middle images
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
            image = torch.zeros(1, *image_size)
        else:
            if img.ndim == 2:
                pass # Already 2D
            else:
                img = img.squeeze() # Remove singleton dimensions

            # Apply transform. load_dicom_image now returns float32 [0,1]
            image = self.transform(img)

        # Generate noise batch_size, 1, H, W
        noise = torch.randn_like(image)
        return noise, image

# === Instantiate dataset and dataloader ===
# These lines would typically be in the main execution script, but included here for completeness
# when demonstrating the dataset usage.
# dataset = LungCTWithGaussianDataset(base_dir, transform=transform)
# dataloader = DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,
#     num_workers=0,  # Set to 0 for macOS compatibility
#     pin_memory=True
# )