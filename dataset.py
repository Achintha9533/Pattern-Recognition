import os
from pathlib import Path
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import warnings

class LungCTWithGaussianDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        for i in range(1, 48):
            folder = base_dir / f"QIN LUNG CT {i}"
            images = []
            for root, _, files in os.walk(folder):
                dicom_files = sorted([
                    os.path.join(root, f) for f in files if f.lower().endswith(".dcm")
                ])
                images.extend(dicom_files[:50])
            self.image_paths.extend(images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = self.load_dicom_image(img_path)
        if img is None:
            image = torch.zeros(1, 64, 64)
        else:
            if img.ndim == 2:
                pass
            else:
                img = img.squeeze()
            if self.transform:
                try:
                    image = self.transform(img)
                except Exception as e:
                    warnings.warn(f"Transform failed on {img_path}: {e}")
                    image = torch.zeros(1, 64, 64)
            else:
                image = torch.tensor(img).unsqueeze(0).float() / 255.0
        noise = torch.randn_like(image)
        return noise, image

    def load_dicom_image(self, file_path):
        try:
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            return img
        except Exception as e:
            warnings.warn(f"Failed to load {file_path}: {e}")
            return None