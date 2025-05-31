import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import pydicom
from PIL import Image

# === Step 1: Load up to 55 DICOM images per folder ===
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")
images = []

for i in range(1, 48):  # folders 1 to 47
    folder = base_dir / f"QIN LUNG CT {i}"
    file_paths = []
    for root, _, file_names in os.walk(folder):
        for file_name in file_names:
            if file_name.lower().endswith(".dcm"):
                file_paths.append(os.path.join(root, file_name))

    sorted_file_paths = []
    for file_path in file_paths:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            inst_num = getattr(ds, 'InstanceNumber', None)
            if inst_num is not None:
                sorted_file_paths.append((inst_num, file_path))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    sorted_file_paths.sort(key=lambda x: x[0])
    sorted_file_paths = [x[1] for x in sorted_file_paths[:55]]  # Up to 55 slices

    for file_path in sorted_file_paths:
        try:
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)
            images.append(img)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

print(f"Loaded {len(images)} slices from all patients (max 55 per patient).")

# === Step 2: Preprocess images (resize and normalize) ===
processed_images = []
resize_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

for img in images:
    img_tensor = resize_transform(img)
    img_tensor = 2*(img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()) - 1
    processed_images.append(img_tensor)

all_images = torch.stack(processed_images)  # shape (N, 1, 64, 64)

# === Step 3: Define Dataset and DataLoader ===
class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        target = self.images[idx]
        noise = torch.randn_like(target)
        return noise, target

dataset = ImageDataset(all_images)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)