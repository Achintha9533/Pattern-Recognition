import os
from pathlib import Path
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import warnings

# === Settings ===
# Base directory containing patient subfolders with DICOM files
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")

# Desired output image size (height, width)
image_size = (64, 64)

# === Image preprocessing transform ===
# Resize images to `image_size`, convert to tensor, normalize pixel values to [-1, 1]
transform = T.Compose([
    T.ToPILImage(),               # Convert numpy array to PIL Image for transformations
    T.Resize(image_size),         # Resize to 64x64
    T.ToTensor(),                 # Convert PIL Image to torch tensor (C x H x W), scales [0,255] to [0,1]
    T.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
])

# === Load and normalize a single DICOM image ===
def load_dicom_image(file_path):
    try:
        # Read DICOM file using pydicom
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array.astype(np.float32)  # Convert pixel data to float32 numpy array

        # Normalize pixel values to range [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Convert normalized image to uint8 in [0,255] for compatibility with PIL transforms
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img

    except Exception as e:
        # Warn if loading fails and return None
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
            images = []

            # Walk folder to find all DICOM files, sorted alphabetically
            for root, _, files in os.walk(folder):
                dicom_files = sorted([
                    os.path.join(root, f) for f in files if f.lower().endswith(".dcm")
                ])
                # Take only first 20 DICOM images per folder to limit dataset size
                images.extend(dicom_files[:50])

            self.image_paths.extend(images)

    def __len__(self):
        # Return total number of images
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image file path at index
        img_path = self.image_paths[idx]

        # Load DICOM image as numpy array
        img = load_dicom_image(img_path)

        if img is None:
            # If loading fails, create a dummy zero tensor of shape [1, H, W]
            image = torch.zeros(1, *image_size)
        else:
            # Ensure image is 2D (H x W)
            if img.ndim == 2:
                pass  # Already correct shape
            else:
                img = img.squeeze()

            if self.transform:
                try:
                    # Apply preprocessing transform (resize, normalize, to tensor)
                    image = self.transform(img)
                except Exception as e:
                    warnings.warn(f"Transform failed on {img_path}: {e}")
                    image = torch.zeros(1, *image_size)
            else:
                # If no transform provided, convert numpy to tensor and scale to [0,1]
                image = torch.tensor(img).unsqueeze(0).float() / 255.0

        # Generate a matching Gaussian noise tensor of the same shape as the image
        noise = torch.randn_like(image)

        # Return tuple: (noise tensor, processed image tensor)
        return noise, image

# === Instantiate dataset and dataloader ===
dataset = LungCTWithGaussianDataset(base_dir, transform=transform)

# DataLoader to create batches and shuffle data
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,      # Set >0 for multiprocessing if on Linux/macOS (0 for Windows or debugging)
    pin_memory=True     # Improves data transfer speed to GPU if using CUDA
)

# === Quick test to load a batch ===
if __name__ == "__main__":
    for noise, image in dataloader:
        # Print shapes to verify batch loading
        print(f"Noise batch shape: {noise.shape}")   # Expected: [batch_size, 1, 64, 64]
        print(f"Image batch shape: {image.shape}")   # Expected: [batch_size, 1, 64, 64]
        break

import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    # Print total number of images loaded in the dataset
    print(f"Total images loaded: {len(dataset)}")  # Should be ~940 (47 patients × ~20 slices)

    # Iterate over one batch from the dataloader
    for noise, image in dataloader:
        # Shape checks
        assert noise.shape == image.shape, f"Mismatch: noise {noise.shape}, image {image.shape}"
        assert noise.shape[1:] == (1, 64, 64), "Unexpected shape; expected [B, 1, 64, 64]"

        print(f"Noise batch shape: {noise.shape}")
        print(f"Image batch shape: {image.shape}")

        # Flatten tensors for histogram plotting
        flat_noise = noise.view(-1).cpu().numpy()
        flat_image = image.view(-1).cpu().numpy()

        # Create histogram plots
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(flat_image, bins=50, color='blue', alpha=0.7)
        plt.title('CT Image Pixel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Count')

        plt.subplot(1, 2, 2)
        plt.hist(flat_noise, bins=50, color='red', alpha=0.7)
        plt.title('Gaussian Noise Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

        # Display sample images and noise
        plt.figure(figsize=(10, 4))
        for i in range(4):
            plt.subplot(2, 4, i + 1)
            plt.imshow(image[i, 0].cpu(), cmap='gray')
            plt.title("CT Image")
            plt.axis('off')

            plt.subplot(2, 4, i + 5)
            plt.imshow(noise[i, 0].cpu(), cmap='gray')
            plt.title("Noise")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Only inspect the first batch
        break
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# === UNet Block ===
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# === CNF-UNet Model ===
class CNF_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetBlock(2, 32)   # input: noise + time
        self.down2 = UNetBlock(32, 64)
        self.mid = UNetBlock(64, 64)
        self.up1 = UNetBlock(64 + 64, 32)
        self.up2 = UNetBlock(32 + 32, 16)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x, t):
        t = t.view(-1, 1, 1, 1).expand(x.size(0), 1, x.size(2), x.size(3))
        xt = torch.cat([x, t], dim=1)

        d1 = self.down1(xt)  # [B, 32, 64, 64]
        d2 = self.down2(F.avg_pool2d(d1, 2))  # [B, 64, 32, 32]
        m = self.mid(F.avg_pool2d(d2, 2))     # [B, 64, 16, 16]

        u1 = F.interpolate(m, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up1(torch.cat([u1, d2], dim=1))  # [B, 32, 32, 32]

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = self.up2(torch.cat([u2, d1], dim=1))  # [B, 16, 64, 64]

        return self.out(u2)

# === Setup Device, Model, Optimizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNF_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# === Training Loop ===
def train(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for z0, x1 in tqdm(dataloader):
            z0, x1 = z0.to(device), x1.to(device)

            # Sample time uniformly
            t = torch.rand(z0.size(0), device=device)

            # Interpolate point at time t
            xt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1

            # Target velocity
            v_target = (x1 - z0)

            # Model predicts velocity field at (xt, t)
            v_pred = model(xt, t)

            # MSE loss for flow matching
            loss = F.mse_loss(v_pred, v_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(dataloader):.6f}")

@torch.no_grad()
def generate(model, z0, steps=100):
    model.eval()
    z = z0.clone().to(device)
    for i in range(steps):
        t = torch.tensor(i / steps, device=device).repeat(z.shape[0])
        v = model(z, t)
        z = z + v / steps
    return z

if __name__ == "__main__":
    # Assume `dataloader` is defined and gives (noise, image)
    train(model, dataloader, epochs=30)

    # Sample generation
    noise = torch.randn(4, 1, 64, 64).to(device)
    generated = generate(model, noise)

    # Visualize results
    import matplotlib.pyplot as plt
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(generated[i, 0].cpu(), cmap='gray')
        plt.axis('off')
    plt.show()