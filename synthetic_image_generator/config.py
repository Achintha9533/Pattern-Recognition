# config.py
from pathlib import Path
import torch
import torchvision.transforms as T

# Base directory containing patient subfolders with DICOM files
# IMPORTANT: Adjust this path to your actual data directory
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")

# Desired output image size (height, width)
image_size = (96, 96)

# Optimizer learning rates
G_LR = 1e-4

# === Checkpoint settings ===
# Directory to save model weights (and load from for pre-trained)
checkpoint_dir = Path("./checkpoints")
generator_checkpoint_path = checkpoint_dir / "generator_final.pth"

# Google Drive file ID for your weights
GOOGLE_DRIVE_FILE_ID = '1TzXuOGzpt1eR4wxE6GahCX8Ig9ia0ErN'

# === Image preprocessing transform ===
# Resize images, convert to tensor, normalize pixel values to [-1, 1]
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])

# Transform for FID calculation (to [0, 255] for saving as images)
# This denormalizes from [-1, 1] to [0, 1] and then PIL converts to [0, 255]
fid_transform = T.Compose([
    T.Normalize(mean=[-1.0], std=[2.0]),
    T.ToPILImage(),
])

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyper-parameters
time_embed_dim = 256