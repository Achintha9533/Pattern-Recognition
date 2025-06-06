import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import LungCTWithGaussianDataset
from transforms import transform
from model import CNF_UNet
from train import train
from generate import generate
from visualize import plot_samples

base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup dataset and dataloader
dataset = LungCTWithGaussianDataset(base_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

# Model + optimizer
model = CNF_UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
train(model, dataloader, optimizer, device, epochs=30)

# Generate and visualize
noise = torch.randn(4, 1, 64, 64).to(device)
generated = generate(model, noise, device)
plot_samples(generated, noise)
