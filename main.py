import torch
from dataset import LungCTWithGaussianDataset
from transform import get_transform
from model import CNF_UNet
from train import train
from generate import generate
from visualize import visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")
    transform = get_transform()
    dataset = LungCTWithGaussianDataset(base_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    model = CNF_UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, dataloader, epochs=30)
    noise = torch.randn(4, 1, 64, 64).to(device)
    generated = generate(model, noise)
    visualize(generated)