import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Import custom modules
from dataset import LungCTWithGaussianDataset # Assuming image_size is set here or passed
from models import CNF_UNet, Discriminator
from train import train_hybrid
from generate import generate

# === Settings ===
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")
image_size = (64, 64) # Keep consistent across modules
hu_window = (-1000, 400)

# === Image preprocessing transform ===
transform_post_hu = T.Compose([
    T.ToPILImage(),
    T.Resize(image_size),
    T.ToTensor(),
])

# === Setup Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == "__main__":
    # === Instantiate dataset and dataloader ===
    dataset = LungCTWithGaussianDataset(
        base_dir,
        transform=transform_post_hu,
        hu_window=hu_window,
        num_patients_limit=25 # For initial testing, keep lower; increase for real training
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16, # Reduced batch size for GAN, requires more VRAM
        shuffle=True,
        num_workers=0, # Set to 0 for macOS compatibility and debugging
        pin_memory=True
    )

    print(f"Total images loaded: {len(dataset)}")

    # Quick test to load a batch and check shapes
    for noise, image in dataloader:
        print(f"Noise batch shape (example): {noise.shape}")
        print(f"Image batch shape (example): {image.shape}")
        break

    # === Initialize Models ===
    model_g = CNF_UNet(time_emb_dim=64).to(device)
    model_d = Discriminator(in_channels=1, img_size=image_size[0]).to(device) # Pass img_size to D

    # === Setup Optimizers and Schedulers ===
    optimizer_g = torch.optim.Adam(model_g.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=1e-4)

    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=300)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=300)

    # === Start Hybrid Training ===
    train_hybrid(model_g, model_d, dataloader, optimizer_g, optimizer_d,
                 scheduler_g, scheduler_d, device, epochs=50)

    # === Sample generation and Visualization ===
    print("\nGenerating sample images (after hybrid training)...")
    noise_sample = torch.randn(8, 1, *image_size).to(device)
    generated_images = generate(model_g, noise_sample, steps=200, device=device)

    # Display generated images
    plt.figure(figsize=(12, 6))
    for i in range(min(8, generated_images.shape[0])):
        plt.subplot(2, 4, i + 1)
        display_img = (generated_images[i, 0].cpu().numpy() + 1) / 2
        plt.imshow(display_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Generated {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle("Generated Lung CT Images (CNF-GAN Hybrid with Attention)")
    plt.show()

    # Optional: Display some real images for comparison
    real_images_batch = next(iter(dataloader))[1] # Get a fresh batch from the dataloader
    plt.figure(figsize=(12, 6))
    for i in range(min(8, real_images_batch.shape[0])):
        plt.subplot(2, 4, i + 1)
        display_real_img = (real_images_batch[i, 0].cpu().numpy() + 1) / 2
        plt.imshow(display_real_img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Real Image {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle("Real Lung CT Images (for comparison)")
    plt.show()