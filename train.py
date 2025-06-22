import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

# Import generate function from its module
from generate import generate

def train_hybrid(model_g, model_d, dataloader, optimizer_g, optimizer_d,
                 scheduler_g, scheduler_d, device, epochs=20, lambda_adv=0.05):
    """
    Trains the CNF_UNet generator and Discriminator in a hybrid GAN-Flow Matching setup.

    Args:
        model_g (torch.nn.Module): The CNF_UNet generator model.
        model_d (torch.nn.Module): The Discriminator model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        optimizer_g (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_d (torch.optim.Optimizer): Optimizer for the discriminator.
        scheduler_g (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for G.
        scheduler_d (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for D.
        device (str or torch.device): The device to train on.
        epochs (int): Number of training epochs.
        lambda_adv (float): Weight for the adversarial loss in the generator's total loss.
    """
    model_g.train()
    model_d.train()
    print(f"Starting hybrid training on {device}...")

    criterion_gan = nn.BCELoss()
    real_label = 1.
    fake_label = 0.

    for epoch in range(epochs):
        epoch_g_loss_total = 0.0
        epoch_g_loss_fm = 0.0
        epoch_g_loss_adv = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for z0, x1 in pbar:
            z0, x1 = z0.to(device), x1.to(device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()

            # 1. Train with real images
            label_real = torch.full((x1.size(0), 1), real_label, dtype=torch.float32, device=device)
            output_real = model_d(x1)
            err_d_real = criterion_gan(output_real, label_real)
            err_d_real.backward()

            # 2. Train with fake images from Generator
            with torch.no_grad(): # Detach generation from generator's graph for D training
                noise_for_d = torch.randn_like(z0)
                fake_images = generate(model_g, noise_for_d, steps=50, device=device) # Use fewer steps for D training speed

            label_fake = torch.full((fake_images.size(0), 1), fake_label, dtype=torch.float32, device=device)
            output_fake = model_d(fake_images.detach()) # Detach fake images
            err_d_fake = criterion_gan(output_fake, label_fake)
            err_d_fake.backward()
            
            err_d = err_d_real + err_d_fake
            optimizer_d.step()
            scheduler_d.step()
            epoch_d_loss += err_d.item()

            # --- Train Generator (CNF_UNet) ---
            optimizer_g.zero_grad()

            # 1. Flow Matching Loss (MSE)
            t = torch.rand(z0.size(0), device=device) # t is now continuous [0, 1]
            xt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1
            v_target = (x1 - z0)
            v_pred = model_g(xt, t)
            loss_flow_matching = F.mse_loss(v_pred, v_target)
            
            # 2. Adversarial Loss (Generator wants D to think fakes are real)
            noise_for_g = torch.randn_like(z0)
            gen_images_for_g_adv = generate(model_g, noise_for_g, steps=50, device=device) # Generate for G's loss

            label_real_for_g = torch.full((gen_images_for_g_adv.size(0), 1), real_label, dtype=torch.float32, device=device)
            output_g_adv = model_d(gen_images_for_g_adv) # Discriminator's output on generator's samples
            loss_g_adv = criterion_gan(output_g_adv, label_real_for_g)

            # Combined Generator Loss
            loss_g_total = loss_flow_matching + lambda_adv * loss_g_adv
            loss_g_total.backward()
            optimizer_g.step()
            scheduler_g.step()

            epoch_g_loss_total += loss_g_total.item()
            epoch_g_loss_fm += loss_flow_matching.item()
            epoch_g_loss_adv += loss_g_adv.item()

            pbar.set_postfix({
                'G_Total': f'{loss_g_total.item():.4f}',
                'G_FM': f'{loss_flow_matching.item():.4f}',
                'G_Adv': f'{loss_g_adv.item():.4f}',
                'D_Loss': f'{err_d.item():.4f}'
            })

        avg_g_total = epoch_g_loss_total / len(dataloader)
        avg_g_fm = epoch_g_loss_fm / len(dataloader)
        avg_g_adv = epoch_g_loss_adv / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        print(f"Epoch {epoch+1}: G_Total_Loss={avg_g_total:.4f} (FM={avg_g_fm:.4f}, Adv={avg_g_adv:.4f}), D_Loss={avg_d_loss:.4f}")