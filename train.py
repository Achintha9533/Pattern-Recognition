import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Assuming these are imported from a main script or model.py
# from model import CNF_UNet, Discriminator
# from dataset import DataLoader # Or directly import DataLoader

# Global training parameters (would typically be passed or imported)
# G_LR = 1e-4
# D_LR = 1e-4
# lambda_gan = 0.1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(generator_model, discriminator_model, dataloader, epochs=50, lambda_gan=0.01):
    generator_model.train()
    discriminator_model.train()

    # Assuming optimizers and criterion are passed or initialized in main
    # For demonstration, initializing here as placeholders if running train.py alone
    # optimizer_gen = torch.optim.Adam(generator_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # optimizer_disc = torch.optim.Adam(discriminator_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    # criterion_gan = nn.BCEWithLogitsLoss()

    gen_flow_losses = []
    gen_gan_losses = []
    disc_real_losses = []
    disc_fake_losses = []
    disc_total_losses = []

    for epoch in range(epochs):
        epoch_gen_flow_loss = 0.0
        epoch_gen_gan_loss = 0.0
        epoch_disc_real_loss = 0.0
        epoch_disc_fake_loss = 0.0
        epoch_disc_total_loss = 0.0

        for i, (z0, x1_real) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            device = x1_real.device # Ensure device is correctly set for current batch
            z0, x1_real = z0.to(device), x1_real.to(device)
            batch_size = x1_real.size(0)

            # Smoothed labels for stability
            real_labels = torch.ones(batch_size, 1, 1, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, 1, 1, device=device) * 0.1

            # --- Train Discriminator ---
            discriminator_model.zero_grad()

            # 1. Train with real images
            output_real = discriminator_model(x1_real)
            loss_disc_real = criterion_gan(output_real, real_labels)
            loss_disc_real.backward()

            # 2. Train with fake images
            # Sample t for the generator's contribution to discriminator training.
            # Using t from [0.5, 1.0] means xt is more image-like, making D's task harder.
            t_gen_for_disc = torch.rand(batch_size, device=device) * 0.5 + 0.5 # Sample t from [0.5, 1.0]
            
            # xt_fake_for_disc is the input to the generator
            xt_fake_for_disc = (1 - t_gen_for_disc.view(-1, 1, 1, 1)) * z0 + t_gen_for_disc.view(-1, 1, 1, 1) * x1_real
            
            # Get velocity prediction from the generator.
            # We explicitly detach the generator's output from the discriminator's graph.
            with torch.no_grad():
                v_pred_for_disc = generator_model(xt_fake_for_disc, t_gen_for_disc)
                
                # Option B: Implicitly predict the target image (x1_real)
                # This assumes the velocity is approximately constant for the remaining time (1-t)
                generated_for_disc = xt_fake_for_disc + v_pred_for_disc * (1 - t_gen_for_disc.view(-1, 1, 1, 1))
                
                # Clip values to ensure they are within the expected range [-1, 1] for discriminator
                # This prevents extremely large or small values from destabilizing D.
                generated_for_disc = torch.clamp(generated_for_disc, -1, 1)


            output_fake = discriminator_model(generated_for_disc.detach())
            loss_disc_fake = criterion_gan(output_fake, fake_labels)
            loss_disc_fake.backward()

            loss_disc_total = loss_disc_real + loss_disc_fake
            optimizer_disc.step()

            # --- Train Generator (CNF_UNet) ---
            generator_model.zero_grad()

            # CNF Flow Matching Loss (main objective)
            t = torch.rand(batch_size, device=device) # Sample t from [0, 1] for flow matching
            xt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1_real
            v_target = (x1_real - z0)
            v_pred = generator_model(xt, t)
            loss_flow_matching = F.mse_loss(v_pred, v_target)

            # GAN Loss for Generator
            # The generator aims to make its 'predicted final image' look real.
            # We predict the final image based on the current xt and the predicted velocity.
            # Same logic as for the discriminator's fake images, but now it's not detached.
            predicted_final_image_for_gen_gan = xt + v_pred * (1 - t.view(-1, 1, 1, 1))
            
            # Clip values before feeding to discriminator
            predicted_final_image_for_gen_gan = torch.clamp(predicted_final_image_for_gen_gan, -1, 1)

            output_gen_fake = discriminator_model(predicted_final_image_for_gen_gan)
            loss_gen_gan = criterion_gan(output_gen_fake, real_labels) # Generator wants fake to be classified as real

            # Combine losses
            total_gen_loss = loss_flow_matching + lambda_gan * loss_gen_gan
            total_gen_loss.backward()
            optimizer_gen.step()

            # Store losses
            epoch_gen_flow_loss += loss_flow_matching.item()
            epoch_gen_gan_loss += loss_gen_gan.item()
            epoch_disc_real_loss += loss_disc_real.item()
            epoch_disc_fake_loss += loss_disc_fake.item()
            epoch_disc_total_loss += loss_disc_total.item()

        avg_gen_flow_loss = epoch_gen_flow_loss / len(dataloader)
        avg_gen_gan_loss = epoch_gen_gan_loss / len(dataloader)
        avg_disc_real_loss = epoch_disc_real_loss / len(dataloader)
        avg_disc_fake_loss = epoch_disc_fake_loss / len(dataloader)
        avg_disc_total_loss = epoch_disc_total_loss / len(dataloader)

        gen_flow_losses.append(avg_gen_flow_loss)
        gen_gan_losses.append(avg_gen_gan_loss)
        disc_real_losses.append(avg_disc_real_loss)
        disc_fake_losses.append(avg_disc_fake_loss)
        disc_total_losses.append(avg_disc_total_loss)

        print(f"Epoch {epoch+1}: "
              f"G_Flow_Loss={avg_gen_flow_loss:.6f}, G_GAN_Loss={avg_gen_gan_loss:.6f}, "
              f"D_Real_Loss={avg_disc_real_loss:.6f}, D_Fake_Loss={avg_disc_fake_loss:.6f}, "
              f"D_Total_Loss={avg_disc_total_loss:.6f}")

    return {
        'gen_flow_losses': gen_flow_losses,
        'gen_gan_losses': gen_gan_losses,
        'disc_real_losses': disc_real_losses,
        'disc_fake_losses': disc_fake_losses,
        'disc_total_losses': disc_total_losses
    }