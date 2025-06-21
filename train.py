import torch
import torch.nn as nn
import torch.nn.functional as F
import piq

def train(model, dataloader, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for z0, x1 in dataloader:
            z0, x1 = z0.to(device), x1.to(device)
            t = torch.rand(z0.size(0), device=device)
            xt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1
            v_target = (x1 - z0)
            v_pred = model(xt, t)
            mse_loss = F.mse_loss(v_pred, v_target)
            ssim_loss = 1 - piq.ssim(
                (v_pred + 1) / 2,
                (v_target + 1) / 2,
                data_range=1.0,
                reduction='mean'
            )
            loss = 0.8 * mse_loss + 0.2 * ssim_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(dataloader):.6f}")