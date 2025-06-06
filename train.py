import torch
import torch.nn.functional as F
from tqdm import tqdm

def train(model, dataloader, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for z0, x1 in tqdm(dataloader):
            z0, x1 = z0.to(device), x1.to(device)
            t = torch.rand(z0.size(0), device=device)
            xt = (1 - t.view(-1, 1, 1, 1)) * z0 + t.view(-1, 1, 1, 1) * x1
            v_target = (x1 - z0)
            v_pred = model(xt, t)
            loss = F.mse_loss(v_pred, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.6f}")
