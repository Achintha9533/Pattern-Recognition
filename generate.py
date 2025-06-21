import torch

@torch.no_grad()
def generate(model, z0, steps=100):
    model.eval()
    z = z0.clone().to(device)
    for i in range(steps):
        t = torch.tensor(i / steps, device=device).repeat(z.shape[0])
        v = model(z, t)
        z = z + v / steps
    return z