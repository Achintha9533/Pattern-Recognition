# generate.py
import torch
from tqdm import tqdm
from config import device, image_size # Assuming image_size is still imported from config

@torch.no_grad()
def generate(model, num_samples, steps=200): # 'num_samples' is an int, e.g., 16
    model.eval()
    
    # This is the CRUCIAL line. We create the initial noise tensor *here*.
    initial_noise = torch.randn(num_samples, 1, *image_size).to(device) 
    
    generated_images = []
    # Generate in smaller batches if num_samples is large to manage memory
    batch_size_gen = 4 # Adjust as needed
    for i in tqdm(range(0, num_samples, batch_size_gen), desc="Generating images"):
        # We take a slice of the *already created* initial_noise tensor
        noise_batch = initial_noise[i : i + batch_size_gen] 
        
        # Now, noise_batch is a PyTorch tensor, so .clone() is valid
        current_z = noise_batch.clone() 
        
        for step in range(steps):
            t_val = step / (steps - 1)
            t = torch.tensor(t_val, device=device).repeat(current_z.shape[0])
            v = model(current_z, t)
            current_z = current_z + v / steps
        generated_images.append(current_z.cpu())
    return torch.cat(generated_images, dim=0)