# evaluate.py
import torch
import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from torch_fidelity import calculate_metrics
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import device, fid_transform, image_size

def evaluate_metrics(generator, dataloader, num_generated_samples=16, steps=200):
    print("\n--- Evaluating Generated Image Fidelity ---")

    real_images_for_eval = []
    
    # Collect real images
    collected_count = 0
    for _, real_img_batch_val in dataloader:
        real_images_for_eval.append(real_img_batch_val)
        collected_count += real_img_batch_val.shape[0]
        if collected_count >= num_generated_samples:
            break
    real_images_batch_tensor = torch.cat(real_images_for_eval, dim=0)[:num_generated_samples]

    if real_images_batch_tensor.numel() == 0:
        print("Could not retrieve enough real images for evaluation. Dataloader might be empty or exhausted.")
        return None, None, None, None

    # Generate images
    initial_noise_for_generation = torch.randn(num_generated_samples, 1, *image_size).to(device)
    with torch.no_grad():
        generator.eval()
        generated_images = []
        # Generate in smaller batches if num_generated_samples is large to manage memory
        batch_size_gen = 4 # Adjust as needed
        for i in tqdm(range(0, num_generated_samples, batch_size_gen), desc="Generating images for evaluation"):
            noise_batch = initial_noise_for_generation[i : i + batch_size_gen]
            gen_batch = []
            current_z = noise_batch.clone()
            for step in range(steps):
                t_val = step / (steps - 1)
                t = torch.tensor(t_val, device=device).repeat(current_z.shape[0])
                v = generator(current_z, t)
                current_z = current_z + v / steps
            generated_images.append(current_z.cpu())
        generated_images = torch.cat(generated_images, dim=0)


    num_compare = min(num_generated_samples, real_images_batch_tensor.shape[0])
    
    real_images_np_01 = ((real_images_batch_tensor[:num_compare].squeeze().numpy() + 1) / 2)
    generated_images_np_01 = ((generated_images[:num_compare].squeeze().numpy() + 1) / 2)

    mses, psnrs, ssims = [], [], []

    for i in range(num_compare):
        real_img = real_images_np_01[i]
        gen_img = generated_images_np_01[i]

        mse = mean_squared_error(real_img, gen_img)
        mses.append(mse)

        psnr = peak_signal_noise_ratio(real_img, gen_img, data_range=1)
        psnrs.append(psnr)

        ssim = structural_similarity(real_img, gen_img, data_range=1)
        ssims.append(ssim)

    print(f"Average MSE: {np.mean(mses):.6f}")
    print(f"Average PSNR: {np.mean(psnrs):.6f} dB")
    print(f"Average SSIM: {np.mean(ssims):.6f}")

    # === FID Calculation ===
    print("\n--- Calculating Fréchet Inception Distance (FID) ---")
    
    real_images_dir = Path("./fid_real_images")
    generated_images_dir = Path("./fid_generated_images")
    
    if real_images_dir.exists():
        shutil.rmtree(real_images_dir)
    if generated_images_dir.exists():
        shutil.rmtree(generated_images_dir)

    real_images_dir.mkdir(exist_ok=True)
    generated_images_dir.mkdir(exist_ok=True)

    fid_value = None
    try:
        print(f"Saving {num_compare} real images to {real_images_dir} for FID calculation...")
        for i in tqdm(range(num_compare)):
            img_to_save = fid_transform(real_images_batch_tensor[i])
            if img_to_save.mode == 'L':
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(real_images_dir / f"real_{i:04d}.png")

        print(f"Saving {num_compare} generated images to {generated_images_dir} for FID calculation...")
        for i in tqdm(range(num_compare)):
            img_to_save = fid_transform(generated_images[i])
            if img_to_save.mode == 'L':
                img_to_save = img_to_save.convert('RGB')
            img_to_save.save(generated_images_dir / f"gen_{i:04d}.png")
        
        print("Initiating FID calculation (this may download InceptionV3 weights if not cached)...")
        metrics = calculate_metrics(
            input1=str(real_images_dir),
            input2=str(generated_images_dir),
            cuda=torch.cuda.is_available(),
            fid=True,
            verbose=True,
        )

        if 'fid' in metrics:
            fid_value = metrics['fid']
            print(f"FID: {fid_value:.4f}")
        else:
            print("FID value not found in metrics. This might indicate an internal error in torch_fidelity.")
            print(f"Full metrics dictionary: {metrics}")

    except Exception as e:
        print(f"An error occurred during FID calculation: {e}")
        print("Please ensure 'torch_fidelity' is installed (`pip install torch_fidelity`) "
              "and that you have the necessary torchvision/Pillow dependencies for image saving."
              "Also, check if Inception V3 model weights are downloaded (torch_fidelity handles this usually).")
    finally:
        if real_images_dir.exists():
            shutil.rmtree(real_images_dir)
        if generated_images_dir.exists():
            shutil.rmtree(generated_images_dir)
        print("Temporary FID image directories cleaned up.")

    return np.mean(mses), np.mean(psnrs), np.mean(ssims), fid_value