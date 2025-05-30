# Cell
import os
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np

# Base directory
base_dir = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")

# Loop through QIN LUNG CT 1 to QIN LUNG CT 47
for i in range(1, 48):  # 1 to 47 inclusive
    folder_name = f"QIN LUNG CT {i}"
    folder_path = base_dir / folder_name

    print(f"\n📁 Showing images from: {folder_name}")

    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):  # Optional: sort for consistency
            if file.lower().endswith(".dcm"):
                file_path = os.path.join(root, file)
                try:
                    ds = pydicom.dcmread(file_path)
                    img = ds.pixel_array.astype(np.float32)

                    # Normalize to [-1, +1]
                    if img.max() != img.min():
                        img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
                    else:
                        img = np.zeros_like(img)

                    # Resize to 64x64
                    img_resized = resize(img, (64, 64), anti_aliasing=True)

                    """# Display the image
                    plt.figure(figsize=(4, 4))
                    plt.imshow(img_resized, cmap='gray', vmin=-1, vmax=1)
                    plt.title(f"{folder_name} - {file}")
                    plt.axis('off')
                    plt.show()"""

                except Exception as e:
                    print(f"❌ Could not display {file_path}: {e}")

