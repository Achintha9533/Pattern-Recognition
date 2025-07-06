# Synthetic Image Generator/config.py

from pathlib import Path

# === General Configuration ===
# Desired output image size (height, width)
IMAGE_SIZE = (64, 64)

# Optimizer learning rates
G_LR = 1e-4

# === Dataset Configuration ===
# Base directory containing patient subfolders with DICOM files
# IMPORTANT: Adjust this path to your actual data directory
# Example: base_dir = Path("/path/to/your/QIN LUNG CT")
BASE_DIR = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT")

# Number of middle images to select from each patient folder
NUM_IMAGES_PER_FOLDER = 5

# === Checkpoint Settings ===
# Directory to save model weights
CHECKPOINT_DIR = Path("./checkpoints")
# Ensure the directory exists
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
GENERATOR_CHECKPOINT_PATH = CHECKPOINT_DIR / "generator_final.pth"

# === Training Settings ===
EPOCHS = 150
BATCH_SIZE = 64
NUM_WORKERS = 0 # Set to 0 for macOS compatibility or if issues arise

# === Generation Settings ===
GENERATION_STEPS = 200
NUM_GENERATED_SAMPLES = 256

# === Evaluation Settings ===
# Number of batches to sample for real image pixel distribution plot
NUM_BATCHES_FOR_DIST_PLOT = 5