Lung CT Image Generation using CNF-UNet
This project implements a Conditional Normalizing Flow (CNF) with a U-Net architecture to generate synthetic Lung CT images from Gaussian noise. It aims to explore the capabilities of flow-based generative models in medical imaging applications, specifically focusing on learning the transformation from a simple noise distribution to complex medical image data.

Features
Custom DICOM Dataset: Efficiently loads and preprocesses Lung CT DICOM images from patient folders.

Flexible Image Preprocessing: Includes resizing, tensor conversion, and normalization to [-1, 1].

CNF-UNet Architecture: A robust U-Net backbone with integrated time-dependent conditioning.

Sinusoidal Positional Embedding: Utilizes sinusoidal embeddings to incorporate time information into the network.

Residual Blocks: Employs residual connections for stable training and improved performance.

Self-Attention Mechanism: Integrates self-attention in the bottleneck to capture global dependencies in images.

Flow Matching Objective: Trains the generative model using the modern Flow Matching loss for continuous-time generative processes.

Comprehensive Evaluation: Assesses generated image quality using MSE, PSNR, SSIM, and Fréchet Inception Distance (FID).

Visualization Tools: Provides plots for pixel distributions, training losses, and sample generated images.

Model Checkpointing: Saves trained generator weights for future use.

Installation
To set up and run this project, follow these steps:

Clone the repository:

git clone https://github.com/your-username/your-project-name.git
cd your-project-name

Create and activate a virtual environment (recommended):

python -m venv venv
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install dependencies:
This project requires PyTorch and several other libraries. You can install them using pip:

pip install -r requirements.txt

(Create a requirements.txt file in your project root with the following content:)

pydicom
numpy
torch
torchvision
tqdm
matplotlib
scikit-image
torch-fidelity

Data Setup:
The project expects Lung CT DICOM images organized in patient subfolders.

Download your DICOM dataset (e.g., QIN LUNG CT if using the same dataset as in the code).

Place the dataset in a convenient location on your system.

Crucially, update the base_dir variable in your main script (e.g., main.py or train_model.py) to point to the absolute path of your dataset's root directory.

Example: base_dir = Path("/path/to/your/QIN LUNG CT")

Usage
Training the Model

To train the CNF-UNet model, execute the main Python script. The training process will iterate for a specified number of epochs (default is 150 in the provided code) and save the generator's final weights.

python your_main_script_name.py # e.g., python train_model.py

During training, progress will be displayed via tqdm, and epoch-wise losses will be printed. Visualizations of initial data distributions and training losses will be shown at the end of training.

Generating Images

After training, the script will automatically generate a batch of images and display them. If you wish to generate images separately or load a specific checkpoint:

# This is a conceptual example. You might create a dedicated script for generation.
import torch
from pathlib import Path
# Assuming your model definition (CNF_UNet) and generate function are in a file like 'model.py'
from model import CNF_UNet, generate 
import torchvision.transforms as T
import matplotlib.pyplot as plt

# Define image size and checkpoint path (must match training settings)
image_size = (64, 64)
checkpoint_dir = Path("./checkpoints")
generator_checkpoint_path = checkpoint_dir / "generator_final.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the trained generator
generator = CNF_UNet(time_embed_dim=256).to(device)
if generator_checkpoint_path.exists():
    generator.load_state_dict(torch.load(generator_checkpoint_path, map_location=device))
    print(f"Loaded generator weights from {generator_checkpoint_path}")
else:
    print(f"Error: Generator checkpoint not found at {generator_checkpoint_path}")
    exit()

generator.eval() # Set model to evaluation mode

num_samples_to_generate = 16
initial_noise_for_generation = torch.randn(num_samples_to_generate, 1, *image_size).to(device)

print(f"Generating {num_samples_to_generate} images...")
with torch.no_grad():
    generated_images = generate(generator, initial_noise_for_generation, steps=200)
generated_images = generated_images.cpu()

# Visualize generated images
plt.figure(figsize=(10, 8))
for i in range(num_samples_to_generate):
    plt.subplot(4, 4, i + 1)
    plt.imshow(generated_images[i, 0], cmap='gray')
    plt.title(f"Generated {i+1}")
    plt.axis('off')
plt.tight_layout()
plt.suptitle("Sample Generated Images", y=1.02)
plt.show()

Project Structure
.
├── checkpoints/           # Directory to save trained model weights
│   └── generator_final.pth
├── Data/                  # Placeholder for your dataset (e.g., QIN LUNG CT)
│   └── QIN LUNG CT/
│       ├── QIN LUNG CT 1/
│       └── ...
├── your_main_script_name.py # Main script containing dataset, model, training, and evaluation logic
├── model.py               # (Optional) Separate file for CNF_UNet, ResidualBlock, etc.
├── dataset.py             # (Optional) Separate file for LungCTWithGaussianDataset
├── requirements.txt       # List of Python dependencies
└── README.md              # This file

(Note: model.py and dataset.py are suggested for better organization if your main script becomes too large.)

Model Architecture and Training
This project utilizes a Conditional Normalizing Flow (CNF) implemented with a U-Net backbone.

The U-Net is a convolutional neural network architecture well-suited for image-to-image translation tasks due to its skip connections, which help preserve fine-grained details during the generation process. Our U-Net incorporates:

Time Embeddings: Sinusoidal positional embeddings are used to encode the continuous time variable t (from 0 to 1), which is then processed by a small MLP and added to the feature maps within each residual block. This allows the network to learn time-dependent transformations, guiding the flow from noise to data.

Residual Blocks: Standard convolutional blocks with identity or 1x1 convolutional shortcut connections ensure stable gradient flow and enable the training of deeper networks.

Self-Attention: A self-attention layer is included in the bottleneck of the U-Net. This mechanism allows the model to capture long-range dependencies across different spatial locations in the image, enhancing its ability to generate globally coherent structures and textures.

The model is trained using the Flow Matching objective. Unlike traditional generative adversarial networks (GANs) that rely on adversarial training, Flow Matching directly trains the U-Net to predict the velocity field that transforms a simple prior distribution (Gaussian noise, z0) into the complex data distribution (real Lung CT images, x1_real) over a continuous time interval. The loss function is the Mean Squared Error (MSE) between the predicted velocity field and the target velocity field at various intermediate time steps. This approach offers stable training and a clear objective for learning the continuous flow.

Evaluation Metrics
The model's performance is evaluated using both pixel-level and perceptual metrics:

Mean Squared Error (MSE): Measures the average squared difference between generated and real images. Lower values indicate higher similarity.

Result: 0.031836

Peak Signal-to-Noise Ratio (PSNR): A logarithmic measure reflecting image quality, often used in image reconstruction. Higher values (in dB) indicate better quality.

Result: 15.159373 dB

Structural Similarity Index Measure (SSIM): Measures perceived structural similarity, considering luminance, contrast, and structure. Values closer to 1 indicate higher similarity.

Result: 0.228885

Fréchet Inception Distance (FID): A widely used perceptual metric that measures the "distance" between the feature distributions of real and generated images using an Inception network. Lower values indicate higher quality and diversity of generated samples.

Result: 262.8561789017828

Interpretation of Results:
The current evaluation metrics indicate that while the model is learning some underlying patterns, there is significant room for improvement in generating high-fidelity and perceptually realistic Lung CT images. The PSNR of ~15 dB and SSIM of ~0.23 are relatively low, suggesting noticeable differences between generated and real images. The very high FID score of ~262.86 further confirms a substantial discrepancy between the distributions of real and generated images, implying that the generated samples may lack fidelity, diversity, or both, compared to the true data distribution. Further training, hyperparameter tuning, or architectural refinements would be necessary to achieve better generative performance.

Examples
Below are conceptual placeholders for visualizing the model's output. You would replace these with actual plots or image files generated by your script.

Pixel Distribution Comparison

(Insert plot showing comparison of pixel distributions between real and generated images)

Sample Real vs. Generated Images

(Insert a grid of real images and their corresponding generated counterparts)

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions or inquiries, please contact:

Kasun Achintha Perera - pereraachintha84@.com/kasunachintha.perera@studio.unibo.it
Project Link: https://github.com/Achintha9533/Pattern-Recognition

