Lung CT Image Generation using Conditional Normalizing Flows (CNF-UNet)
This project implements a Conditional Normalizing Flow (CNF) with a U-Net architecture to generate synthetic Lung CT images from Gaussian noise. It serves as a practical exploration of flow-based generative models in the domain of medical imaging, aiming to learn a continuous transformation from a simple prior distribution to complex real-world data. The project emphasizes clean code, comprehensive testing, and clear documentation, adhering to best practices for reproducible scientific software.

Table of Contents
Features

Installation

Usage

Project Structure

Model Architecture and Training

Evaluation Metrics

Examples

Testing

Contributing

License

Contact

Features
Custom DICOM Dataset Loader: Efficiently loads and preprocesses Lung CT DICOM images, including intelligent selection of central slices from patient series.

Flexible Image Preprocessing: Configurable transformations for resizing, tensor conversion, and normalization to [-1, 1] pixel values.

CNF-UNet Generative Architecture: A robust U-Net backbone integrated with time-dependent conditioning for continuous flow learning.

Sinusoidal Positional Embedding: Incorporates continuous time information into the network using sinusoidal embeddings processed by a dedicated MLP.

Residual Blocks & Self-Attention: Employs modern deep learning components for stable training and enhanced feature learning, including a self-attention mechanism in the bottleneck for capturing global image dependencies.

Flow Matching Objective: Trains the generative model using the stable and effective Flow Matching loss.

Comprehensive Evaluation: Assesses generated image quality using standard metrics: Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Fréchet Inception Distance (FID).

Visualization Tools: Provides integrated plotting functionalities for pixel distributions, training losses, and visual comparison of real vs. generated samples.

Modular Codebase: Organized into distinct Python modules (dataset, model, train, generate, evaluate, visualize, config) for clarity and maintainability.

Model Checkpointing: Automatically saves trained generator weights, allowing for model persistence and reusability.

Installation
To set up and run this project, follow these steps:

Clone the repository:

git clone https://github.com/Achintha9533/Pattern-Recognition.git
cd Pattern-Recognition

Create and activate a virtual environment (recommended):

python -m venv venv
# On Windows:
# .\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install dependencies:
All required Python packages are listed in requirements.txt. With your virtual environment activated, install them using pip:

pip install -r requirements.txt

(Note: The torch package installation might be specific to your CUDA version if you have a GPU. The requirements.txt will typically provide a CPU-only version by default or a specific CUDA version if frozen from a CUDA environment. Refer to PyTorch's official installation guide for GPU-specific instructions if needed.)

Data Setup:
This project requires a Lung CT DICOM dataset. The code is configured to expect patient subfolders named QIN LUNG CT 1 through QIN LUNG CT 47.

Download your dataset (e.g., the QIN LUNG CT dataset if you have access to it).

Place the root directory of your dataset (e.g., the folder containing QIN LUNG CT 1, etc.) in a convenient location on your system.

IMPORTANT: Update the BASE_DIR variable in Synthetic Image Generator/config.py to the absolute path of your dataset's root directory.

# Synthetic Image Generator/config.py
BASE_DIR = Path("/path/to/your/QIN LUNG CT") # <-- Update this line

Usage
The primary entry point for the application is main.py within the Synthetic Image Generator package.

To run the full workflow (data loading, model setup, training, generation, evaluation, and visualization), execute the following command from the root of your project:

# On Windows:
python "Synthetic Image Generator"/main.py

# On macOS/Linux:
python Synthetic\ Image\ Generator/main.py

The script will:

Load and preprocess the dataset.

Initialize the CNF-UNet model and optimizer.

Display initial pixel distributions and sample images/noise.

Train the generator model for the configured number of epochs (config.EPOCHS).

Save the trained model weights to ./checkpoints/generator_final.pth.

Plot the training loss curve.

Generate a batch of synthetic images.

Calculate and print quantitative evaluation metrics (MSE, PSNR, SSIM, FID).

Display comparative pixel distribution plots and visual examples of generated images.

Project Structure
The project is organized into a modular structure for clarity and maintainability:

.
├── checkpoints/                   # Directory for saving trained model weights
│   └── generator_final.pth
├── Data/                          # Placeholder for your DICOM dataset (not committed)
│   └── QIN LUNG CT/
│       ├── QIN LUNG CT 1/
│       └── ...
├── Synthetic Image Generator/     # All core Python source code
│   ├── __init__.py                # Makes this directory a Python package
│   ├── config.py                  # Global configuration parameters
│   ├── dataset.py                 # DICOM loading and custom PyTorch Dataset
│   ├── evaluate.py                # Functions for calculating evaluation metrics (MSE, PSNR, SSIM, FID)
│   ├── generate.py                # Function for generating images from noise
│   ├── main.py                    # Main application entry point (orchestrates workflow)
│   ├── model.py                   # Neural network architecture (CNF_UNet, ResidualBlock, Attention)
│   ├── train.py                   # Training loop logic
│   └── transforms.py              # Image preprocessing and FID-specific transformations
├── tests/                         # Unit and integration tests for modules
│   ├── test_config.py
│   ├── test_dataset.py
│   ├── test_evaluate.py
│   ├── test_generate.py
│   ├── test_main.py
│   ├── test_model.py
│   ├── test_train.py
│   └── test_transforms.py
├── images/                        # (Optional) Directory for README visuals (e.g., plots, sample images)
│   └── pixel_distribution_comparison.png
│   └── training_loss.png
│   └── generated_samples.png
│   └── real_vs_generated.png
├── .gitignore                     # Specifies files and directories to be ignored by Git
├── LICENSE                        # Project licensing information (MIT License)
├── README.md                      # This documentation file
└── requirements.txt               # List of Python dependencies with exact versions

Model Architecture and Training
This project employs a Conditional Normalizing Flow (CNF), implemented with a U-Net backbone, to learn the complex distribution of Lung CT images.

The U-Net architecture is well-suited for image-to-image translation tasks due to its encoder-decoder structure with skip connections, which allow the network to capture both high-level semantic information and fine-grained spatial details. Key components of our U-Net include:

Time Embeddings: A crucial aspect of CNFs, continuous time t (sampled from [0, 1]) is encoded using sinusoidal positional embeddings and then processed by a small Multi-Layer Perceptron (MLP). This time embedding is additively conditioned into the feature maps within each ResidualBlock, enabling the network to learn time-dependent transformations.

Residual Blocks: Standard convolutional blocks with identity or 1x1 convolutional shortcut connections ensure stable gradient flow and facilitate the training of deeper networks.

Self-Attention: A SelfAttention2d layer is strategically placed within the U-Net's bottleneck. This mechanism allows the model to capture long-range dependencies across different spatial locations in the image, enhancing its ability to generate globally coherent structures and textures, which is particularly important for medical images.

The model is trained using the Flow Matching objective. Unlike traditional generative adversarial networks (GANs) that rely on adversarial training, Flow Matching directly trains the U-Net to predict the velocity field that transforms a simple prior distribution (Gaussian noise) into the complex data distribution (real Lung CT images) over a continuous time interval. The loss function is the Mean Squared Error (MSE) between the predicted velocity field and the target velocity field at various intermediate time steps. This approach offers stable training and a clear objective for learning the continuous flow.

Evaluation Metrics
The model's generative performance is quantitatively assessed using a suite of widely accepted image quality metrics:

Mean Squared Error (MSE): Measures the average squared difference between generated and real images. Lower values indicate higher pixel-level similarity.

Result (Example): 0.031836

Peak Signal-to-Noise Ratio (PSNR): A logarithmic measure reflecting image quality, often used in image reconstruction. Higher values (in dB) indicate better quality.

Result (Example): 15.159373 dB

Structural Similarity Index Measure (SSIM): Measures perceived structural similarity, considering luminance, contrast, and structure. Values closer to 1 indicate higher perceptual similarity.

Result (Example): 0.228885

Fréchet Inception Distance (FID): A robust perceptual metric that measures the "distance" between the feature distributions of real and generated images using an Inception network. Lower values indicate higher quality and diversity of generated samples.

Result (Example): 262.8561789017828

Interpretation of Current Results:
The provided example metrics (which you should replace with your actual results) suggest that while the model is learning some underlying patterns, there is significant room for improvement in generating high-fidelity and perceptually realistic Lung CT images. The relatively low PSNR and SSIM, coupled with a very high FID score, indicate a considerable discrepancy between the distributions of real and generated images. This implies that the generated samples may currently lack fine details, realistic textures, or the overall statistical properties necessary for high-quality medical image synthesis. Further training, hyperparameter tuning, architectural refinements, or an expanded dataset would be crucial next steps to enhance generative performance.

Examples
Below are visual examples demonstrating the model's capabilities and current performance.

Initial Data and Noise Pixel Distributions

(Replace with your actual plot file path)

Training Loss Curve

(Replace with your actual plot file path)

Sample Generated Images

(Replace with your actual plot file path)

Real vs. Generated Images (Side-by-Side Comparison)

(Replace with your actual plot file path)

Testing
The project includes a comprehensive suite of unit and high-level integration tests to ensure the correctness and reliability of individual modules and the overall application flow. Tests cover data loading, transformations, model components, training, generation, and evaluation logic.

To run all tests, navigate to the root directory of the repository in your terminal and execute:

pytest tests/

This command will discover and execute all test files within the tests/ directory, providing a summary of passing and failing tests.

Contributing
Contributions are welcome! If you find a bug, have a feature request, or wish to contribute code, please feel free to:

Open an issue to discuss the proposed changes.

Fork the repository and create a new branch for your contributions.

Ensure your code adheres to the project's style and includes relevant tests.

Submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any questions, inquiries, or collaborations, please feel free to reach out:

Kasun Achintha Perera
Email: pereraachintha84@gmail.com/kasunachintha.perera@studio.unibo.it
GitHub Profile: https://github.com/Pattern-Recognition
Project Link: Achintha9533/Pattern-Recognition