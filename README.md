# Lung CT Image Generation using Conditional Normalizing Flows (CNF-UNet)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Achintha9533/Pattern-Recognition/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**Lung CT Image Generation using CNF-UNet** is a deep learning-based tool designed to synthesize realistic Lung CT images from Gaussian noise. It leverages a Conditional Normalizing Flow (CNF) architecture with a U-Net backbone, focusing on learning a continuous transformation from a simple prior distribution to complex medical image data. This repository was created for the final examination of the Software and Computing course in the Applied Physics curriculum at the University of Bologna, emphasizing reproducible and modular deep learning workflows.

The tool offers:
* **Reproducible Results:** Designed for consistent output generation given the same inputs and environment.
* **Modular Design:** Codebase is structured into distinct, testable modules for clarity and maintainability.
* **Comprehensive Evaluation:** Includes standard and perceptual metrics for assessing generated image quality.

It includes:
* **Comprehensive API Documentation:** Automated generation of detailed API documentation using Sphinx from reStructuredText-formatted docstrings.
* Fully documented source code (via docstrings and comments).
* Image preprocessing, model training, inference, and visualization steps.
* Automated testing with high coverage using `pytest`.
* Centralized configuration for easy modification of parameters.

---

## Table of Contents

* [Features](#features)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Example Output](#example-output)
* [Model Architecture and Training](#model-architecture-and-training)
* [Evaluation Metrics](#evaluation-metrics)
* [Testing](#testing)
* [Documentation](#documentation)
* [Limitations and Notes](#limitations-and-notes)
* [License](#license)
* [Contact](#contact)

---

## Features

* **Custom DICOM Dataset Loader:** Efficiently loads and preprocesses Lung CT DICOM images, including intelligent selection of central slices from patient series.
* **Flexible Image Preprocessing:** Configurable transformations for resizing, tensor conversion, and normalization to `[-1, 1]` pixel values.
* **CNF-UNet Generative Architecture:** A robust U-Net backbone integrated with time-dependent conditioning for continuous flow learning.
* **Sinusoidal Positional Embedding:** Incorporates continuous time information into the network using sinusoidal embeddings processed by a dedicated MLP.
* **Residual Blocks & Self-Attention:** Employs modern deep learning components for stable training and enhanced feature learning, including a self-attention mechanism in the bottleneck for capturing global image dependencies.
* **Flow Matching Objective:** Trains the generative model using the stable and effective Flow Matching loss.
* **Comprehensive Evaluation:** Assesses generated image quality using standard metrics: Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Fréchet Inception Distance (FID).
* **Visualization Tools:** Provides integrated plotting functionalities for pixel distributions, training losses, and visual comparison of real vs. generated samples.
* **Modular Codebase:** Organized into distinct Python modules (`config`, `dataset`, `model`, `train`, `generate`, `evaluate`, `visualize`) for clarity and maintainability.
* **Model Checkpointing:** Automatically saves trained generator weights, allowing for model persistence and reusability.
* **Pre-trained Model Loading:** Includes utility to automatically download and load pre-trained model weights from Google Drive for quick setup and inference (`load_model.py`). The first run will automatically download the weights.
* **Comprehensive API Documentation:** Automated generation of detailed API documentation using Sphinx from reStructuredText-formatted docstrings, providing insights into every module, class, and function.


---

## Installation & Setup

This project has been tested with Python 3.9+ on macOS and Windows, and should work with compatible PyTorch versions.

See `requirements.txt` for the full list of dependencies.

To install the application, you can just clone this repository and use pip.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Achintha9533/Pattern-Recognition.git](https://github.com/Achintha9533/Pattern-Recognition.git)
    cd Pattern-Recognition
    ```

2.  **Create and activate a virtual environment (recommended):**
    On macOS/Linux:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    On Windows:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    All required Python packages are listed in `requirements.txt`. With your virtual environment activated, install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note on PyTorch:** The `requirements.txt` will install the CPU version of PyTorch by default (or the specific version frozen). If you have a CUDA-enabled GPU and wish to utilize it, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the exact command that matches your CUDA version.

4.  **Data Setup:**
    This project requires a Lung CT DICOM dataset. The code is configured to expect patient subfolders named `QIN LUNG CT 1` through `QIN LUNG CT 47`.
    * Download your dataset (e.g., the QIN LUNG CT dataset if you have access to it).
    * Place the root directory of your dataset (e.g., the folder containing `QIN LUNG CT 1`, etc.) in a convenient location on your system.
    * **IMPORTANT:** Update the `BASE_DIR` variable in `Synthetic Image Generator/config.py` to the **absolute path** of your dataset's root directory.
        ```python
        # Synthetic Image Generator/config.py
        from pathlib import Path
        BASE_DIR = Path("/Users/kasunachinthaperera/Documents/VS Code/Pattern Recognition/Data/QIN LUNG CT") # <-- Location
        ```

---

## Usage

The primary entry point for the application is `main.py` within the `Synthetic Image Generator` package.

To run the full workflow (data loading, model setup, training, generation, evaluation, and visualization), execute the following command from the root directory of your project:

On macOS/Linux:
```bash
python Synthetic\ Image\ Generator/main.py