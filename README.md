# Lung CT Image Generation using Conditional Normalizing Flows (CNF-UNet)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://achintha9533.github.io/Pattern-Recognition/license.html)
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

-----

## Project Overview

This section provides an introduction to the Synthetic Image Generator, its purpose, and core components.

### Project Details

* **Installation & Setup**: This section details the steps required to install and configure the Synthetic Image Generator project. It covers prerequisites and environmental setup to get the software running.
* **Usage**: This section explains how to use the Synthetic Image Generator. It provides instructions on running the tool, understanding its commands, and applying it to generate images.
* **Model Architecture**: This section delves into the technical design of the model used in the Synthetic Image Generator.
    * **Overview of CNF-UNet**: Provides a high-level summary of the Conditional Normalizing Flow (CNF) with a U-Net backbone.
    * **Normalizing Flows (NF)**: Explains the concept and role of Normalizing Flows within the architecture.
    * **UNet Integration**: Describes how the U-Net structure is integrated into the model.
    * **Conditional Aspect**: Discusses the conditional elements of the model that influence image generation.
    * **Component Details**: Provides a more granular look at the individual components of the model.
    * **Model Architecture Diagram**: Likely includes a visual representation to aid understanding of the model's structure.


  * # Installation & Setup

This project has been tested with Python 3.9+ on macOS and Windows, and should work with compatible PyTorch versions.

See `requirements.txt` for the full list of dependencies.

To install the application, you can just clone this repository and use pip.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Achintha9533/Pattern-Recognition.git
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

    **Note on PyTorch:** The `requirements.txt` will install the CPU version of PyTorch by default (or the specific version frozen). If you have a CUDA-enabled GPU and wish to utilize it, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the exact command that matches your CUDA version.

4.  **Data Setup:**

    This project requires a Lung CT DICOM dataset. The code is configured to expect patient subfolders named `QIN LUNG CT 1` through `QIN LUNG CT 47`.

      * Download your dataset (e.g., the QIN LUNG CT dataset if you have access to it).
      * Place the root directory of your dataset (e.g., the folder containing `QIN LUNG CT 1`, etc.) in a convenient location on your system.
      * **IMPORTANT:** Update the `BASE_DIR` variable in `synthetic_image_generator/config.py` to the **absolute path** of your dataset’s root directory.

    <!-- end list -->

    ## 🔧 Pretrained Weights Setup

To use the pretrained model weights:

1. **Download the weights** from Google Drive:
   - [Click here to download](https://drive.google.com/uc?export=download&id=1BWrRqSEY2KSE-u3TI8c2JYucm69g6oo7)
   - Or run in terminal:
     ```bash
     wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1BWrRqSEY2KSE-u3TI8c2JYucm69g6oo7" -O pretrained_weights.pth
     ```


    
    
## Usage

The primary way to use the application is through `main.py`.

To execute the complete workflow, including data loading, model setup, training, image generation, evaluation, and visualization, run the following command from the project's root directory:

On macOS/Linux:

```bash
python synthetic_image_generator/main.py
```


# Example Outputs

This section provides visual examples generated by the Conditional Normalizing Flows (CNF-UNet) model, along with relevant analysis.

---

**Example of a Synthetically Generated Lung CT Image**

This figure displays a representative synthetically generated Lung CT image. This demonstrates the direct output quality of the model.

---

**Comparison of Generated vs. Real Images**

This figure provides a visual comparison between an image generated by the model and a corresponding real Lung CT image from the dataset. It allows for a direct qualitative assessment of the synthetic data’s realism.

![Comparison of generated and real Lung CT images.](_images/campare.png)
*Visual comparison showing a generated image next to a real image.*

---

**Distribution of Image Characteristics**

This figure illustrates the distribution of certain image characteristics or metrics (e.g., pixel intensities, texture features) across both generated and real datasets. This helps in quantitatively assessing the similarity of the distributions.

![Distribution comparison of image characteristics between real and generated datasets.](_images/distribution.png)
*Comparison of feature distributions for real vs. generated images.*

---

*(You can add more text explaining each figure’s significance and what a reader should observe.)*


# Model Architecture

This section provides a detailed overview of the Conditional Normalizing Flow with UNet (CNF-UNet) architecture employed in this project for Lung CT image generation. Understanding this architecture is key to grasping how the model learns to generate high-quality synthetic images.

## Overview of CNF-UNet

The core of this project’s image generation capability lies in the integration of Conditional Normalizing Flows (CNF) with a UNet-like structure. This combination allows for a powerful generative model that can learn complex data distributions while leveraging the hierarchical feature extraction benefits of the UNet.

[Elaborate here on the high-level concept: How CNF works (invertible transformations, exact likelihood), how UNet contributes (multi-scale feature extraction, skip connections), and how conditioning is applied (e.g., through UNet features).]

## Normalizing Flows (NF)

Normalizing Flows are a class of generative models that transform a simple base distribution (e.g., a Gaussian) into a complex data distribution through a sequence of invertible and differentiable transformations. This allows for exact likelihood computation and efficient sampling.

[Detail the specific types of flow layers used (e.g., coupling layers, permutations, non-linearities) and why they were chosen.]

## UNet Integration

The UNet architecture, originally developed for biomedical image segmentation, is well-suited for processing images due to its encoder-decoder structure with skip connections. In our CNF-UNet, the UNet part typically acts as a powerful feature extractor that provides rich, multi-scale conditional information to the normalizing flow.

[Explain how the UNet is used: Does it provide features at different scales to different flow blocks? Is it an encoder for the conditioning? How are skip connections utilized in this context?]

## Conditional Aspect

The “Conditional” aspect of CNF-UNet means the model’s generation process is guided by certain input conditions. This allows for controlled image synthesis.

[Describe what the conditioning variables are (e.g., clinical parameters, partial images, noise levels) and how they are incorporated into the CNF and/or UNet parts of the architecture. How does this conditioning influence the generated output?]

## Component Details

* **UNetBlock:** [Describe the structure of your basic UNet building block (e.g., convolutional layers, activation functions, batch normalization). You can link to its API documentation here.]
* **Flow Layers:** [Detail the specific flow layers like Affine Coupling Layers, Invertible 1x1 Convolutions, etc., if applicable.]
* **Loss Function:** [Briefly mention the loss function, typically negative log-likelihood for NFs.]

## Model Architecture Diagram

A visual representation of the CNF-UNet architecture helps in understanding the data flow and the interaction between the Normalizing Flow and UNet components.

![Diagram of the CNF-UNet model architecture.](_images/model_architecture_diagram.png)
*Conceptual diagram illustrating the integrated CNF-UNet architecture.*

[Remember to replace `model_architecture_diagram.png` with the actual path to your diagram image. You’ll need to create this diagram (e.g., using draw.io, Excalidraw, PowerPoint, or a scientific plotting library) and save it in `docs/source/_static/`.]


# Evaluation Metrics

Evaluating the performance of generative models, especially for complex data like medical images, requires robust quantitative metrics in addition to qualitative visual inspection. This section outlines the key metrics employed to assess the quality, realism, and diversity of the synthetically generated Lung CT images from the CNF-UNet model, based on the `synthetic_image_generator.evaluate` module.

## Quantitative Metrics

We utilized a combination of established metrics to provide a comprehensive evaluation:

1.  **Mean Squared Error (MSE), Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index (SSIM)**
    * **Purpose:** These metrics quantify the per-pixel and structural similarity between images.
        * **MSE** measures the average squared difference between pixels. A lower MSE indicates better pixel-wise similarity.
        * **PSNR** measures the ratio between the maximum possible pixel value and the power of distorting noise. A higher PSNR indicates better quality.
        * **SSIM** is a perceptual metric that quantifies image quality degradation based on structural information, luminance, and contrast changes. Values range from -1 to 1, with 1 being perfect similarity.

    * **Implementation in Project:** These metrics are calculated per-image by the `calculate_image_metrics` function within the `evaluate` module. They use `skimage.metrics` on NumPy arrays that are assumed to be normalized to the `[0, 1]` range. For `peak_signal_noise_ratio` and `structural_similarity`, a `data_range` of `1.0` is specified, and `channel_axis=None` for single-channel (grayscale) images. In the `evaluate_model` function, these metrics are computed for a `num_compare` subset of real and generated image pairs and then averaged.
    * **Results:** [Present your average MSE, PSNR, and SSIM scores here, along with interpretations. E.g., “Our model achieved an average MSE of X.XXX, PSNR of Y.YY dB, and SSIM of Z.ZZZ, suggesting a good fidelity at the pixel level on the compared samples.”]

2.  **Fréchet Inception Distance (FID)**
    * **Purpose:** FID is a more robust and perceptually relevant metric for assessing the quality and diversity of images generated by generative models. It measures the “distance” between the feature distributions of real and generated images, extracted from a pre-trained Inception-v3 network. A lower FID score indicates that the distribution of generated images is closer to the real image distribution, implying both higher quality and and better diversity.
    * **Implementation in Project:** FID is calculated using the `torch_fidelity` library within the `evaluate_model` function. Before calculation, both real and generated images (which are initially in `[-1, 1]` pixel range) are processed using a `fid_transform` (which typically denormalizes them to `[0, 255]` and converts them to PIL Image format). These preprocessed images are then temporarily saved to disk in designated `fid_real_images` and `fid_generated_images` directories. `torch_fidelity` then uses these directories as input to compute the FID score, leveraging CUDA if available. The temporary directories are cleaned up after calculation.
    * **Results:** [Present your FID scores here. E.g., “The FID score obtained was X.XX. This low score demonstrates that the generated images capture the overall statistical properties and diversity of the real dataset effectively.”]

## Qualitative Evaluation

Beyond quantitative metrics, visual inspection by human observers is crucial for generative models. This involves manually reviewing a diverse set of generated images to assess:

* **Realism:** Do the images look like actual Lung CT scans?
* **Anatomical Correctness:** Are anatomical structures (e.g., lungs, blood vessels) plausible?
* **Diversity:** Does the model generate a wide variety of distinct images, or does it suffer from mode collapse?
* **Artifacts:** Are there any noticeable artifacts, noise, or blurring?
* **Clinical Relevance:** Do the images maintain clinical relevance, such as showing variations in lung conditions?


# Testing Strategy

Robust testing is crucial for ensuring the reliability, correctness, and maintainability of the Synthetic Image Generator project. This section outlines the testing strategy employed and provides an overview of the unit test modules.

## Testing Framework and Approach

The project utilizes `pytest` as its primary testing framework due to its simplicity, extensibility, and rich set of features for writing clear and concise tests. `unittest.mock` is extensively used to isolate the units under test from their external dependencies (e.g., file system, network operations, PyTorch operations, other modules), ensuring that tests are fast, deterministic, and truly focused on the specific logic being verified.

The testing approach primarily focuses on **unit testing**, where individual functions, methods, or classes are tested in isolation. This allows for:

* **Early Bug Detection:** Identifying issues in small, isolated components before they propagate.
* **Code Quality Assurance:** Ensuring each piece of code behaves as expected.
* **Refactoring Confidence:** Providing a safety net when making changes to the codebase.
* **Documentation by Example:** Tests often serve as executable documentation for how code components are intended to be used.

## Test Modules Overview

The project’s test suite is organized into several modules, each responsible for testing a specific part of the application’s functionality:

* **`tests/test_main.py`**
    * **Purpose:** This module provides comprehensive unit testing of the main execution flow (`main.py`). It verifies the correct orchestration and interaction of various components, ensuring that functions are called with expected arguments and in the correct sequence.

* **`tests/test_dataset.py`**
    * **Purpose:** This test suite covers the `dataset` module, including the `load_dicom_image` function and the `LungCTWithGaussianDataset` class. It ensures correct DICOM file loading, image preprocessing, and overall dataset behavior, including error handling for missing data.

* **`tests/test_transforms.py`**
    * **Purpose:** This module contains unit tests for the `transforms` module, specifically `get_transforms` and `get_fid_transforms` functions. It verifies that image preprocessing and post-processing pipelines (e.g., resizing, type conversion, normalization/denormalization) behave as expected.

* **`tests/test_model.py`**
    * **Purpose:** [This file was not provided, but typically would contain tests for the `CNF_UNet` architecture, ensuring forward passes work, and potentially basic shape checks.]

* **`tests/test_load_model.py`**
    * **Purpose:** This module focuses on testing the `load_model_from_drive` function, which handles downloading pre-trained model weights and initializing the `CNF_UNet` model. It covers scenarios such as successful loading, handling CUDA unavailability, file not found errors, and download failures.

* **`tests/test_train.py`**
    * **Purpose:** This test suite verifies the `train_model` function within the `train` module. It ensures that the training loop correctly updates model weights, that the loss generally decreases over epochs, and that no NaN/Inf values are produced during training.

* **`tests/test_generate.py`**
    * **Purpose:** This module contains unit tests for the `generate_images` function in the `generate` module. It ensures that the image generation process produces outputs of the correct shape and type, and verifies that different initial noise inputs lead to distinct generated images.

* **`tests/test_evaluate.py`**
    * **Purpose:** This test suite covers the `evaluate` module, specifically the `evaluate_model` function. It ensures that image quality metrics (MSE, PSNR, SSIM, FID) are calculated correctly and that temporary directories required for FID calculation are properly managed and cleaned up.


# Limitations

While the Conditional Normalizing Flow (CNF) based Synthetic Image Generator demonstrates promising capabilities in producing realistic Lung CT images, it is important to acknowledge certain limitations inherent to the model architecture, the nature of the data, and the scope of this project. Understanding these limitations is crucial for interpreting the model’s performance and guiding future research.

1.  **Computational Intensity and Scalability**
    * **Explanation:** Normalizing Flows, by design, involve computing the determinant of the Jacobian matrix, which can be computationally intensive, especially for high-resolution images or deep flow architectures. Training these models requires significant computational resources (GPUs with large memory) and can be time-consuming. Generating a large number of high-resolution images can also be slow. This limits immediate scalability to extremely high-resolution (e.g., 512x512 or 1024x1024 and beyond) 3D medical volumes without substantial architectural or hardware advancements.

2.  **Fidelity vs. Diversity Trade-off and Subtle Artifacts**
    * **Explanation:** Achieving a perfect balance between generating highly realistic (high fidelity) images and ensuring a wide range of diverse samples (avoiding mode collapse) remains a challenge for all generative models, including CNFs. While quantitative metrics like FID aim to capture both, minor or subtle artifacts, imperceptible to current metrics, might still be present in generated images. These artifacts could be crucial in sensitive applications like medical diagnosis, requiring expert human review. The model might struggle with rare anatomical variations or pathological findings if they are not sufficiently represented in the training data.

3.  **Data Dependency and Generalization**
    * **Explanation:** Like all data-driven deep learning models, the performance and generalizability of the CNF model are highly dependent on the quality, size, and diversity of the training dataset. If the training data is biased, contains artifacts, or lacks representation of certain conditions or patient demographics, the generated images will reflect these limitations. Generating images for novel pathologies or unseen anatomical configurations, which were not present in the training set, remains a significant challenge.

4.  **Lack of Fine-Grained Controllability**
    * **Explanation:** While the conditional aspect of the CNF-UNet allows for generating images based on a latent variable (e.g., different types of noise leading to different images), achieving precise, fine-grained control over specific anatomical features or the explicit introduction/removal of particular pathologies is complex. Current conditioning mechanisms might offer coarse control, but manipulating very specific, localized characteristics (e.g., the exact size and location of a small nodule) is not straightforward.

5.  **Clinical Validation and Interpretability**
    * **Explanation:** The synthetic images produced by this research project are intended for research and development purposes (e.g., data augmentation, privacy-preserving sharing). They have not undergone rigorous clinical validation by medical professionals for diagnostic accuracy or suitability in real-world clinical workflows. Furthermore, the “black-box” nature of deep learning models means that understanding precisely *why* a certain image was generated or *how* a specific input led to a particular output remains challenging, which can be a barrier to trust and adoption in clinical settings.

These limitations highlight areas for future work and underscore that while synthetic data can be incredibly useful, its application, especially in critical domains like healthcare, requires careful consideration and further validation.


# License

This project, Synthetic Image Generator, is made available under the terms of the MIT License. This permissive license allows for wide use, modification, and distribution, while requiring attribution.

* [MIT License](https://achintha9533.github.io/Pattern-Recognition/license.html)


# Contact

For inquiries, feedback, or collaborations related to the Synthetic Image Generator project, please use the contact information provided below.

## Project Author

* **Name:** Kasun Achintha Perera
* **Email:** [pereraachintha84@gmail.com](mailto:pereraachintha84@gmail.com)
* **GitHub:** [https://github.com/Achintha9533](https://github.com/Achintha9533)
* **LinkedIn:** [https://www.linkedin.com/in/kasun-achintha-perera-068a43174/](https://www.linkedin.com/in/kasun-achintha-perera-068a43174/)

## Project Repository

For technical issues, bug reports, feature requests, or contributions, please refer to the project’s GitHub repository:

* **GitHub Repository:** [https://github.com/Achintha9533/Pattern-Recognition](https://github.com/Achintha9533/Pattern-Recognition)

We welcome your feedback and appreciate any contributions to improve this project.



-----

## Documentation

This project includes complete documentation to help users understand, use, and extend the codebase. It covers usage instructions, command-line options, example outputs, testing setup, and detailed API reference generated from inline docstrings. The structure and design choices are also explained for easier onboarding and maintainability.

For a complete guide, please see the [full documentation here](https://achintha9533.github.io/Pattern-Recognition/).

