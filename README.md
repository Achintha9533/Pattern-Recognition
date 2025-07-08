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

* [Project Overview](docs/build/html/project_overview.html)
* [Installation Guide](docs/build/html/installation.html)
* [Usage Guide](docs/build/html/usage.html)
* [Example Output](docs/build/html/example_outputs.html)
* [Model Architecture and Training](docs/build/html/architecture.html)
* [Evaluation Metrics](docs/build/html/evaluation.html)
* [Testing Strategy](docs/build/html/testing.html)
* [Limitations](docs/build/html/limitation.html)
* [License](docs/build/html/license.html)
* [Contact](docs/build/html/contact.html)
* [API Reference](docs/build/html/modules.html)

---
## Documentation

This project includes complete documentation to help users understand, use, and extend the codebase. It covers usage instructions, command-line options, example outputs, testing setup, and detailed API reference generated from inline docstrings. The structure and design choices are also explained for easier onboarding and maintainability.

For a complete guide, please see the [full documentation here](docs/build/html/index.html).

If you prefer to build the docs locally:

```bash
cd docs
make html