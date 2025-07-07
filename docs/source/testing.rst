Testing Strategy
================

Robust testing is crucial for ensuring the reliability, correctness, and maintainability of the Synthetic Image Generator project. This section outlines the testing strategy employed and provides an overview of the unit test modules.

Testing Framework and Approach
------------------------------

The project utilizes `pytest` as its primary testing framework due to its simplicity, extensibility, and rich set of features for writing clear and concise tests. `unittest.mock` is extensively used to isolate the units under test from their external dependencies (e.g., file system, network operations, PyTorch operations, other modules), ensuring that tests are fast, deterministic, and truly focused on the specific logic being verified.

The testing approach primarily focuses on **unit testing**, where individual functions, methods, or classes are tested in isolation. This allows for:

* **Early Bug Detection:** Identifying issues in small, isolated components before they propagate.
* **Code Quality Assurance:** Ensuring each piece of code behaves as expected.
* **Refactoring Confidence:** Providing a safety net when making changes to the codebase.
* **Documentation by Example:** Tests often serve as executable documentation for how code components are intended to be used.

Test Modules Overview
---------------------

The project's test suite is organized into several modules, each responsible for testing a specific part of the application's functionality:

* :py:mod:`tests/test_main.py`
    * **Purpose:** This module provides comprehensive unit testing of the main execution flow (`main.py`). It verifies the correct orchestration and interaction of various components, ensuring that functions are called with expected arguments and in the correct sequence.

* :py:mod:`tests/test_dataset.py`
    * **Purpose:** This test suite covers the `dataset` module, including the `load_dicom_image` function and the `LungCTWithGaussianDataset` class. It ensures correct DICOM file loading, image preprocessing, and overall dataset behavior, including error handling for missing data.

* :py:mod:`tests/test_transforms.py`
    * **Purpose:** This module contains unit tests for the `transforms` module, specifically `get_transforms` and `get_fid_transforms` functions. It verifies that image preprocessing and post-processing pipelines (e.g., resizing, type conversion, normalization/denormalization) behave as expected.

* :py:mod:`tests/test_model.py`
    * **Purpose:** [This file was not provided, but typically would contain tests for the `CNF_UNet` architecture, ensuring forward passes work, and potentially basic shape checks.]

* :py:mod:`tests/test_load_model.py`
    * **Purpose:** This module focuses on testing the `load_model_from_drive` function, which handles downloading pre-trained model weights and initializing the `CNF_UNet` model. It covers scenarios such as successful loading, handling CUDA unavailability, file not found errors, and download failures.

* :py:mod:`tests/test_train.py`
    * **Purpose:** This test suite verifies the `train_model` function within the `train` module. It ensures that the training loop correctly updates model weights, that the loss generally decreases over epochs, and that no NaN/Inf values are produced during training.

* :py:mod:`tests/test_generate.py`
    * **Purpose:** This module contains unit tests for the `generate_images` function in the `generate` module. It ensures that the image generation process produces outputs of the correct shape and type, and verifies that different initial noise inputs lead to distinct generated images.

* :py:mod:`tests/test_evaluate.py`
    * **Purpose:** This test suite covers the `evaluate` module, specifically the `evaluate_model` function. It ensures that image quality metrics (MSE, PSNR, SSIM, FID) are calculated correctly and that temporary directories required for FID calculation are properly managed and cleaned up.