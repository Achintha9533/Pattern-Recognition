# tests/test_evaluate.py

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import os
import shutil
import logging
from typing import Dict, Any, Generator, Tuple

# Import modules from your package
from Synthetic_Image_Generator.evaluate import evaluate_model
from Synthetic_Image_Generator.transforms import get_fid_transforms # Needed for fid_transform fixture
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

"""
Test suite for the evaluation module.

This module contains unit tests for the `evaluate_model` function,
ensuring that image quality metrics (MSE, PSNR, SSIM, FID) are
calculated correctly and that temporary directories are managed properly.
Mocks are used extensively to isolate the `evaluate_model` logic from
actual file system operations and external library calls (e.g., `torch_fidelity`).
"""

# Suppress logging during tests to keep output clean, allowing specific logs via caplog
logging.getLogger().setLevel(logging.CRITICAL)

@pytest.fixture
def dummy_image_tensors() -> Dict[str, torch.Tensor]:
    """
    Fixture for providing dummy real and generated image tensors.

    These tensors are normalized to [-1, 1] and are designed with simple,
    predictable patterns to facilitate accurate metric calculation in tests.
    It provides various scenarios: perfect match, slight difference, and large difference.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing various dummy image tensors:
                                 - "real": A base real image tensor.
                                 - "perfect": A generated image tensor identical to "real".
                                 - "slight_diff": A generated image tensor with a small, uniform difference from "real".
                                 - "large_diff": A generated image tensor with a significant difference from "real".

    Potential Exceptions Raised:
        None.

    Example of Usage:
    ```python
    def test_metrics(dummy_image_tensors):
        real = dummy_image_tensors["real"]
        generated = dummy_image_tensors["perfect"]
        # ... use real and generated for evaluation ...
    ```

    Relationships with Other Functions:
        * Used by `test_evaluate_model_metrics_calculation` and other evaluation tests.

    Explanation of the Theory:
        Fixtures like this ensure consistent test data across multiple tests,
        promoting test maintainability and reducing redundancy. The use of
        simple, predictable data simplifies the verification of metric calculations.
    """
    # Create dummy images for testing, normalized to [-1, 1]
    real_image = torch.full((1, 1, 64, 64), 0.5, dtype=torch.float32) # Uniform image
    perfect_gen_image = torch.full((1, 1, 64, 64), 0.5, dtype=torch.float32) # Identical
    slight_diff_gen_image = torch.full((1, 1, 64, 64), 0.51, dtype=torch.float32) # Slight difference
    large_diff_gen_image = torch.full((1, 1, 64, 64), -0.5, dtype=torch.float32) # Large difference

    return {
        "real": real_image,
        "perfect": perfect_gen_image,
        "slight_diff": slight_diff_gen_image,
        "large_diff": large_diff_gen_image
    }

@pytest.fixture
def fid_transform_fixture() -> T.Compose:
    """
    Fixture for providing the FID-specific image transformations.

    This returns a `torchvision.transforms.Compose` object configured for
    FID calculation, which typically involves denormalization and conversion
    to PIL Image format.

    Returns:
        T.Compose: A composed transformation pipeline suitable for FID calculation.

    Potential Exceptions Raised:
        None.

    Example of Usage:
    ```python
    def test_fid_calculation(fid_transform_fixture):
        # fid_transform_fixture can be passed to evaluate_model
        evaluate_model(..., fid_transform=fid_transform_fixture, ...)
    ```

    Relationships with Other Functions:
        * Depends on `get_fid_transforms` from the `transforms` module.
        * Used by `test_evaluate_model_fid_calculation` and other FID-related tests.

    Explanation of the Theory:
        This fixture ensures that tests for FID calculation use the exact
        transformation pipeline expected by the `torch_fidelity` library,
        leading to accurate and reliable test results.
    """
    return get_fid_transforms()

@pytest.fixture(autouse=True)
def mock_external_dependencies(mocker: Any) -> None:
    """
    Fixture to mock external dependencies like `torch_fidelity.calculate_metrics`
    and `shutil.rmtree` to isolate `evaluate_model` logic during tests.

    This fixture automatically patches these functions, preventing actual
    file system operations or calls to external libraries, which makes tests
    faster and more reliable.

    Args:
        mocker (Any): The `pytest-mock` fixture for patching.

    Returns:
        None.

    Potential Exceptions Raised:
        None directly, but patching can fail if the target is incorrectly specified.

    Example of Usage:
    ```python
    # This fixture is autoused, so it runs automatically for all tests in this module.
    # No explicit call needed in test functions.
    ```

    Relationships with Other Functions:
        * Patches `torch_fidelity.calculate_metrics` and `shutil.rmtree`.
        * Affects all tests in this module by controlling external interactions.

    Explanation of the Theory:
        Mocking is a crucial technique in unit testing to isolate the code
        under test from its dependencies. This ensures that a test fails only
        if the code being tested has a bug, not because of issues in external
        systems or slow I/O operations.
    """
    mocker.patch('Synthetic_Image_Generator.evaluate.calculate_metrics', return_value={'fid': 100.0})
    mocker.patch('shutil.rmtree')

# --- Tests for evaluate_model function ---

def test_evaluate_model_metrics_calculation(
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    caplog: Any
) -> None:
    """
    Test that `evaluate_model` correctly calculates and logs pixel-wise metrics (MSE, PSNR, SSIM)
    for different levels of image similarity.

    Given batches of real and generated images (perfect match, slight difference, large difference),
    When `evaluate_model` is called,
    Then the logged MSE, PSNR, and SSIM values should be as expected for each scenario.
    The FID calculation is mocked, so its value is not directly tested here but ensured to be called.

    Args:
        dummy_image_tensors (Dict[str, torch.Tensor]): Fixture providing various dummy image tensors.
        fid_transform_fixture (T.Compose): Fixture providing the FID-specific transformations.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If the calculated or logged metric values do not match expectations.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_image_tensors` and `fid_transform_fixture` for inputs.
        * Tests the core metric calculation logic within `evaluate_model`.
        * Verifies interactions with `skimage.metrics` functions (though not mocked).

    Explanation of the Theory:
        This test validates the quantitative assessment of generative model performance.
        By providing controlled inputs, it ensures the correctness of the metric
        implementations, which are crucial for tracking model improvement.
    """
    real = dummy_image_tensors["real"]

    # Test perfect match
    generated_perfect = dummy_image_tensors["perfect"]
    with caplog.at_level(logging.INFO):
        evaluate_model(real, generated_perfect, fid_transform_fixture, num_compare=1)
    assert "MSE: 0.0000" in caplog.text
    assert "PSNR: inf" in caplog.text # PSNR is inf for perfect match
    assert "SSIM: 1.0000" in caplog.text
    caplog.clear()

    # Test slight difference
    generated_slight = dummy_image_tensors["slight_diff"]
    with caplog.at_level(logging.INFO):
        evaluate_model(real, generated_slight, fid_transform_fixture, num_compare=1)
    # Expected MSE for (0.5 - 0.51)^2 = (-0.01)^2 = 0.0001
    assert "MSE: 0.0001" in caplog.text or "MSE: 0.000100" in caplog.text
    assert "PSNR: 40.0000" in caplog.text # PSNR = 10 * log10((MAX_I^2) / MSE) where MAX_I = 2 (for [-1,1] range)
                                         # 10 * log10(2^2 / 0.0001) = 10 * log10(4 / 0.0001) = 10 * log10(40000) = 10 * 4.602 = 46.02
                                         # The PSNR calculation in skimage might use a different MAX_I or range.
                                         # Let's verify based on normalized [0,1] for skimage.
                                         # If images are scaled to [0,1] then MSE is 0.0001, MAX_I=1. PSNR = 10 * log10(1/0.0001) = 10 * 4 = 40.
    assert "SSIM: 0.99" in caplog.text # SSIM for slight diff will be very high
    caplog.clear()

    # Test large difference
    generated_large = dummy_image_tensors["large_diff"]
    with caplog.at_level(logging.INFO):
        evaluate_model(real, generated_large, fid_transform_fixture, num_compare=1)
    # Expected MSE for (0.5 - (-0.5))^2 = (1.0)^2 = 1.0
    assert "MSE: 1.0000" in caplog.text
    assert "PSNR: 0.0000" in caplog.text # PSNR for large diff will be very low or 0
    assert "SSIM: 0." in caplog.text # SSIM for large diff will be very low
    caplog.clear()


def test_evaluate_model_fid_calculation(
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    mocker: Any,
    caplog: Any
) -> None:
    """
    Test that `evaluate_model` calls `torch_fidelity.calculate_metrics` with the correct
    arguments and logs the FID score.

    Given batches of real and generated images and a mocked `calculate_metrics` function,
    When `evaluate_model` is called,
    Then `calculate_metrics` should be called exactly once with the expected `input1` and `input2`
    (temporary directories for real and generated images), and the FID value should be logged.

    Args:
        dummy_image_tensors (Dict[str, torch.Tensor]): Fixture providing dummy image tensors.
        fid_transform_fixture (T.Compose): Fixture providing the FID-specific transformations.
        mocker (Any): The `pytest-mock` fixture for patching.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If `calculate_metrics` is not called as expected or if the FID log is missing.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_image_tensors` and `fid_transform_fixture` for inputs.
        * Tests the integration with `torch_fidelity.calculate_metrics`.
        * Relies on the `mock_external_dependencies` fixture.

    Explanation of the Theory:
        This test focuses on the integration point with the external `torch_fidelity` library.
        By mocking the library's actual execution, it verifies that the `evaluate_model`
        function correctly prepares the inputs and invokes the FID calculation,
        without incurring the computational cost of a full FID run.
    """
    real = dummy_image_tensors["real"]
    generated = dummy_image_tensors["perfect"]

    mock_calculate_metrics = mocker.patch(
        'Synthetic_Image_Generator.evaluate.calculate_metrics',
        return_value={'fid': 123.456} # Custom return value for this test
    )

    with caplog.at_level(logging.INFO):
        evaluate_model(real, generated, fid_transform_fixture, num_compare=1)

    mock_calculate_metrics.assert_called_once()
    args, kwargs = mock_calculate_metrics.call_args
    assert 'input1' in kwargs and 'input2' in kwargs, "calculate_metrics should be called with input1 and input2."
    assert "FID: 123.4560" in caplog.text, "Expected FID value not logged."

def test_evaluate_model_temporary_directory_cleanup(
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    mocker: Any
) -> None:
    """
    Test that `evaluate_model` attempts to clean up temporary directories after FID calculation.

    Given batches of real and generated images, and a mocked `shutil.rmtree` function,
    When `evaluate_model` is called,
    Then `shutil.rmtree` should be called twice (once for real, once for generated images)
    to clean up the temporary directories, regardless of whether FID calculation succeeds or fails.

    Args:
        dummy_image_tensors (Dict[str, torch.Tensor]): Fixture providing dummy image tensors.
        fid_transform_fixture (T.Compose): Fixture providing the FID-specific transformations.
        mocker (Any): The `pytest-mock` fixture for patching.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If `shutil.rmtree` is not called the expected number of times.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_image_tensors` and `fid_transform_fixture` for inputs.
        * Tests the resource management aspect of `evaluate_model`.
        * Relies on the `mock_external_dependencies` fixture.

    Explanation of the Theory:
        Proper resource management, especially cleaning up temporary files, is critical
        for preventing disk space issues and ensuring tests leave no side effects.
        This test verifies that the cleanup mechanism is correctly invoked.
    """
    real = dummy_image_tensors["real"]
    generated = dummy_image_tensors["perfect"]

    mock_rmtree = mocker.patch('shutil.rmtree')

    evaluate_model(real, generated, fid_transform_fixture, num_compare=1)

    assert mock_rmtree.call_count == 2, "shutil.rmtree should be called twice for cleanup."

    # Verify that it's called even if FID calculation fails (by raising an error in calculate_metrics)
    mock_calculate_metrics_fail = mocker.patch(
        'Synthetic_Image_Generator.evaluate.calculate_metrics',
        side_effect=Exception("FID calculation failed for test.")
    )
    mock_rmtree.reset_mock() # Reset call count for the next part of the test

    with pytest.raises(Exception): # Expect the exception from mocked calculate_metrics
        evaluate_model(real, generated, fid_transform_fixture, num_compare=1)

    assert mock_rmtree.call_count == 2, "shutil.rmtree should still be called twice even if FID calculation fails."


def test_evaluate_model_empty_real_images(
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    caplog: Any,
    mocker: Any
) -> None:
    """
    Test that `evaluate_model` handles cases where no real images are provided for evaluation.

    Given an empty tensor for real images and a non-empty tensor for generated images,
    When `evaluate_model` is called,
    Then a warning should be logged indicating insufficient real images, and pixel-wise
    metrics (MSE, PSNR, SSIM) should be skipped. FID calculation might still proceed
    if generated images are present and the `torch_fidelity` mock allows it.

    Args:
        dummy_image_tensors (Dict[str, torch.Tensor]): Fixture providing dummy image tensors.
        fid_transform_fixture (T.Compose): Fixture providing the FID-specific transformations.
        caplog (Any): pytest fixture to capture log messages.
        mocker (Any): The `pytest-mock` fixture for patching.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If the expected warning is not logged or if metrics are incorrectly calculated.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Depends on `dummy_image_tensors` and `fid_transform_fixture` for inputs.
        * Tests the robustness and error handling within `evaluate_model`.

    Explanation of the Theory:
        This test ensures the function gracefully handles edge cases, such as missing
        or insufficient input data. Providing clear warnings and selectively
        skipping parts of the evaluation when data is incomplete is important
        for user feedback and preventing unnecessary errors.
    """
    # Create an empty tensor for real images.
    empty_real_tensor: torch.Tensor = torch.empty(0, 1, 64, 64)
    # Use a dummy generated images tensor for the test.
    dummy_gen_tensor: torch.Tensor = dummy_image_tensors["perfect"] # Using a valid generated image

    # Mock `calculate_metrics` to prevent actual FID calculation and simplify assertion.
    mock_calculate_metrics = mocker.patch('Synthetic_Image_Generator.evaluate.calculate_metrics', return_value={'fid': 0.0})

    with caplog.at_level(logging.WARNING): # Capture WARNING level logs
        evaluate_model(empty_real_tensor, dummy_gen_tensor, fid_transform_fixture, num_compare=1)

    assert "Could not retrieve enough real images for evaluation. Skipping pixel-wise metrics." in caplog.text, \
        "Expected warning for empty real images not logged."

    # `evaluate_model` should still attempt FID if generated images are present.
    mock_calculate_metrics.assert_called_once()
    caplog.clear() # Clear logs for next test if any