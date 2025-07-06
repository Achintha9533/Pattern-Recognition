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

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing various dummy image tensors:
                                 - "real": A base real image tensor.
                                 - "perfect": A generated image tensor identical to "real".
                                 - "noisy": A generated image tensor with some noise.
                                 - "zeros": A generated image tensor filled with -1.0.
    """
    # Create simple patterns for predictable metric calculation
    real_img: torch.Tensor = torch.ones(1, 1, 10, 10) * 0.8 # All pixels 0.8
    gen_img_perfect: torch.Tensor = torch.ones(1, 1, 10, 10) * 0.8 # Perfect match
    gen_img_noisy: torch.Tensor = torch.ones(1, 1, 10, 10) * 0.7 + torch.randn(1, 1, 10, 10) * 0.1 # Some noise
    gen_img_zeros: torch.Tensor = torch.zeros(1, 1, 10, 10) * -1.0 # All pixels -1.0 (after normalization)

    return {
        "real": real_img,
        "perfect": gen_img_perfect,
        "noisy": gen_img_noisy,
        "zeros": gen_img_zeros
    }

@pytest.fixture
def fid_transform_fixture() -> T.Compose:
    """
    Fixture for providing the FID transformation pipeline.

    Returns:
        T.Compose: An instance of `torchvision.transforms.Compose` for FID.
    """
    return get_fid_transforms()

def test_evaluate_model_pixel_metrics_perfect_match(
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    caplog: Any # pytest's caplog fixture for capturing logs
) -> None:
    """
    Test that MSE, PSNR, and SSIM are correctly calculated for a perfect match
    between real and generated images.

    Given real and perfectly matched generated image tensors,
    When `evaluate_model` is called,
    Then the logged MSE should be approximately 0, PSNR should be very high (inf),
    and SSIM should be approximately 1.
    """
    real_img: torch.Tensor = dummy_image_tensors["real"]
    gen_img: torch.Tensor = dummy_image_tensors["perfect"]
    
    # Mock `calculate_metrics` to prevent actual FID computation, as this test focuses on pixel metrics.
    with patch('Synthetic_Image_Generator.evaluate.calculate_metrics') as mock_calculate_metrics:
        mock_calculate_metrics.return_value = {'fid': 0.0} # Provide a dummy FID return value

        # Use caplog to capture log messages for assertion.
        with caplog.at_level(logging.INFO):
            evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

        # Convert to [0,1] range for metric calculation, as `skimage.metrics` expects this.
        real_np: np.ndarray = ((real_img.squeeze().numpy() + 1) / 2)
        gen_np_perfect: np.ndarray = ((gen_img.squeeze().numpy() + 1) / 2)

        expected_mse: float = mean_squared_error(real_np, gen_np_perfect)
        expected_psnr: float = peak_signal_noise_ratio(real_np, gen_np_perfect, data_range=1)
        expected_ssim: float = structural_similarity(real_np, gen_np_perfect, data_range=1)

        # Assert that the expected log messages are present with the correct calculated values.
        # Use f-strings for precise matching of logged output.
        assert f"Average MSE: {expected_mse:.6f}" in caplog.text, \
            f"Expected MSE {expected_mse:.6f} not found in logs."
        # PSNR for perfect match is typically 'inf' or a very large number.
        assert ("Average PSNR: inf dB" in caplog.text or f"Average PSNR: {expected_psnr:.6f} dB" in caplog.text), \
            f"Expected PSNR 'inf' or {expected_psnr:.6f} not found in logs."
        assert f"Average SSIM: {expected_ssim:.6f}" in caplog.text, \
            f"Expected SSIM {expected_ssim:.6f} not found in logs."


def test_evaluate_model_pixel_metrics_zeros_vs_real(
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    caplog: Any
) -> None:
    """
    Test that MSE, PSNR, and SSIM are correctly calculated for a generated image
    of all zeros against a real image, expecting lower quality metrics.

    Given a real image and a generated image tensor filled with zeros (or -1.0 normalized),
    When `evaluate_model` is called,
    Then the logged MSE should be high, PSNR low, and SSIM low, reflecting poor quality.
    """
    real_img: torch.Tensor = dummy_image_tensors["real"] # All 0.8 normalized to 0.9 in [0,1]
    gen_img: torch.Tensor = dummy_image_tensors["zeros"] # All -1.0 normalized to 0.0 in [0,1]

    with patch('Synthetic_Image_Generator.evaluate.calculate_metrics') as mock_calculate_metrics:
        mock_calculate_metrics.return_value = {'fid': 1000.0} # Dummy FID

        with caplog.at_level(logging.INFO):
            evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

        real_np: np.ndarray = ((real_img.squeeze().numpy() + 1) / 2)
        gen_np_zeros: np.ndarray = ((gen_img.squeeze().numpy() + 1) / 2)

        expected_mse: float = mean_squared_error(real_np, gen_np_zeros)
        expected_psnr: float = peak_signal_noise_ratio(real_np, gen_np_zeros, data_range=1)
        expected_ssim: float = structural_similarity(real_np, gen_np_zeros, data_range=1)

        assert f"Average MSE: {expected_mse:.6f}" in caplog.text, \
            f"Expected MSE {expected_mse:.6f} not found in logs."
        assert f"Average PSNR: {expected_psnr:.6f} dB" in caplog.text, \
            f"Expected PSNR {expected_psnr:.6f} dB not found in logs."
        assert f"Average SSIM: {expected_ssim:.6f}" in caplog.text, \
            f"Expected SSIM {expected_ssim:.6f} not found in logs."
        
        # Verify the values are in the expected range for poor quality.
        assert expected_mse > 0.1, f"Expected MSE to be high (>0.1), but got {expected_mse}."
        assert expected_psnr < 20, f"Expected PSNR to be low (<20), but got {expected_psnr}."
        assert expected_ssim < 0.5, f"Expected SSIM to be low (<0.5), but got {expected_ssim}."


def test_evaluate_model_fid_calculation_success(
    tmp_path: Path,
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    mocker: Any,
    caplog: Any
) -> None:
    """
    Test that FID calculation is attempted, returns a valid value, and temporary
    directories are correctly cleaned up.

    Given real and generated image tensors,
    When `evaluate_model` is called with `calculate_metrics` mocked to succeed,
    Then `calculate_metrics` should be called with correct paths, the FID value
    should be logged, and temporary directories should be removed.
    """
    real_img: torch.Tensor = dummy_image_tensors["real"]
    gen_img: torch.Tensor = dummy_image_tensors["perfect"]

    # Mock `calculate_metrics` to return a dummy FID value.
    mock_calculate_metrics: MagicMock = mocker.patch(
        'Synthetic_Image_Generator.evaluate.calculate_metrics',
        return_value={'fid': 123.456}
    )

    # Mock `shutil.rmtree` to prevent actual directory deletion during the test
    # and allow us to assert it was called.
    mock_rmtree: MagicMock = mocker.patch('shutil.rmtree')
    # Mock Path.mkdir to prevent actual directory creation by the evaluate module
    mocker.patch('pathlib.Path.mkdir')

    with caplog.at_level(logging.INFO):
        evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

    # Assert `calculate_metrics` was called once with the correct temporary paths and parameters.
    mock_calculate_metrics.assert_called_once()
    args, kwargs = mock_calculate_metrics.call_args
    assert 'input1' in kwargs and 'input2' in kwargs, "FID inputs 'input1' and 'input2' missing."
    assert 'fid_real_images' in kwargs['input1'], "Real images directory path is incorrect."
    assert 'fid_generated_images' in kwargs['input2'], "Generated images directory path is incorrect."
    assert kwargs['fid'] is True, "FID calculation should be enabled."

    # Assert temporary directories were attempted to be cleaned up in the `finally` block.
    assert mock_rmtree.call_count == 2, "shutil.rmtree should be called twice for cleanup."
    assert any("fid_real_images" in str(call_arg[0]) for call_arg in mock_rmtree.call_args_list), \
        "Cleanup not called for real images directory."
    assert any("fid_generated_images" in str(call_arg[0]) for call_arg in mock_rmtree.call_args_list), \
        "Cleanup not called for generated images directory."

    # Check if FID value was logged.
    assert "FID: 123.4560" in caplog.text, "FID value not logged correctly."

def test_evaluate_model_fid_calculation_error_handling(
    tmp_path: Path,
    dummy_image_tensors: Dict[str, torch.Tensor],
    fid_transform_fixture: T.Compose,
    mocker: Any,
    caplog: Any
) -> None:
    """
    Test that `evaluate_model` handles errors during FID calculation gracefully
    by logging the error and still attempting cleanup.

    Given `calculate_metrics` mocked to raise an exception,
    When `evaluate_model` is called,
    Then an error message should be logged, and temporary directories should still
    be attempted to be removed.
    """
    real_img: torch.Tensor = dummy_image_tensors["real"]
    gen_img: torch.Tensor = dummy_image_tensors["perfect"]

    # Mock `calculate_metrics` to raise an exception, simulating a failure.
    mock_calculate_metrics: MagicMock = mocker.patch(
        'Synthetic_Image_Generator.evaluate.calculate_metrics',
        side_effect=Exception("FID calculation failed!")
    )

    # Mock `shutil.rmtree` to ensure cleanup still happens and can be asserted.
    mock_rmtree: MagicMock = mocker.patch('shutil.rmtree')
    mocker.patch('pathlib.Path.mkdir') # Mock mkdir as well

    with caplog.at_level(logging.ERROR): # Capture ERROR level logs
        evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

    assert "An error occurred during FID calculation: FID calculation failed!" in caplog.text, \
        "Expected error message for FID calculation failure not logged."
    assert "Please ensure 'torch_fidelity' is installed" in caplog.text, \
        "Expected installation suggestion not logged."

    # Assert temporary directories were still cleaned up in the `finally` block.
    assert mock_rmtree.call_count == 2, "shutil.rmtree should still be called twice for cleanup."

def test_evaluate_model_empty_real_images(
    generated_images: Dict[str, torch.Tensor], # Using dict for consistency with other fixtures
    fid_transform_fixture: T.Compose,
    caplog: Any
) -> None:
    """
    Test that `evaluate_model` handles an empty `real_images_batch_tensor` gracefully
    by skipping pixel-wise metrics and logging a warning.

    Given an empty real images tensor,
    When `evaluate_model` is called,
    Then a warning should be logged, and pixel-wise metrics should be skipped.
    FID calculation might still proceed if generated images are present.
    """
    # Create an empty tensor for real images.
    empty_real_tensor: torch.Tensor = torch.empty(0, 1, 64, 64)
    # Use a dummy generated images tensor for the test.
    dummy_gen_tensor: torch.Tensor = generated_images["perfect"] # Using a valid generated image

    # Mock FID calculation to prevent errors from missing image files, as it might still run.
    with patch('Synthetic_Image_Generator.evaluate.calculate_metrics') as mock_calculate_metrics:
        mock_calculate_metrics.return_value = {'fid': 0.0} # Dummy FID return

        with caplog.at_level(logging.ERROR): # Capture ERROR level logs
            evaluate_model(empty_real_tensor, dummy_gen_tensor, fid_transform_fixture, num_compare=1)

        assert "Could not retrieve enough real images for evaluation. Skipping pixel-wise metrics." in caplog.text, \
            "Expected warning for empty real images not logged."
        
        # Depending on `evaluate_model`'s exact flow, FID might still be attempted.
        # This test primarily ensures the pixel-wise metrics are skipped.
        # If evaluate_model exits early, mock_calculate_metrics might not be called.
        # In the current implementation, it will proceed to FID if generated_images is not empty.
        mock_calculate_metrics.assert_called_once() # Should be called if generated_images is not empty.
