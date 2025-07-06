# tests/test_evaluate.py

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import os
import shutil
import logging

# Import modules from your package
from Synthetic_Image_Generator.evaluate import evaluate_model
from Synthetic_Image_Generator.transforms import get_fid_transforms # Needed for fid_transform fixture

# Suppress logging during tests
logging.getLogger().setLevel(logging.CRITICAL)

@pytest.fixture
def dummy_image_tensors():
    """Fixture for dummy real and generated image tensors (normalized to [-1, 1])."""
    # Create simple patterns for predictable metric calculation
    real_img = torch.ones(1, 1, 10, 10) * 0.8 # All pixels 0.8
    gen_img_perfect = torch.ones(1, 1, 10, 10) * 0.8 # Perfect match
    gen_img_noisy = torch.ones(1, 1, 10, 10) * 0.7 + torch.randn(1, 1, 10, 10) * 0.1 # Some noise
    gen_img_zeros = torch.zeros(1, 1, 10, 10) * -1.0 # All pixels -1.0 (after normalization)

    return {
        "real": real_img,
        "perfect": gen_img_perfect,
        "noisy": gen_img_noisy,
        "zeros": gen_img_zeros
    }

@pytest.fixture
def fid_transform_fixture():
    """Fixture for the FID transform."""
    return get_fid_transforms()

def test_evaluate_model_pixel_metrics_perfect_match(dummy_image_tensors, fid_transform_fixture, caplog):
    """
    Test MSE, PSNR, SSIM for a perfect match between real and generated images.
    Expect MSE close to 0, PSNR very high, SSIM close to 1.
    """
    real_img = dummy_image_tensors["real"]
    gen_img = dummy_image_tensors["perfect"]
    
    # Mock calculate_metrics to prevent actual FID computation
    with patch('Synthetic_Image_Generator.evaluate.calculate_metrics') as mock_calculate_metrics:
        # We don't need a return value for this test as we're only checking pixel metrics
        mock_calculate_metrics.return_value = {'fid': 0.0}
        
        evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

        # Check if the expected log messages for pixel metrics are present and values are close
        assert "Average MSE:" in caplog.text
        assert "Average PSNR:" in caplog.text
        assert "Average SSIM:" in caplog.text

        # Extract values from log (or calculate directly if needed for precision)
        # For perfect match, MSE should be ~0, PSNR very high, SSIM ~1
        # Convert to [0,1] range for metric calculation: (0.8 + 1)/2 = 0.9
        # MSE for perfect match should be 0.0
        # PSNR for perfect match should be inf (or very high)
        # SSIM for perfect match should be 1.0

        # Since we are using np.mean on a single value, it's just the value itself
        real_np = ((real_img.squeeze().numpy() + 1) / 2)
        gen_np_perfect = ((gen_img.squeeze().numpy() + 1) / 2)

        expected_mse = mean_squared_error(real_np, gen_np_perfect)
        expected_psnr = peak_signal_noise_ratio(real_np, gen_np_perfect, data_range=1)
        expected_ssim = structural_similarity(real_np, gen_np_perfect, data_range=1)

        # Check against logged output (approximate string matching)
        assert f"Average MSE: {expected_mse:.6f}" in caplog.text
        # PSNR for perfect match is inf, so check for a very high number or "inf"
        assert "Average PSNR: inf dB" in caplog.text or f"Average PSNR: {expected_psnr:.6f} dB" in caplog.text
        assert f"Average SSIM: {expected_ssim:.6f}" in caplog.text


def test_evaluate_model_pixel_metrics_zeros_vs_real(dummy_image_tensors, fid_transform_fixture, caplog):
    """
    Test MSE, PSNR, SSIM for a generated image of all zeros against a real image.
    Expect higher MSE, lower PSNR, lower SSIM.
    """
    real_img = dummy_image_tensors["real"] # All 0.8
    gen_img = dummy_image_tensors["zeros"] # All -1.0

    with patch('Synthetic_Image_Generator.evaluate.calculate_metrics') as mock_calculate_metrics:
        mock_calculate_metrics.return_value = {'fid': 1000.0}

        evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

        real_np = ((real_img.squeeze().numpy() + 1) / 2) # 0.9
        gen_np_zeros = ((gen_img.squeeze().numpy() + 1) / 2) # 0.0

        expected_mse = mean_squared_error(real_np, gen_np_zeros) # (0.9 - 0.0)^2 = 0.81
        expected_psnr = peak_signal_noise_ratio(real_np, gen_np_zeros, data_range=1)
        expected_ssim = structural_similarity(real_np, gen_np_zeros, data_range=1)

        assert f"Average MSE: {expected_mse:.6f}" in caplog.text
        assert f"Average PSNR: {expected_psnr:.6f} dB" in caplog.text
        assert f"Average SSIM: {expected_ssim:.6f}" in caplog.text
        
        # Verify the values are in the expected range (e.g., MSE > 0, PSNR < inf, SSIM < 1)
        assert expected_mse > 0.1
        assert expected_psnr < 20 # Should be low
        assert expected_ssim < 0.5 # Should be low


def test_evaluate_model_fid_calculation_success(tmp_path, dummy_image_tensors, fid_transform_fixture, mocker, caplog):
    """
    Test that FID calculation is attempted and temporary directories are cleaned up.
    Mocks calculate_metrics to control its behavior.
    """
    real_img = dummy_image_tensors["real"]
    gen_img = dummy_image_tensors["perfect"]

    # Mock calculate_metrics to return a dummy FID value
    mock_calculate_metrics = mocker.patch(
        'Synthetic_Image_Generator.evaluate.calculate_metrics',
        return_value={'fid': 123.456}
    )

    # Mock Path.mkdir to prevent actual directory creation (though tmp_path handles it)
    mocker.patch('pathlib.Path.mkdir')
    # Mock shutil.rmtree to prevent actual directory deletion
    mock_rmtree = mocker.patch('shutil.rmtree')

    # Run evaluation
    evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

    # Assert calculate_metrics was called with the correct temporary paths
    mock_calculate_metrics.assert_called_once()
    args, kwargs = mock_calculate_metrics.call_args
    assert 'input1' in kwargs and 'input2' in kwargs
    assert 'fid_real_images' in kwargs['input1']
    assert 'fid_generated_images' in kwargs['input2']
    assert kwargs['fid'] is True

    # Assert temporary directories were attempted to be cleaned up
    assert mock_rmtree.call_count == 2
    assert any("fid_real_images" in str(call_arg[0]) for call_arg in mock_rmtree.call_args_list)
    assert any("fid_generated_images" in str(call_arg[0]) for call_arg in mock_rmtree.call_args_list)

    # Check if FID value was logged
    assert "FID: 123.4560" in caplog.text

def test_evaluate_model_fid_calculation_error_handling(tmp_path, dummy_image_tensors, fid_transform_fixture, mocker, caplog):
    """
    Test that FID calculation handles errors gracefully and logs them.
    """
    real_img = dummy_image_tensors["real"]
    gen_img = dummy_image_tensors["perfect"]

    # Mock calculate_metrics to raise an exception
    mock_calculate_metrics = mocker.patch(
        'Synthetic_Image_Generator.evaluate.calculate_metrics',
        side_effect=Exception("FID calculation failed!")
    )

    # Mock shutil.rmtree to ensure cleanup still happens
    mock_rmtree = mocker.patch('shutil.rmtree')

    # Run evaluation
    evaluate_model(real_img, gen_img, fid_transform_fixture, num_compare=1)

    # Assert error message was logged
    assert "An error occurred during FID calculation: FID calculation failed!" in caplog.text
    assert "Please ensure 'torch_fidelity' is installed" in caplog.text

    # Assert temporary directories were still cleaned up in finally block
    assert mock_rmtree.call_count == 2

def test_evaluate_model_empty_real_images(generated_images, fid_transform_fixture, caplog):
    """
    Test that evaluate_model handles an empty real_images_batch_tensor gracefully.
    """
    # Create an empty tensor for real images
    empty_real_tensor = torch.empty(0, 1, 64, 64)
    # Use a dummy generated images tensor
    dummy_gen_tensor = torch.randn(1, 1, 64, 64)

    # Mock FID calculation to prevent errors from missing image files
    with patch('Synthetic_Image_Generator.evaluate.calculate_metrics') as mock_calculate_metrics:
        mock_calculate_metrics.return_value = {'fid': 0.0} # Not actually used in this branch

        evaluate_model(empty_real_tensor, dummy_gen_tensor, fid_transform_fixture, num_compare=1)

        assert "Could not retrieve enough real images for evaluation. Skipping pixel-wise metrics." in caplog.text
        # FID calculation should still attempt to run if generated images are present,
        # but the pixel-wise metrics part should be skipped.
        # The current implementation will still attempt FID if generated_images is not empty.
        # This test primarily checks the pixel-wise skip.