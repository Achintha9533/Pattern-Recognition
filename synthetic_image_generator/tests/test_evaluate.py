# tests/test_evaluate.py
import torch
import pytest
from unittest.mock import patch

# Adjust import path
from evaluate import calculate_mse_psnr_ssim, evaluate_metrics

def test_calculate_mse_psnr_ssim():
    """Test metric calculations with known tensor values."""
    # Images are in [0, 1] range for this calculation
    real_images = torch.ones(2, 1, 32, 32) * 0.5
    gen_images = torch.ones(2, 1, 32, 32) * 0.75
    
    # Mock SSIM as it requires an external library
    with patch('evaluate.ssim', return_value=torch.tensor(0.9)) as mock_ssim:
        mse, psnr, ssim = calculate_mse_psnr_ssim(real_images, gen_images)

    expected_mse = torch.mean((real_images - gen_images) ** 2).item()
    expected_psnr = 10 * torch.log10(1 / torch.tensor(expected_mse)).item()

    assert isinstance(mse, float)
    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
    assert pytest.approx(mse, 0.001) == expected_mse
    assert pytest.approx(psnr, 0.01) == expected_psnr
    assert pytest.approx(ssim, 0.01) == 0.9 # Using pytest.approx for ssim as well

def test_evaluate_metrics_pipeline(mocker, mock_generator):
    """Test the main evaluate_metrics function by mocking its dependencies."""
    num_samples = 8
    
    # Mock dataloader to return dummy batches of real images [-1, 1]
    dummy_real_images = torch.rand(num_samples, 1, 96, 96) * 2 - 1
    mock_dataloader = [(None, dummy_real_images)]

    # Mock the `generate` function to return predictable generated images [-1, 1]
    dummy_gen_images = torch.rand(num_samples, 1, 96, 96) * 2 - 1
    mocker.patch('evaluate.generate', return_value=dummy_gen_images)
    
    # Mock the `calculate_metrics` from torch_fidelity
    mock_fid_result = {'frechet_inception_distance': 50.0}
    mocker.patch('evaluate.calculate_metrics', return_value=mock_fid_result)
    
    # Mock the temporary directory functions to avoid filesystem writes
    mocker.patch('evaluate.save_images_to_temp_dir')
    mocker.patch('shutil.rmtree')

    # Mock the ssim calculation within the helper
    mocker.patch('evaluate.ssim', return_value=torch.tensor(0.85))

    mse, psnr, ssim, fid = evaluate_metrics(
        generator=mock_generator,
        eval_dataloader=mock_dataloader,
        num_generated_samples=num_samples,
        steps=10
    )
    
    assert isinstance(mse, float)
    assert isinstance(psnr, float)
    assert isinstance(ssim, float)
    assert fid == 50.0
    assert pytest.approx(ssim, 0.01) == 0.85 # Using pytest.approx for ssim