# tests/test_load_model.py
import torch
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# The module uses dummy classes if imports fail, so we ensure the real ones are tested
# This requires `synthetic_image_generator` to be in the python path.
# A simple way is to run pytest from the root directory: `python -m pytest`
from model import CNF_UNet 
import config as app_config

# Adjust import for the function to be tested
from load_model import load_model_from_drive

@pytest.fixture
def mock_dependencies(mocker):
    """Mock external dependencies for load_model."""
    # Mock gdown download function
    mocker.patch('gdown.download')
    
    # Mock torch.load to return a state dictionary
    mock_state_dict = {'param1': torch.randn(1), 'param2': torch.randn(1)}
    # We can't mock CNF_UNet directly as it's part of the test, so we create a simple compatible model state
    simple_model = CNF_UNet(time_embed_dim=256)
    mocker.patch('torch.load', return_value=simple_model.state_dict())
    
    # Mock the CNF_UNet class to track its instantiation
    mock_model_class = MagicMock(return_value=simple_model)
    mocker.patch('load_model.CNF_UNet', new=mock_model_class)
    
    return mock_model_class

def test_load_model_from_drive_downloads_if_not_exists(mocker, tmp_path):
    """Test that the model is downloaded if the weights file doesn't exist."""
    mock_gdown = mocker.patch('gdown.download')
    mocker.patch('torch.load', return_value={})
    
    dummy_path = tmp_path / "weights.pth"
    
    with pytest.raises(RuntimeError, match="Error loading model"):
        # Expect RuntimeError because the empty state_dict won't match the model
        load_model_from_drive(
            drive_url="dummy_url",
            output_path=dummy_path,
            image_size=(96, 96), device=torch.device('cpu'),
            image_channels=1, base_channels=64, time_embed_dim=256
        )
    
    # Verify download was called
    mock_gdown.assert_called_once()

def test_load_model_from_drive_skips_download_if_exists(mocker, tmp_path):
    """Test that download is skipped if the weights file already exists."""
    mock_gdown = mocker.patch('gdown.download')
    mocker.patch('torch.load', return_value={})
    
    # Create a dummy file to simulate its existence
    dummy_path = tmp_path / "weights.pth"
    dummy_path.touch()

    with pytest.raises(RuntimeError, match="Error loading model"):
        load_model_from_drive(
            drive_url="dummy_url",
            output_path=dummy_path,
            image_size=(96, 96), device=torch.device('cpu'),
            image_channels=1, base_channels=64, time_embed_dim=256
        )

    # Verify download was NOT called
    mock_gdown.assert_not_called()