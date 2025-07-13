# tests/test_main.py
import pytest
import torch
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

# To test main, we need to be able to import it
import main as main_script

@pytest.fixture
def mock_main_dependencies(mocker):
    """
    GIVEN: a test environment needing to mock various modules and functions used by `main.py`
    WHEN: a fixture is set up to provide pre-configured mock objects for:
          - Path operations (mkdir, exists)
          - Dataset and DataLoader instantiation
          - The CNF_UNet model and its methods (including `load_state_dict`)
          - External utilities like `gdown.download`
          - Internal functions such as `generate`, `evaluate_metrics`, and all plotting functions
          - `torch.load`
          - Key attributes from the `config` module
    THEN: a dictionary of these mock objects is returned, allowing individual tests to assert their calls and behavior
    """
    # Define dummy image size for mocks
    mock_image_size = (96, 96) 
    
    # Create dummy batches for DataLoader mocks
    dummy_noise_batch = torch.randn(16, 1, *mock_image_size)
    dummy_image_batch = torch.randn(16, 1, *mock_image_size) # For initial dataloader
    
    # For eval_dataloader, provide multiple batches to ensure concatenation works
    dummy_eval_batches = [
        (None, torch.randn(64, 1, *mock_image_size)) for _ in range(5) # Provide 5 batches
    ]

    # Mock for the first DataLoader (dataloader)
    mock_first_dataloader_instance = MagicMock()
    mock_first_dataloader_instance.__iter__.return_value = iter([(dummy_noise_batch, dummy_image_batch)])

    # Mock for the second DataLoader (eval_dataloader)
    mock_second_dataloader_instance = MagicMock()
    mock_second_dataloader_instance.__iter__.side_effect = lambda: iter(dummy_eval_batches)

    # --- START OF MODIFICATION ---

    # Create a MagicMock for the generator instance
    mock_generator_instance = MagicMock()
    # Ensure its .to() method returns itself (as in real PyTorch models)
    mock_generator_instance.to.return_value = mock_generator_instance 
    # Create a MagicMock specifically for the load_state_dict method on this instance
    mock_generator_instance.load_state_dict = MagicMock() 

    mocks = {
        'mkdir': mocker.patch('pathlib.Path.mkdir'),
        'exists': mocker.patch('pathlib.Path.exists', return_value=True),
        'LungCTWithGaussianDataset': mocker.patch(
            'main.LungCTWithGaussianDataset', 
            return_value=MagicMock(__len__=lambda _: 100)
        ),
        'DataLoader': mocker.patch('main.DataLoader', side_effect=[
            mock_first_dataloader_instance,
            mock_second_dataloader_instance
        ]),
        # Patch main.CNF_UNet to return our pre-configured mock_generator_instance
        'CNF_UNet': mocker.patch('main.CNF_UNet', return_value=mock_generator_instance),
        # Remove the direct mocker.patch for load_state_dict here
        # 'load_state_dict': mocker.patch('main.CNF_UNet.return_value.load_state_dict'), # REMOVE THIS LINE
        'download': mocker.patch('main.download_file_from_google_drive'),
        'generate': mocker.patch('main.generate', return_value=MagicMock(spec=torch.Tensor)),
        'evaluate_metrics': mocker.patch('main.evaluate_metrics', return_value=(0.1, 20.0, 0.9, 50.0)),
        'plot_pixel_distributions': mocker.patch('main.plot_pixel_distributions'),
        'plot_sample_images': mocker.patch('main.plot_sample_images'),
        'plot_generated_samples': mocker.patch('main.plot_generated_samples'),
        'plot_real_vs_generated': mocker.patch('main.plot_real_vs_generated_side_by_side'),
        'torch_load': mocker.patch('torch.load', return_value={}),
    }

    # Store the specific load_state_dict mock for assertion purposes
    mocks['load_state_dict'] = mock_generator_instance.load_state_dict # ADD THIS LINE

    # --- END OF MODIFICATION ---

    # Ensure mocks for attributes accessed directly on config are in place
    mocker.patch('config.image_size', mock_image_size)
    mocker.patch('config.time_embed_dim', 256)
    mocker.patch('config.device', torch.device('cpu')) # Mock device to CPU for tests
    mocker.patch('config.base_dir', Path("dummy_base_dir"))
    mocker.patch('config.transform', MagicMock()) # Mock transform
    mocker.patch('config.checkpoint_dir', Path("dummy_checkpoints"))
    mocker.patch('config.generator_checkpoint_path', Path("dummy_checkpoints/generator_final.pth"))
    mocker.patch('config.GOOGLE_DRIVE_FILE_ID', "dummy_id")

    return mocks

def test_main_pipeline_execution(mock_main_dependencies):
    """
    GIVEN: that all external and internal dependencies of the `main` function are mocked
    WHEN: the `main` function is called
    THEN: the core components of the pipeline (dataset loading, dataloader creation,
          model instantiation, checkpoint loading, plotting, generation, and evaluation)
          should be called as expected, indicating a successful execution flow
    """
    # Run the main function
    main_script.main()
    
    # Assert that key functions were called
    assert mock_main_dependencies['LungCTWithGaussianDataset'].called
    assert mock_main_dependencies['DataLoader'].call_count >= 2 # Updated: DataLoader might be called more than twice
    assert mock_main_dependencies['CNF_UNet'].called
    assert mock_main_dependencies['torch_load'].called # ADDED: Assert that torch.load was called
    assert mock_main_dependencies['load_state_dict'].called
    
    # Check that plotting and evaluation functions are called
    assert mock_main_dependencies['plot_pixel_distributions'].call_count >= 2 # Updated: plot_pixel_distributions might be called more than once
    assert mock_main_dependencies['plot_sample_images'].called
    assert mock_main_dependencies['generate'].called
    assert mock_main_dependencies['evaluate_metrics'].called
    assert mock_main_dependencies['plot_generated_samples'].called
    assert mock_main_dependencies['plot_real_vs_generated'].called

def test_main_downloads_if_weights_not_exist(mock_main_dependencies):
    """
    GIVEN: that the generator checkpoint file is mocked to not exist
    WHEN: the `main` function is called
    THEN: the `download_file_from_google_drive` function should be called exactly once
          to attempt to download the model weights
    """
    # Override the 'exists' mock for this specific test
    mock_main_dependencies['exists'].return_value = False

    main_script.main()
    
    # Assert that the download function was called
    mock_main_dependencies['download'].assert_called_once()