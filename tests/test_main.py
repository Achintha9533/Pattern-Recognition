# tests/test_main.py

import pytest
from unittest.mock import patch, MagicMock
import torch
import logging

# Import the main function
from Synthetic_Image_Generator.main import main

# Suppress most logging during tests to keep output clean, but allow critical
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def mock_dependencies(mocker):
    """
    Mocks all external and internal dependencies of main() to isolate its logic.
    """
    # Mock config values (important for setup)
    mocker.patch('Synthetic_Image_Generator.config.BASE_DIR', MagicMock(exists=lambda: True, is_dir=lambda: True))
    mocker.patch('Synthetic_Image_Generator.config.CHECKPOINT_DIR', MagicMock(mkdir=lambda parents, exist_ok: None))
    mocker.patch('Synthetic_Image_Generator.config.GENERATOR_CHECKPOINT_PATH', MagicMock())
    mocker.patch('Synthetic_Image_Generator.config.IMAGE_SIZE', (16, 16))
    mocker.patch('Synthetic_Image_Generator.config.BATCH_SIZE', 2)
    mocker.patch('Synthetic_Image_Generator.config.EPOCHS', 1) # Shorten training for test
    mocker.patch('Synthetic_Image_Generator.config.NUM_WORKERS', 0)
    mocker.patch('Synthetic_Image_Generator.config.GENERATION_STEPS', 10)
    mocker.patch('Synthetic_Image_Generator.config.NUM_GENERATED_SAMPLES', 4)
    mocker.patch('Synthetic_Image_Generator.config.NUM_IMAGES_PER_FOLDER', 1)
    mocker.patch('Synthetic_Image_Generator.config.NUM_BATCHES_FOR_DIST_PLOT', 1)


    # Mock PyTorch components
    mocker.patch('torch.device', return_value='cpu') # Force CPU for tests
    mocker.patch('torch.optim.Adam')
    mocker.patch('torch.save')
    mocker.patch('torch.randn', return_value=torch.zeros(1, 1, 16, 16)) # Consistent noise
    mocker.patch('torch.full_like', return_value=torch.zeros(1, 1, 16, 16)) # Consistent time

    # Mock dataset and dataloader
    mock_dataset = mocker.patch('Synthetic_Image_Generator.dataset.LungCTWithGaussianDataset')
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.__len__.return_value = 10 # Dummy length
    # Mock __getitem__ to return dummy noise and image
    mock_dataset_instance.__getitem__.return_value = (
        torch.randn(1, 16, 16), # Dummy noise
        torch.randn(1, 16, 16)  # Dummy image
    )
    mock_dataset.return_value = mock_dataset_instance

    mock_dataloader = mocker.patch('torch.utils.data.DataLoader')
    # Mock dataloader to return a single batch for simplicity
    mock_dataloader.return_value = [
        (torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)) # z0, x1_real
    ]

    # Mock transforms
    mocker.patch('Synthetic_Image_Generator.transforms.get_transforms', return_value=MagicMock())
    mocker.patch('Synthetic_Image_Generator.transforms.get_fid_transforms', return_value=MagicMock())

    # Mock model components
    mock_generator_model = mocker.patch('Synthetic_Image_Generator.model.CNF_UNet')
    mock_generator_instance = MagicMock()
    mock_generator_instance.to.return_value = mock_generator_instance # Allow .to(device) chaining
    mock_generator_instance.train.return_value = None
    mock_generator_instance.eval.return_value = None
    mock_generator_instance.state_dict.return_value = {}
    mock_generator_instance.parameters.return_value = [] # For optimizer init
    mock_generator_instance.return_value = torch.randn(2, 1, 16, 16) # Dummy output for forward pass
    mock_generator_model.return_value = mock_generator_instance


    # Mock train, generate, evaluate, visualize functions
    mocker.patch('Synthetic_Image_Generator.train.train_model', return_value={'gen_flow_losses': [0.1, 0.05]})
    mocker.patch('Synthetic_Image_Generator.generate.generate_images', return_value=torch.randn(4, 1, 16, 16))
    mocker.patch('Synthetic_Image_Generator.evaluate.evaluate_model')

    # Mock all visualization functions
    mocker.patch('Synthetic_Image_Generator.visualize.plot_pixel_distributions')
    mocker.patch('Synthetic_Image_Generator.visualize.plot_sample_images_and_noise')
    mocker.patch('Synthetic_Image_Generator.visualize.plot_training_losses')
    mocker.patch('Synthetic_Image_Generator.visualize.plot_generated_pixel_distribution_comparison')
    mocker.patch('Synthetic_Image_Generator.visualize.plot_sample_generated_images')
    mocker.patch('Synthetic_Image_Generator.visualize.plot_real_vs_generated_side_by_side')

def test_main_function_flow(mock_dependencies):
    """
    Test that the main function calls all expected sub-functions in the correct order.
    This is an integration test at a high level, verifying orchestration.
    """
    # Call the main function
    main()

    # Assertions to check if key functions were called
    # Config values are mocked, so no direct call to config module itself,
    # but its values are used in other mocked calls.

    # Check device setup
    torch.device.assert_called_once()

    # Check transforms setup
    Synthetic_Image_Generator.transforms.get_transforms.assert_called_once_with(
        Synthetic_Image_Generator.config.IMAGE_SIZE
    )
    Synthetic_Image_Generator.transforms.get_fid_transforms.assert_called_once()

    # Check dataset and dataloader setup
    Synthetic_Image_Generator.dataset.LungCTWithGaussianDataset.assert_called_once()
    Synthetic_Image_Generator.dataset.LungCTWithGaussianDataset.return_value.__len__.assert_called() # Check if len() was called
    torch.utils.data.DataLoader.assert_called_once()

    # Check model and optimizer initialization
    Synthetic_Image_Generator.model.CNF_UNet.assert_called_once()
    torch.optim.Adam.assert_called_once()
    Synthetic_Image_Generator.model.CNF_UNet.return_value.to.assert_called_once()

    # Check initial visualization calls
    Synthetic_Image_Generator.visualize.plot_pixel_distributions.assert_called_once()
    Synthetic_Image_Generator.visualize.plot_sample_images_and_noise.assert_called_once()

    # Check training call
    Synthetic_Image_Generator.train.train_model.assert_called_once()
    Synthetic_Image_Generator.train.train_model.assert_called_with(
        generator_model=Synthetic_Image_Generator.model.CNF_UNet.return_value.to.return_value,
        dataloader=Synthetic_Image_Generator.dataloader.DataLoader.return_value,
        optimizer_gen=torch.optim.Adam.return_value,
        epochs=Synthetic_Image_Generator.config.EPOCHS,
        device='cpu' # Because we mocked torch.device to return 'cpu'
    )

    # Check model saving
    torch.save.assert_called_once_with(
        Synthetic_Image_Generator.model.CNF_UNet.return_value.state_dict.return_value,
        Synthetic_Image_Generator.config.GENERATOR_CHECKPOINT_PATH
    )

    # Check training losses plot
    Synthetic_Image_Generator.visualize.plot_training_losses.assert_called_once_with({'gen_flow_losses': [0.1, 0.05]})

    # Check generation call
    Synthetic_Image_Generator.generate.generate_images.assert_called_once()
    Synthetic_Image_Generator.generate.generate_images.assert_called_with(
        model=Synthetic_Image_Generator.model.CNF_UNet.return_value.to.return_value,
        initial_noise=torch.zeros(4, 1, 16, 16), # Based on mocked torch.randn
        steps=Synthetic_Image_Generator.config.GENERATION_STEPS,
        device='cpu'
    )

    # Check evaluation call
    Synthetic_Image_Generator.evaluate.evaluate_model.assert_called_once()
    Synthetic_Image_Generator.evaluate.evaluate_model.assert_called_with(
        real_images_batch_tensor=torch.any_instance_of(torch.Tensor), # Check type, not exact content
        generated_images=torch.any_instance_of(torch.Tensor),
        fid_transform=Synthetic_Image_Generator.transforms.get_fid_transforms.return_value,
        num_compare=Synthetic_Image_Generator.config.NUM_GENERATED_SAMPLES
    )

    # Check post-evaluation visualization calls
    Synthetic_Image_Generator.visualize.plot_generated_pixel_distribution_comparison.assert_called_once()
    Synthetic_Image_Generator.visualize.plot_sample_generated_images.assert_called_once()
    Synthetic_Image_Generator.visualize.plot_real_vs_generated_side_by_side.assert_called_once()

    logger.info("Main function orchestration test passed.")

def test_main_function_handles_dataset_loading_error(mocker, caplog):
    """
    Test that main() gracefully handles a ValueError during dataset loading and exits.
    """
    # Mock LungCTWithGaussianDataset to raise an error during initialization
    mocker.patch(
        'Synthetic_Image_Generator.dataset.LungCTWithGaussianDataset',
        side_effect=ValueError("Test: No data found for dataset.")
    )

    # Ensure other critical functions are NOT called
    mock_train = mocker.patch('Synthetic_Image_Generator.train.train_model')
    mock_generate = mocker.patch('Synthetic_Image_Generator.generate.generate_images')

    with caplog.at_level(logging.CRITICAL): # Capture critical logs
        main()

    assert "Failed to load dataset: Test: No data found for dataset." in caplog.text
    mock_train.assert_not_called()
    mock_generate.assert_not_called()