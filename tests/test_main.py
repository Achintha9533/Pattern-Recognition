# tests/test_main.py

import pytest
from unittest.mock import patch, MagicMock
import torch
import logging
from typing import Any, Dict, Generator

# Import the main function
from Synthetic_Image_Generator.main import main
# Import config and other modules for type hinting and assertion clarity
from Synthetic_Image_Generator import config
from Synthetic_Image_Generator import dataset
from Synthetic_Image_Generator import transforms
from Synthetic_Image_Generator import model
from Synthetic_Image_Generator import train
from Synthetic_Image_Generator import generate
from Synthetic_Image_Generator import evaluate
from Synthetic_Image_Generator import visualize

"""
Test suite for the main application entry point.

This module contains high-level integration tests for the `main` function,
verifying that it orchestrates the various components (data loading, model setup,
training, generation, evaluation, and visualization) in the correct sequence.
Dependencies are heavily mocked to isolate the `main` function's logic.
"""

# Suppress most logging during tests to keep output clean, but allow critical messages.
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def mock_dependencies(mocker: Any) -> None:
    """
    Fixture to mock all external and internal dependencies of the `main()` function.

    This ensures that the `main` function's orchestration logic is tested in isolation,
    without requiring actual file I/O, heavy computations, or external API calls.
    All mocked components return predictable values or have their calls recorded.

    Args:
        mocker (Any): pytest-mock's mocker fixture for easy mocking.
    """
    # Mock config values (important for setup and ensuring predictable test behavior)
    mocker.patch.object(config, 'BASE_DIR', MagicMock(exists=lambda: True, is_dir=lambda: True))
    mocker.patch.object(config, 'CHECKPOINT_DIR', MagicMock(mkdir=lambda parents, exist_ok: None))
    mocker.patch.object(config, 'GENERATOR_CHECKPOINT_PATH', MagicMock())
    mocker.patch.object(config, 'IMAGE_SIZE', (16, 16))
    mocker.patch.object(config, 'BATCH_SIZE', 2)
    mocker.patch.object(config, 'EPOCHS', 1) # Shorten training for test speed
    mocker.patch.object(config, 'NUM_WORKERS', 0)
    mocker.patch.object(config, 'GENERATION_STEPS', 10)
    mocker.patch.object(config, 'NUM_GENERATED_SAMPLES', 4)
    mocker.patch.object(config, 'NUM_IMAGES_PER_FOLDER', 1)
    mocker.patch.object(config, 'NUM_BATCHES_FOR_DIST_PLOT', 1)

    # Mock PyTorch core components
    mocker.patch('torch.device', return_value='cpu') # Force CPU for consistent test environment
    mocker.patch('torch.optim.Adam') # Mock optimizer constructor
    mocker.patch('torch.save') # Mock model saving
    # Mock random tensor generation for predictable inputs
    mocker.patch('torch.randn', return_value=torch.zeros(1, 1, 16, 16))
    mocker.patch('torch.full', return_value=torch.zeros(1)) # For `torch.full((batch_size,), t_val)` in generate

    # Mock dataset and dataloader
    mock_dataset_class = mocker.patch.object(dataset, 'LungCTWithGaussianDataset')
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.__len__.return_value = 10 # Dummy length for dataset
    # Mock `__getitem__` to return dummy noise and image for dataloader's internal use
    mock_dataset_instance.__getitem__.return_value = (
        torch.randn(1, 16, 16), # Dummy noise
        torch.randn(1, 16, 16)  # Dummy image
    )
    mock_dataset_class.return_value = mock_dataset_instance

    mock_dataloader_class = mocker.patch('torch.utils.data.DataLoader')
    # Mock dataloader to return a single batch for simplicity in tests
    mock_dataloader_class.return_value = [
        (torch.randn(2, 1, 16, 16), torch.randn(2, 1, 16, 16)) # Dummy z0, x1_real batch
    ]

    # Mock transforms functions
    mocker.patch.object(transforms, 'get_transforms', return_value=MagicMock())
    mocker.patch.object(transforms, 'get_fid_transforms', return_value=MagicMock())

    # Mock model components
    mock_generator_class = mocker.patch.object(model, 'CNF_UNet')
    mock_generator_instance = MagicMock()
    mock_generator_instance.to.return_value = mock_generator_instance # Allow .to(device) chaining
    mock_generator_instance.train.return_value = None
    mock_generator_instance.eval.return_value = None
    mock_generator_instance.state_dict.return_value = {} # For torch.save
    mock_generator_instance.parameters.return_value = [] # For optimizer init
    # Mock the forward pass of the generator to return a dummy tensor
    mock_generator_instance.return_value = torch.randn(2, 1, 16, 16)
    mock_generator_class.return_value = mock_generator_instance


    # Mock train, generate, evaluate, visualize functions
    mocker.patch.object(train, 'train_model', return_value={'gen_flow_losses': [0.1, 0.05]})
    mocker.patch.object(generate, 'generate_images', return_value=torch.randn(4, 1, 16, 16))
    mocker.patch.object(evaluate, 'evaluate_model')

    # Mock all visualization functions
    mocker.patch.object(visualize, 'plot_pixel_distributions')
    mocker.patch.object(visualize, 'plot_sample_images_and_noise')
    mocker.patch.object(visualize, 'plot_training_losses')
    mocker.patch.object(visualize, 'plot_generated_pixel_distribution_comparison')
    mocker.patch.object(visualize, 'plot_sample_generated_images')
    mocker.patch.object(visualize, 'plot_real_vs_generated_side_by_side')

def test_main_function_flow(mock_dependencies: Any) -> None:
    """
    Test that the `main` function calls all expected sub-functions in the correct order.

    This is a high-level integration test that verifies the orchestration logic
    of the main application flow. All external interactions are mocked.

    Given all dependencies are mocked,
    When the `main` function is called,
    Then all expected sub-functions should be called with appropriate arguments
    and in the correct sequence.
    """
    # Call the main function
    main()

    # Assertions to check if key functions were called as expected.
    # Note: config values themselves are mocked, so we assert on their usage by other modules.

    # Check device setup
    torch.device.assert_called_once_with("cuda" if torch.cuda.is_available() else "cpu")

    # Check transforms setup
    transforms.get_transforms.assert_called_once_with(config.IMAGE_SIZE)
    transforms.get_fid_transforms.assert_called_once()

    # Check dataset and dataloader setup
    dataset.LungCTWithGaussianDataset.assert_called_once_with(
        base_dir=config.BASE_DIR,
        transform=transforms.get_transforms.return_value,
        num_images_per_folder=config.NUM_IMAGES_PER_FOLDER,
        image_size=config.IMAGE_SIZE
    )
    torch.utils.data.DataLoader.assert_called_once_with(
        dataset.LungCTWithGaussianDataset.return_value,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # Check model and optimizer initialization
    model.CNF_UNet.assert_called_once_with(time_embed_dim=256)
    model.CNF_UNet.return_value.to.assert_called_once_with(torch.device.return_value)
    torch.optim.Adam.assert_called_once() # Arguments are internal to Adam mock

    # Check initial visualization calls
    visualize.plot_pixel_distributions.assert_called_once()
    visualize.plot_sample_images_and_noise.assert_called_once()

    # Check training call
    train.train_model.assert_called_once_with(
        generator_model=model.CNF_UNet.return_value.to.return_value,
        dataloader=torch.utils.data.DataLoader.return_value,
        optimizer_gen=torch.optim.Adam.return_value,
        epochs=config.EPOCHS,
        device=torch.device.return_value # Should be 'cpu' based on mock
    )

    # Check model saving
    torch.save.assert_called_once_with(
        model.CNF_UNet.return_value.state_dict.return_value,
        config.GENERATOR_CHECKPOINT_PATH
    )

    # Check training losses plot
    train_model_return_value = train.train_model.return_value
    visualize.plot_training_losses.assert_called_once_with(train_model_return_value)

    # Check generation call
    generate.generate_images.assert_called_once()
    generate.generate_images.assert_called_with(
        model=model.CNF_UNet.return_value.to.return_value,
        initial_noise=torch.any_instance_of(torch.Tensor), # Check type, content is mocked
        steps=config.GENERATION_STEPS,
        device=torch.device.return_value
    )

    # Check evaluation call
    evaluate.evaluate_model.assert_called_once()
    evaluate.evaluate_model.assert_called_with(
        real_images_batch_tensor=torch.any_instance_of(torch.Tensor),
        generated_images=torch.any_instance_of(torch.Tensor),
        fid_transform=transforms.get_fid_transforms.return_value,
        num_compare=config.NUM_GENERATED_SAMPLES
    )

    # Check post-evaluation visualization calls
    visualize.plot_generated_pixel_distribution_comparison.assert_called_once()
    visualize.plot_sample_generated_images.assert_called_once()
    visualize.plot_real_vs_generated_side_by_side.assert_called_once()

    logger.info("Main function orchestration test passed.")

def test_main_function_handles_dataset_loading_error(mocker: Any, caplog: Any) -> None:
    """
    Test that `main()` gracefully handles a `ValueError` during dataset loading and exits.

    Given `LungCTWithGaussianDataset` is mocked to raise a `ValueError` during initialization,
    When the `main` function is called,
    Then a critical error message should be logged, and subsequent functions
    (training, generation, evaluation) should not be called.
    """
    # Mock `LungCTWithGaussianDataset` to raise an error during initialization.
    mocker.patch.object(
        dataset,
        'LungCTWithGaussianDataset',
        side_effect=ValueError("Test: No data found for dataset.")
    )

    # Ensure other critical functions are NOT called by mocking them.
    mock_train = mocker.patch.object(train, 'train_model')
    mock_generate = mocker.patch.object(generate, 'generate_images')
    mock_evaluate = mocker.patch.object(evaluate, 'evaluate_model')
    mock_visualize_pixel_dist = mocker.patch.object(visualize, 'plot_pixel_distributions')

    with caplog.at_level(logging.CRITICAL): # Capture critical logs for assertion
        main()

    assert "Failed to load dataset: Test: No data found for dataset." in caplog.text, \
        "Expected critical log message for dataset loading error not found."
    mock_train.assert_not_called(), "Training should not be called after dataset loading error."
    mock_generate.assert_not_called(), "Generation should not be called after dataset loading error."
    mock_evaluate.assert_not_called(), "Evaluation should not be called after dataset loading error."
    mock_visualize_pixel_dist.assert_not_called(), "Initial visualization should not be called after dataset loading error."
