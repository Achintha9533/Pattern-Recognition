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
Dependencies are heavily mocked to isolate the `main` function's logic and
ensure tests are fast and reliable without needing actual data or full model training.
"""

# Suppress most logging during tests to keep output clean, but allow critical messages.
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def mock_dependencies(mocker: Any) -> None:
    """
    Fixture to mock all external and internal dependencies of the `main()` function.

    This ensures that the `main` function's orchestration logic is tested in
    isolation from actual data loading, model training, generation, evaluation,
    and visualization. It provides mock objects that return predictable values
    or track calls. This fixture is `autouse=True`, meaning it runs automatically
    before every test in this module.

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
        * Patches `torch.cuda.is_available`, `dataset.LungCTWithGaussianDataset`,
          `transforms.get_transforms`, `transforms.get_fid_transforms`, `model.CNF_UNet`,
          `train.train_model`, `generate.generate_images`, `evaluate.evaluate_model`,
          and all `visualize` functions.

    Explanation of the Theory:
        This comprehensive mocking strategy implements a form of "mocking all the way down"
        for integration tests, allowing focus on the high-level flow of the `main` function
        without the complexities and time commitment of true end-to-end testing.
    """
    # Mock torch.cuda.is_available to control device
    mocker.patch('torch.cuda.is_available', return_value=False) # Force CPU for tests

    # Mock dataset and dataloader
    mock_image_size = (64, 64)
    mock_real_image_batch = torch.randn(config.BATCH_SIZE, 1, *mock_image_size)
    mock_noise_batch = torch.randn(config.BATCH_SIZE, 1, *mock_image_size)
    mock_dataloader = MagicMock()
    mock_dataloader.__len__.return_value = 1 # Simulate one batch
    mock_dataloader.__iter__.return_value = iter([(mock_noise_batch, mock_real_image_batch)])
    
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = config.BATCH_SIZE
    mocker.patch.object(dataset, 'LungCTWithGaussianDataset', return_value=mock_dataset)
    mocker.patch('torch.utils.data.DataLoader', return_value=mock_dataloader)


    # Mock transforms
    mock_transform = MagicMock()
    mocker.patch.object(transforms, 'get_transforms', return_value=mock_transform)
    mocker.patch.object(transforms, 'get_fid_transforms', return_value=mock_transform)

    # Mock model
    mock_generator = MagicMock()
    mocker.patch.object(model, 'CNF_UNet', return_value=mock_generator)
    mocker.patch('torch.optim.Adam', return_value=MagicMock()) # Mock optimizer

    # Mock training and generation functions
    mock_train_results = {'gen_flow_losses': [0.1, 0.05]}
    mocker.patch.object(train, 'train_model', return_value=mock_train_results)
    
    # Generated images for evaluation and visualization
    mock_generated_images = torch.randn(config.NUM_GENERATED_SAMPLES, 1, *mock_image_size)
    mocker.patch.object(generate, 'generate_images', return_value=mock_generated_images)

    # Mock evaluation and visualization functions
    mocker.patch.object(evaluate, 'evaluate_model')
    mocker.patch.object(visualize, 'plot_pixel_distributions')
    mocker.patch.object(visualize, 'plot_sample_images_and_noise')
    mocker.patch.object(visualize, 'plot_training_losses')
    mocker.patch.object(visualize, 'plot_generated_pixel_distribution_comparison')
    mocker.patch.object(visualize, 'plot_sample_generated_images')
    mocker.patch.object(visualize, 'plot_real_vs_generated_side_by_side')

def test_main_successful_pipeline(mocker: Any) -> None:
    """
    Test that the `main` function orchestrates the entire workflow successfully,
    calling all expected components in the correct order.

    Given all external and internal dependencies are mocked to simulate success,
    When the `main` function is called,
    Then all key functions (dataset loading, transforms, model setup, training,
    generation, evaluation, and visualization) should be called exactly once
    with appropriate arguments.

    Args:
        mocker (Any): The `pytest-mock` fixture for patching and asserting calls.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If any mocked function is not called, called out of order,
                        or called with incorrect arguments.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the overall control flow of the `main` function, ensuring
          proper integration of `dataset`, `transforms`, `model`, `train`,
          `generate`, `evaluate`, and `visualize` modules.

    Explanation of the Theory:
        This is an integration test that verifies the correct sequencing and
        interaction between different parts of the application. By mocking
        individual components, it can pinpoint issues in the orchestration
        logic of the `main` function without being affected by the internal
        details or potential failures of the mocked parts.
    """
    # Retrieve mocks to check call counts and arguments
    mock_dataset_class = mocker.patch.object(dataset, 'LungCTWithGaussianDataset')
    mock_dataloader_class = mocker.patch('torch.utils.data.DataLoader')
    mock_get_transforms = mocker.patch.object(transforms, 'get_transforms')
    mock_get_fid_transforms = mocker.patch.object(transforms, 'get_fid_transforms')
    mock_cnf_unet = mocker.patch.object(model, 'CNF_UNet')
    mock_adam_optimizer = mocker.patch('torch.optim.Adam')
    mock_train_model = mocker.patch.object(train, 'train_model')
    mock_generate_images = mocker.patch.object(generate, 'generate_images')
    mock_evaluate_model = mocker.patch.object(evaluate, 'evaluate_model')
    mock_plot_pixel_distributions = mocker.patch.object(visualize, 'plot_pixel_distributions')
    mock_plot_sample_images_and_noise = mocker.patch.object(visualize, 'plot_sample_images_and_noise')
    mock_plot_training_losses = mocker.patch.object(visualize, 'plot_training_losses')
    mock_plot_generated_pixel_distribution_comparison = mocker.patch.object(visualize, 'plot_generated_pixel_distribution_comparison')
    mock_plot_sample_generated_images = mocker.patch.object(visualize, 'plot_sample_generated_images')
    mock_plot_real_vs_generated_side_by_side = mocker.patch.object(visualize, 'plot_real_vs_generated_side_by_side')

    main()

    # Assert that all expected components were called
    mock_dataset_class.assert_called_once_with(config.BASE_DIR, transform=mock_get_transforms.return_value, num_images_per_folder=config.NUM_IMAGES_PER_FOLDER, image_size=config.IMAGE_SIZE)
    mock_dataloader_class.assert_called_once_with(mock_dataset_class.return_value, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    mock_get_transforms.assert_called_once_with(image_size=config.IMAGE_SIZE)
    mock_get_fid_transforms.assert_called_once()
    mock_cnf_unet.assert_called_once_with(image_size=config.IMAGE_SIZE)
    mock_adam_optimizer.assert_called_once_with(mock_cnf_unet.return_value.parameters(), lr=config.G_LR)
    mock_plot_pixel_distributions.assert_called_once()
    mock_plot_sample_images_and_noise.assert_called_once()
    mock_train_model.assert_called_once_with(
        generator_model=mock_cnf_unet.return_value,
        dataloader=mock_dataloader_class.return_value,
        optimizer_gen=mock_adam_optimizer.return_value,
        epochs=config.EPOCHS,
        device=mocker.ANY # Device can be 'cpu' or 'cuda', so use ANY
    )
    mock_plot_training_losses.assert_called_once_with(mock_train_model.return_value['gen_flow_losses'])
    mock_generate_images.assert_called_once_with(
        model=mock_cnf_unet.return_value,
        initial_noise=mocker.ANY, # Noise is generated dynamically
        steps=config.GENERATION_STEPS,
        device=mocker.ANY
    )
    mock_evaluate_model.assert_called_once_with(
        real_images_batch_tensor=mocker.ANY,
        generated_images=mock_generate_images.return_value,
        fid_transform=mock_get_fid_transforms.return_value,
        num_compare=config.NUM_GENERATED_SAMPLES
    )
    mock_plot_generated_pixel_distribution_comparison.assert_called_once()
    mock_plot_sample_generated_images.assert_called_once_with(mock_generate_images.return_value)
    mock_plot_real_vs_generated_side_by_side.assert_called_once()


def test_main_handles_dataset_loading_error(mocker: Any, caplog: Any) -> None:
    """
    Test that the `main` function gracefully handles `ValueError` during dataset initialization,
    logs a critical error, and prevents subsequent training, generation, and evaluation steps.

    Given that `LungCTWithGaussianDataset` is mocked to raise a `ValueError` during initialization,
    When the `main` function is called,
    Then a critical error message should be logged, and subsequent critical functions
    like `train_model`, `generate_images`, and `evaluate_model` should not be called.

    Args:
        mocker (Any): The `pytest-mock` fixture for patching.
        caplog (Any): pytest fixture to capture log messages.

    Returns:
        None.

    Potential Exceptions Raised:
        AssertionError: If the expected critical log message is not found or
                        if any subsequent functions are unexpectedly called.

    Example of Usage:
    ```python
    # This test function itself is an example of usage.
    ```

    Relationships with Other Functions:
        * Tests the error handling and early exit logic within `main` when
          the `dataset.LungCTWithGaussianDataset` fails to initialize.

    Explanation of the Theory:
        This test verifies the application's robustness to critical initialization
        failures. By stopping execution early and providing informative logging,
        it helps prevent cascading errors and facilitates debugging.
    """
    # Mock `LungCTWithGaussianDataset` to raise an error during initialization.
    mocker.patch.object(
        dataset,
        'LungCTWithGaussianDataset',
        side_effect=ValueError("Test: No data found for dataset.")
    )

    # Ensure other critical functions are NOT called by mocking them and asserting `assert_not_called()`.
    mock_train = mocker.patch.object(train, 'train_model')
    mock_generate = mocker.patch.object(generate, 'generate_images')
    mock_evaluate = mocker.patch.object(evaluate, 'evaluate_model')
    mock_visualize_pixel_dist = mocker.patch.object(visualize, 'plot_pixel_distributions')
    mock_plot_sample_images_and_noise = mocker.patch.object(visualize, 'plot_sample_images_and_noise')


    with caplog.at_level(logging.CRITICAL): # Capture critical logs for assertion
        main()

    assert "Failed to load dataset: Test: No data found for dataset." in caplog.text, \
        "Expected critical log message for dataset loading error not found."
    mock_train.assert_not_called(), "Training should not be called after dataset loading error."
    mock_generate.assert_not_called(), "Generation should not be called after dataset loading error."
    mock_evaluate.assert_not_called(), "Evaluation should not be called after dataset loading error."
    mock_visualize_pixel_dist.assert_not_called(), "Initial pixel distribution visualization should not be called after dataset loading error."
    mock_plot_sample_images_and_noise.assert_not_called(), "Initial sample images visualization should not be called after dataset loading error."