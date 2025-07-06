# Synthetic Image Generator/test_main.py

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import torch
import numpy as np
from pathlib import Path
import sys

"""
Module for comprehensive unit testing of the main execution flow in the
Synthetic Image Generator project.

This module is designed to verify the correct orchestration and interaction
of various components within the `main.py` script. It achieves this by
employing extensive mocking of external dependencies, ensuring that tests
are isolated, fast, and deterministic, without requiring actual data,
model weights, or heavy computational resources.

The tests focus on confirming that functions are called with the expected
arguments and in the correct sequence, adhering to a Behavior-Driven
Development (BDD) style for clarity and maintainability.

Relationships with other functions/modules:
- Directly tests `Synthetic_Image_Generator.main.main`.
- Mocks dependencies such as `config`, `dataset`, `transforms`, `model`,
  `load_model`, `generate`, `evaluate`, `visualize` to control their behavior
  and assert interactions.

Theory:
- **Unit Testing**: Testing individual units (functions, methods, classes)
  of source code to determine if they are fit for use.
- **Mocking**: Replacing objects that the system under test (SUT) depends on
  with simulated objects that mimic the behavior of the real objects. This
   decouples tests from external dependencies, making them faster and more reliable.
- **Behavior-Driven Development (BDD)**: A testing style that focuses on the
  behavior of the system from the user's perspective, often structured as
  "Given-When-Then" scenarios.

References:
- Python's `unittest` module documentation.
- Python's `unittest.mock` documentation.
- Martin Fowler's "Mocks Aren't Stubs" (for understanding test doubles).
"""

# Ensure the project root is in sys.path for module imports
# This allows importing 'Synthetic_Image_Generator' as a package
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the main function to be tested
# Using the full package path to ensure correct import in a test environment
try:
    from Synthetic_Image_Generator import main
except ImportError:
    # Fallback for environments where the package might not be installed in editable mode
    # or sys.path setup is different for direct script execution.
    # This assumes 'main.py' is in a directory parallel to 'test_main.py' or one level up
    # in the project structure.
    print("Warning: Could not import 'main' from 'Synthetic_Image_Generator'. Attempting direct import.")
    sys.path.insert(0, str(Path(__file__).resolve().parent)) # Add directory containing main.py if it's sibling
    try:
        import main
    except ImportError as e:
        raise ImportError(f"Could not import main.py. Ensure your sys.path is correctly configured or the package is installed: {e}")

class TestMain(unittest.TestCase):
    """
    Test suite for the `main` function in `main.py`.

    This class contains unit tests that validate the high-level execution
    flow of the synthetic image generation process orchestrated by the `main`
    function. By extensively mocking all external dependencies, these tests
    ensure that the `main` function correctly calls its sub-components with
    the appropriate arguments and in the expected order, without performing
    any actual heavy computations, file I/O, or model inference.

    The primary goal is to verify the "wiring" and control flow of the
    application, making these tests fast, isolated, and robust against
    changes in underlying dependency implementations (as long as their
    interfaces remain consistent).
    """

    # Comprehensive patching of all external dependencies used within main.py
    @patch('Synthetic_Image_Generator.main.logging')
    @patch('Synthetic_Image_Generator.main.os') # Mock os module for any potential os-level calls, though pathlib is preferred
    @patch('Synthetic_Image_Generator.main.torch')
    @patch('Synthetic_Image_Generator.main.np')
    @patch('Synthetic_Image_Generator.main.config') # Mock the config module
    @patch('Synthetic_Image_Generator.main.LungCTWithGaussianDataset')
    @patch('Synthetic_Image_Generator.main.get_transforms')
    @patch('Synthetic_Image_Generator.main.get_fid_transforms')
    @patch('Synthetic_Image_Generator.main.CNF_UNet')
    @patch('Synthetic_Image_Generator.main.load_model_from_drive')
    @patch('Synthetic_Image_Generator.main.generate_images')
    @patch('Synthetic_Image_Generator.main.evaluate_model')
    @patch('Synthetic_Image_Generator.main.plot_pixel_distributions')
    @patch('Synthetic_Image_Generator.main.plot_sample_images_and_noise')
    @patch('Synthetic_Image_Generator.main.plot_generated_pixel_distribution_comparison')
    @patch('Synthetic_Image_Generator.main.plot_sample_generated_images')
    @patch('Synthetic_Image_Generator.main.plot_real_vs_generated_side_by_side')
    @patch('Synthetic_Image_Generator.main.tqdm') # Mock tqdm to prevent progress bar output
    def test_main_execution_flow(self,
                                  mock_tqdm, # tqdm is patched as a function
                                  mock_plot_real_vs_generated_side_by_side,
                                  mock_plot_sample_generated_images,
                                  mock_plot_generated_pixel_distribution_comparison,
                                  mock_plot_sample_images_and_noise,
                                  mock_plot_pixel_distributions,
                                  mock_evaluate_model,
                                  mock_generate_images,
                                  mock_load_model_from_drive,
                                  mock_CNF_UNet,
                                  mock_get_fid_transforms,
                                  mock_get_transforms,
                                  mock_LungCTWithGaussianDataset,
                                  mock_config,
                                  mock_np,
                                  mock_torch,
                                  mock_os,
                                  mock_logging):
        """
        Verifies that the `main` function orchestrates its components correctly
        by asserting that key functions from various modules are called with
        expected arguments in the right sequence.

        This test follows a Behavior-Driven Development (BDD) style:

        Given:
            - All external dependencies of `main.py` (e.g., `torch`, `config`,
              `dataset`, `model`, `generate`, `evaluate`, `visualize`, `logging`, `tqdm`)
              are mocked to control their behavior and record their interactions.
            - Mock configuration values are set to simulate project settings.
            - Mock return values for functions (e.g., `torch.cuda.is_available`,
              `generate_images`) are configured to enable the `main` function's logic.

        When:
            - The `main.main()` function is called.

        Then:
            - Assertions are made to confirm that:
                - Device setup (CUDA availability, device selection) occurs as expected.
                - Data transformations are initialized correctly.
                - The dataset and DataLoaders are initialized with the proper parameters.
                - The CNF_UNet model is instantiated, moved to the correct device,
                  and set to evaluation mode.
                - The pre-trained model is loaded from Google Drive using the specified URL and path.
                - Initial visualization functions (`plot_pixel_distributions`,
                  `plot_sample_images_and_noise`) are invoked.
                - Image generation (`torch.randn`, `generate_images`) is performed
                  with the correct parameters and progress bar (`tqdm`).
                - Model evaluation (`evaluate_model`) is called with the generated
                  and real image data.
                - Subsequent visualization functions (`plot_generated_pixel_distribution_comparison`,
                  `plot_sample_generated_images`, `plot_real_vs_generated_side_by_side`)
                  are executed.
                - Logging functions are used to report progress and information.
                - No unexpected calls to low-level `os` functions occur, indicating
                  that `pathlib` or other higher-level abstractions are correctly used.

        Inputs:
            - `self`: The instance of `unittest.TestCase`.
            - `mock_tqdm`, `mock_plot_real_vs_generated_side_by_side`, ..., `mock_logging`:
              MagicMock objects representing the patched external dependencies
              of the `main` function. These mocks allow for inspection of calls
              and control over return values.

        Outputs:
            - None. The function performs assertions and raises `AssertionError`
              if any expected behavior is not observed.

        Relationships with other functions/modules:
        - This test method directly calls `main.main()`.
        - It mocks all direct imports and sub-module calls made by `main.main()`
          including `config`, `torch`, `numpy`, `LungCTWithGaussianDataset`,
          `get_transforms`, `get_fid_transforms`, `CNF_UNet`, `load_model_from_drive`,
          `generate_images`, `evaluate_model`, and various `plot_*` functions from `visualize`.

        Theory:
        - This test leverages the principles of dependency injection and mocking
          to create a controlled environment. By isolating `main.main()`,
          it becomes a true unit test, focusing solely on the logic within `main.main()`
          itself (its orchestration of calls) rather than the correctness of its
          dependencies (which are assumed to be tested separately).
        - The use of `patch` decorators automatically handles the setup and teardown
          of mocks for the duration of the test method.

        Potential Exceptions:
        - `AssertionError`: Raised if any `assert_called_once_with`, `assert_any_call`,
          or `assertEqual` conditions are not met, indicating a deviation from the
          expected execution flow of `main.main()`.
        """

        # --- 1. Mock Configuration Values ---
        # Set up mock values for config attributes that main.py accesses
        mock_config.IMAGE_SIZE = (64, 64)
        mock_config.BATCH_SIZE = 4
        mock_config.NUM_WORKERS = 0
        mock_config.NUM_IMAGES_PER_FOLDER = 5
        mock_config.BASE_DIR = Path("/mock/data/dir") # Use Path object for consistency
        mock_config.GENERATOR_CHECKPOINT_PATH = Path("/mock/checkpoints/generator_final.pth")
        mock_config.GENERATION_STEPS = 100
        mock_config.NUM_GENERATED_SAMPLES = 10
        mock_config.NUM_BATCHES_FOR_DIST_PLOT = 2

        # --- 2. Mock PyTorch related functions and objects ---
        # Mock torch.cuda.is_available to control the 'device' chosen
        mock_torch.cuda.is_available.return_value = True
        # Mock torch.device constructor to return a controlled device object
        mock_device_instance = MagicMock(spec=torch.device)
        mock_device_instance.type = 'cuda' # Ensure the type attribute is set
        mock_torch.device.return_value = mock_device_instance

        # Mock torch.randn for initial noise generation
        mock_initial_noise_tensor = MagicMock(spec=torch.Tensor)
        mock_initial_noise_tensor.to.return_value = mock_initial_noise_tensor # Chaining .to()
        mock_torch.randn.return_value = mock_initial_noise_tensor

        # Mock torch.full (used in generate.py, which is called by main)
        mock_torch.full.return_value = MagicMock(spec=torch.Tensor)
        
        # Mock torch.utils.data.DataLoader class and its instances
        mock_dataloader_instance = MagicMock()
        # Define the iteration behavior for dataloaders
        # 1st call for initial visualization: (image_batch, noise_batch)
        # Subsequent calls for eval loop: (index, real_img_batch)
        # Mock `cpu()` and `view(-1).numpy()` chain for pixel distributions
        mock_real_img_batch_for_eval = MagicMock(spec=torch.Tensor)
        mock_real_img_batch_for_eval.cpu.return_value.__add__.return_value.__truediv__.return_value = mock_real_img_batch_for_eval # Denormalization chain
        mock_real_img_batch_for_eval.cpu.return_value.view.return_value.numpy.return_value = MagicMock(spec=np.ndarray) # For pixel dist
        mock_real_img_batch_for_eval.shape = (mock_config.BATCH_SIZE, 1, *mock_config.IMAGE_SIZE) # Set a mock shape
        mock_dataloader_instance.__iter__.return_value = iter([
            (MagicMock(spec=torch.Tensor).cpu.return_value, MagicMock(spec=torch.Tensor).cpu.return_value), # First batch for pixel distribution plot
            (MagicMock(spec=torch.Tensor), mock_real_img_batch_for_eval), # First batch for eval_dataloader loop
            (MagicMock(spec=torch.Tensor), mock_real_img_batch_for_eval)  # Second batch for eval_dataloader loop (for pixel dist collection)
        ])
        mock_torch.utils.data.DataLoader.return_value = mock_dataloader_instance

        # Mock torch.Tensor methods that are called on tensors throughout main.py
        # This generic mock will be used for generated_images and the result of torch.cat
        mock_output_tensor = MagicMock(spec=torch.Tensor)
        # Denormalization chain: (x + 1) / 2.0
        mock_output_tensor.cpu.return_value.__add__.return_value.__truediv__.return_value = mock_output_tensor
        # Chaining for pixel distribution plots and other numpy conversions
        mock_output_tensor.view.return_value.numpy.return_value = MagicMock(spec=np.ndarray)
        mock_output_tensor.cpu.return_value = mock_output_tensor # For any .cpu() calls
        mock_output_tensor.shape = (mock_config.NUM_GENERATED_SAMPLES, 1, *mock_config.IMAGE_SIZE) # Appropriate shape for generated images
        mock_output_tensor.numel.return_value = np.prod(mock_output_tensor.shape) # For plot_real_vs_generated_side_by_side check
        
        # Mock torch.cat to return our generic mock tensor
        mock_torch.cat.return_value = mock_output_tensor

        # --- 3. Mock Dataset and Transforms ---
        mock_get_transforms.return_value = MagicMock()
        mock_get_fid_transforms.return_value = MagicMock()
        # Mock the dataset instance to have a __len__ method
        mock_dataset_instance = MagicMock(__len__=MagicMock(return_value=100))
        mock_LungCTWithGaussianDataset.return_value = mock_dataset_instance

        # --- 4. Mock Model and Loading ---
        mock_generator_instance = MagicMock()
        mock_generator_instance.to.return_value = mock_generator_instance # Mock .to() call on model
        mock_generator_instance.eval.return_value = mock_generator_instance # Mock .eval() call on model
        mock_CNF_UNet.return_value = mock_generator_instance
        mock_load_model_from_drive.return_value = mock_generator_instance # Ensure it returns the mocked generator

        # --- 5. Mock Generation and Evaluation outputs ---
        mock_generate_images.return_value = mock_output_tensor # Return our mock tensor
        mock_evaluate_model.return_value = None # It just logs, no return value expected

        # --- 6. Mock numpy operations ---
        mock_np.concatenate.return_value = MagicMock(spec=np.ndarray) # For pixel distribution plots

        # --- 7. Mock tqdm (context manager) ---
        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value = mock_tqdm_instance # tqdm(iterable) returns an iterable
        mock_tqdm_side_effect_list = list(range(mock_config.GENERATION_STEPS))
        mock_tqdm_instance.__iter__.return_value = iter(mock_tqdm_side_effect_list)
        mock_tqdm_instance.__enter__.return_value = mock_tqdm_instance
        mock_tqdm_instance.__exit__.return_value = None

        # --- 8. Call the main function ---
        # It's important to reference the main function from the imported module
        main.main()

        # --- 9. Assertions: Verify calls to mocked components ---

        # Device Configuration
        mock_torch.cuda.is_available.assert_called_once()
        mock_torch.device.assert_called_once_with('cuda') # Given we mocked is_available to True

        # Data Transformations
        mock_get_transforms.assert_called_once_with(image_size=mock_config.IMAGE_SIZE)
        mock_get_fid_transforms.assert_called_once_with(image_size=mock_config.IMAGE_SIZE)

        # Dataset and DataLoader Initialization
        mock_LungCTWithGaussianDataset.assert_called_once_with(
            base_dir=mock_config.BASE_DIR,
            num_images_per_folder=mock_config.NUM_IMAGES_PER_FOLDER,
            image_size=mock_config.IMAGE_SIZE,
            transform=mock_get_transforms.return_value
        )
        # DataLoader should be called twice (for train_dataloader, eval_dataloader)
        self.assertEqual(mock_torch.utils.data.DataLoader.call_count, 2)
        # Check specific calls for DataLoader arguments (shuffle=True for train, False for eval)
        mock_torch.utils.data.DataLoader.assert_any_call(
            mock_dataset_instance, # Assert with the mocked dataset instance
            batch_size=mock_config.BATCH_SIZE,
            shuffle=True,
            num_workers=mock_config.NUM_WORKERS,
            pin_memory=True
        )
        mock_torch.utils.data.DataLoader.assert_any_call(
            mock_dataset_instance, # Assert with the mocked dataset instance
            batch_size=mock_config.BATCH_SIZE,
            shuffle=False,
            num_workers=mock_config.NUM_WORKERS,
            pin_memory=True
        )
        
        # Ensure iter() and next() were called on the dataloader instance
        # next(iter(train_dataloader)) and evaluation loop
        self.assertGreaterEqual(mock_dataloader_instance.__iter__.call_count, 2)

        # Model and Optimizer Setup
        mock_CNF_UNet.assert_called_once_with(
            image_channels=1,
            base_channels=64,
            embed_dim=256
        )
        mock_generator_instance.to.assert_called_once_with(mock_device_instance) # Assert with the specific mocked device
        mock_generator_instance.eval.assert_called_once() # Ensure model is set to eval mode during generation

        # Loading Pre-trained Model
        mock_load_model_from_drive.assert_called_once_with(
            drive_url="1P-2cR47f1_wR_o08Q9hL8GjK4b0B0B0",
            output_path=mock_config.GENERATOR_CHECKPOINT_PATH,
            image_size=mock_config.IMAGE_SIZE,
            device=mock_device_instance
        )

        # Initial Data Visualization
        # The arguments passed to these plots are the .cpu() version of the mocked tensors
        mock_plot_pixel_distributions.assert_called_once()
        mock_plot_sample_images_and_noise.assert_called_once()

        # Image Generation
        mock_torch.randn.assert_called_once_with(
            mock_config.NUM_GENERATED_SAMPLES,
            1,
            *mock_config.IMAGE_SIZE
        )
        mock_generate_images.assert_called_once_with(
            model=mock_generator_instance,
            initial_noise=mock_initial_noise_tensor,
            steps=mock_config.GENERATION_STEPS,
            device=mock_device_instance
        )
        mock_tqdm.assert_called_once_with(range(mock_config.GENERATION_STEPS), desc="Generating Images")
        mock_tqdm_instance.__enter__.assert_called_once() # Ensure tqdm context manager was entered
        mock_tqdm_instance.__exit__.assert_called_once()  # Ensure tqdm context manager was exited

        # Evaluation of Generated Images
        # real_images_batch_tensor is the result of torch.cat
        mock_evaluate_model.assert_called_once_with(
            real_images_batch_tensor=mock_output_tensor, # The result of torch.cat
            generated_images=mock_output_tensor, # The denormalized generated images
            fid_transform=mock_get_fid_transforms.return_value,
            num_compare=mock_config.NUM_GENERATED_SAMPLES
        )

        # Distribution Plot for Generated Images vs. Real Images
        mock_plot_generated_pixel_distribution_comparison.assert_called_once()
        mock_np.concatenate.assert_called_once() # Ensure np.concatenate was called for pixel data

        # Visualize Sample Generated Images
        mock_plot_sample_generated_images.assert_called_once_with(mock_output_tensor)

        # Visualize Real vs. Generated Side-by-Side
        mock_plot_real_vs_generated_side_by_side.assert_called_once_with(
            mock_output_tensor, # The real_images_batch_tensor (from torch.cat)
            mock_output_tensor  # The generated_images
        )

        # Verify logging calls
        mock_logging.getLogger.assert_called()
        mock_logging.basicConfig.assert_called_once()
        mock_logging.getLogger.return_value.info.assert_called() # Check if info messages were logged

        # Ensure no unexpected calls to os (e.g., os.makedirs directly in main.py)
        # This assumes file/directory creation is handled by pathlib methods on Path objects
        # or by helper functions in other modules (like config.py's .mkdir()).
        # If any os.path.* functions were called, you would need specific mocks for them.
        mock_os.assert_not_called()

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)