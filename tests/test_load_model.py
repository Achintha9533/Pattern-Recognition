# Synthetic Image Generator/test_load_model.py

import unittest
from unittest.mock import patch, MagicMock, call
import torch
from pathlib import Path
import sys
import os

"""
Module for comprehensive unit testing of the model loading functionality
in the Synthetic Image Generator project.

This module focuses on testing the `load_model_from_drive` function, which is
responsible for downloading pre-trained model weights from Google Drive and
initializing a `CNF_UNet` model with these weights. The tests meticulously
mock all external dependencies (e.g., `gdown`, `torch`, file system operations)
to ensure isolation, speed, and determinism.

The test suite covers various scenarios, including successful loading,
handling of CUDA unavailability, file not found errors, download failures,
and issues during model state dictionary loading. Each test method adheres
to a Behavior-Driven Development (BDD) style for enhanced readability and
clarity of intent.

Relationships with other functions/modules:
- Directly tests `Synthetic_Image_Generator.load_model.load_model_from_drive`.
- Mocks `gdown` for file downloads.
- Mocks `torch` for device handling, model instantiation, and state_dict loading.
- Mocks `pathlib.Path` for file system interactions (e.g., `exists()`).
- Mocks `Synthetic_Image_Generator.model.CNF_UNet` for model creation and method calls.
- Mocks `Synthetic_Image_Generator.config` for default `IMAGE_SIZE`.
- Mocks `logging` to verify log messages.

Theory:
- **Unit Testing**: Verifying the smallest testable parts of an application in isolation.
- **Mocking**: Replacing objects with controlled substitutes that simulate real object behavior.
  This is crucial for isolating the unit under test from its dependencies, especially
  external ones like network requests (gdown) or hardware-dependent operations (CUDA).
- **Behavior-Driven Development (BDD)**: A software development process that encourages
  collaboration among developers, QA, and non-technical or business participants in a
  software project. It describes tests in terms of expected behaviors from the user's
  perspective, typically using "Given-When-Then" clauses.

References:
- Python's `unittest` module documentation: [https://docs.python.org/3/library/unittest.html](https://docs.python.org/3/library/unittest.html)
- Python's `unittest.mock` documentation: [https://docs.python.org/3/library/unittest.mock.html](https://docs.python.org/3/library/unittest.mock.html)
- Google Drive Downloader (`gdown`) documentation (for understanding its API).
"""

# Adjust sys.path to allow importing from the project root
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the function to be tested
try:
    from Synthetic_Image_Generator.load_model import load_model_from_drive
    # Also import the dummy classes if they are defined as fallbacks in load_model.py
    # These are used for `spec` in MagicMock to ensure mock objects have real methods.
    from Synthetic_Image_Generator.model import CNF_UNet as RealCNF_UNet
    from Synthetic_Image_Generator import config as RealConfig # Import config to use for spec if needed
except ImportError:
    # Fallback for direct execution of tests if package structure isn't fully set up.
    # This block ensures the tests can run even if the package is not installed via pip.
    print("Warning: Could not import from 'Synthetic_Image_Generator' package. Attempting direct import from sibling directory.")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from load_model import load_model_from_drive
    # If CNF_UNet and config are not found via package import, use dummy ones
    # or handle the ImportError for direct `spec` assignment if necessary.
    # For now, if the above fails, RealCNF_UNet will not be defined, and `spec` might fail.
    # A more robust solution would be to define dummy classes here if the import truly fails.
    # For this exercise, we assume the package import will eventually succeed for `spec`.
    print("Assuming dummy CNF_UNet and config if direct package import fails for `spec`.")
    class RealCNF_UNet: # Dummy class for spec if real import fails
        def __init__(self, image_size, *args, **kwargs): pass
        def to(self, device): return self
        def eval(self): pass
        def load_state_dict(self, state_dict): pass
    class RealConfig: # Dummy class for spec if real import fails
        IMAGE_SIZE = (64, 64)


class TestLoadModel(unittest.TestCase):
    """
    Test suite for the `load_model_from_drive` function in `load_model.py`.

    This class contains unit tests that verify the comprehensive functionality
    of the `load_model_from_drive` function. It ensures that the model weights
    are correctly downloaded, loaded, configured for the target device (CPU/CUDA),
    and set to evaluation mode. The tests isolate `load_model_from_drive` by
    mocking all its external dependencies, allowing for predictable and fast
    validation of its logic without actual network or hardware interaction.
    """

    # Comprehensive patching of all external dependencies used within load_model.py's load_model_from_drive
    @patch('Synthetic_Image_Generator.load_model.gdown')
    @patch('Synthetic_Image_Generator.load_model.torch')
    @patch('Synthetic_Image_Generator.load_model.Path') # Patch Path class for exists() and __init__
    @patch('Synthetic_Image_Generator.load_model.CNF_UNet') # Patch the CNF_UNet class itself
    @patch('Synthetic_Image_Generator.load_model.config') # Patch the config module
    @patch('Synthetic_Image_Generator.load_model.logging') # Patch logging to suppress output and check calls
    def setUp(self, mock_logging, mock_config, mock_CNF_UNet, mock_Path, mock_torch, mock_gdown):
        """
        Sets up the testing environment before each test method runs.

        This involves initializing and configuring all necessary mock objects
        that simulate the behavior of external dependencies for `load_model_from_drive`.
        Default behaviors for mocks are set here to reflect common successful scenarios,
        which can then be overridden by individual test methods for specific failure
        or edge-case testing.

        Args:
            mock_logging (MagicMock): Mock object for the `logging` module.
            mock_config (MagicMock): Mock object for the `config` module.
            mock_CNF_UNet (MagicMock): Mock object for the `CNF_UNet` class.
            mock_Path (MagicMock): Mock object for the `pathlib.Path` class.
            mock_torch (MagicMock): Mock object for the `torch` module.
            mock_gdown (MagicMock): Mock object for the `gdown` module.

        Returns:
            None
        """
        self.mock_gdown = mock_gdown
        self.mock_torch = mock_torch
        self.mock_Path = mock_Path
        self.mock_CNF_UNet = mock_CNF_UNet
        self.mock_config = mock_config
        self.mock_logging = mock_logging

        # Configure mock_config.IMAGE_SIZE to simulate a default configuration
        self.mock_config.IMAGE_SIZE = (128, 128)

        # Mock Path.exists() for the output_path check
        self.mock_path_instance = MagicMock(spec=Path) # Ensure mock has Path methods/attributes
        self.mock_path_instance.exists.return_value = True # Default to file exists after download
        self.mock_Path.return_value = self.mock_path_instance # Path() returns this instance
        self.mock_path_instance.__str__.return_value = "/mock/path/to/weights.pth" # For gdown.download

        # Mock CNF_UNet instance methods
        # Using RealCNF_UNet as spec ensures that the mock has `to`, `eval`, `load_state_dict`
        self.mock_model_instance = MagicMock(spec=RealCNF_UNet)
        self.mock_model_instance.to.return_value = self.mock_model_instance # Allow method chaining for .to()
        self.mock_model_instance.eval.return_value = None # .eval() usually returns None
        self.mock_model_instance.load_state_dict.return_value = None
        self.mock_CNF_UNet.return_value = self.mock_model_instance # CNF_UNet() returns this mock instance

        # Mock torch functions
        self.mock_torch.device.return_value = MagicMock(spec=torch.device, type='cpu') # Default device to CPU
        self.mock_torch.cuda.is_available.return_value = False # Default CUDA availability to False
        self.mock_torch.load.return_value = {'dummy_key': 'dummy_value'} # Mock the loaded state_dict

        # Suppress logging output during tests and get access to the logger's mock methods
        self.mock_logging.getLogger.return_value = MagicMock()
        self.mock_logger = self.mock_logging.getLogger.return_value


    def test_successful_model_loading(self):
        """
        Verifies the successful end-to-end process of downloading and loading a model.

        Given:
            - `gdown.download` successfully downloads a file.
            - The downloaded file exists at the specified `output_path`.
            - CUDA is available on the system.
            - `torch.load` successfully loads a state dictionary.
            - `CNF_UNet` instantiation and method calls (`to`, `load_state_dict`, `eval`) are successful.

        When:
            - `load_model_from_drive` is called with a Google Drive URL, output path, image size, and `device="cuda"`.

        Then:
            - `gdown.download` should be called once with the correct URL and path.
            - `Path(output_path).exists()` should be checked once.
            - `torch.device` should be initialized for CUDA.
            - `torch.cuda.is_available()` should be checked once.
            - `CNF_UNet` should be instantiated with the specified `image_size`.
            - The model instance's `to()` method should be called to move it to the CUDA device.
            - `torch.load` should be called once with the output path and map_location set to the device.
            - The model instance's `load_state_dict()` method should be called with the loaded state dictionary.
            - The model instance's `eval()` method should be called to set it to evaluation mode.
            - The function should return the correctly configured model instance.
            - Appropriate info log messages should be generated.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path` (str): Mock path for the downloaded weights file.
            - `image_size` (Tuple[int, int]): Mock image dimensions.
            - `device` (str): "cuda".

        Outputs:
            - `torch.nn.Module`: The mocked `CNF_UNet` model instance.

        Potential Exceptions:
            - `AssertionError`: If any mock call or return value does not match expectations.

        Relationships:
            - This test verifies the correct sequence and arguments of calls to `gdown`, `torch`, `Path`, and `CNF_UNet` within `load_model_from_drive`.

        Theory:
            - Demonstrates a full "happy path" scenario for a unit test, ensuring all
              integrated components are called as expected under ideal conditions.
        """
        # Given
        drive_url = "http://mock.drive.com/weights.pth"
        output_path = "test_model.pth"
        image_size = (64, 64)
        device = "cuda" # Request CUDA

        self.mock_torch.cuda.is_available.return_value = True # CUDA is available
        self.mock_torch.device.return_value.type = 'cuda' # Ensure mock device type is cuda

        # When
        loaded_model = load_model_from_drive(drive_url, output_path, image_size, device)

        # Then
        self.mock_gdown.download.assert_called_once_with(drive_url, str(self.mock_path_instance), quiet=False)
        self.mock_Path.assert_called_once_with(output_path) # Check Path object creation
        self.mock_path_instance.exists.assert_called_once() # Check if file existence was verified

        self.mock_torch.device.assert_called_once_with(device) # Check device creation
        self.mock_torch.cuda.is_available.assert_called_once() # Check CUDA availability

        self.mock_CNF_UNet.assert_called_once_with(image_size=image_size) # Model instantiated with correct size
        self.mock_model_instance.to.assert_called_once_with(self.mock_torch.device.return_value) # Model moved to device
        self.mock_torch.load.assert_called_once_with(output_path, map_location=self.mock_torch.device.return_value) # State dict loaded
        self.mock_model_instance.load_state_dict.assert_called_once_with(self.mock_torch.load.return_value) # Weights applied
        self.mock_model_instance.eval.assert_called_once() # Model set to eval mode

        self.assertEqual(loaded_model, self.mock_model_instance) # Correct model instance returned
        self.mock_logger.info.assert_has_calls([
            call(f"Attempting to download model weights from: {drive_url}"),
            call(f"Model weights downloaded to: {self.mock_path_instance}"),
            call(f"Loading model weights from {self.mock_path_instance}..."),
            call(f"Loading model on {self.mock_torch.device.return_value}."),
            call("Model weights loaded successfully.")
        ])


    def test_fallback_to_cpu_when_cuda_unavailable(self):
        """
        Verifies that the function correctly falls back to CPU if CUDA is requested but not available.

        Given:
            - `torch.cuda.is_available()` returns `False`.
            - `load_model_from_drive` is called requesting a "cuda" device.

        When:
            - `load_model_from_drive` is called.

        Then:
            - `torch.cuda.is_available()` should be checked once.
            - A warning log message indicating CUDA unavailability and CPU fallback should be issued.
            - `torch.device` should be called first with "cuda", then with "cpu".
            - The model should be moved to the CPU device.
            - `torch.load` should map the loaded state dictionary to the CPU.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path` (str): Mock path for the downloaded weights file.
            - `image_size` (Tuple[int, int]): Mock image dimensions.
            - `device` (str): "cuda".

        Outputs:
            - `torch.nn.Module`: The mocked `CNF_UNet` model instance loaded on CPU.

        Potential Exceptions:
            - `AssertionError`: If any mock call or log message does not match expectations.

        Relationships:
            - Tests the conditional logic within `load_model_from_drive` related to device selection.

        Theory:
            - This test validates the robustness of the function against varying hardware
              configurations, ensuring graceful degradation from GPU to CPU when necessary.
        """
        # Given
        drive_url = "http://mock.drive.com/weights.pth"
        output_path = "test_model.pth"
        image_size = (64, 64)
        device = "cuda"

        self.mock_torch.cuda.is_available.return_value = False # CUDA is NOT available
        # Configure side_effect for torch.device to simulate initial request and then fallback
        self.mock_torch.device.side_effect = [
            MagicMock(spec=torch.device, type='cuda'), # First call for requested device
            MagicMock(spec=torch.device, type='cpu')   # Second call for fallback device
        ]

        # When
        loaded_model = load_model_from_drive(drive_url, output_path, image_size, device)

        # Then
        self.mock_torch.cuda.is_available.assert_called_once()
        self.mock_torch.device.assert_has_calls([call("cuda"), call("cpu")]) # Check both calls
        self.mock_logger.warning.assert_called_once_with(
            "CUDA is not available, falling back to CPU for model loading."
        )
        self.mock_model_instance.to.assert_called_once_with(self.mock_torch.device.side_effect[1]) # Model moved to CPU
        self.mock_torch.load.assert_called_once_with(output_path, map_location=self.mock_torch.device.side_effect[1])
        self.assertEqual(loaded_model, self.mock_model_instance)


    def test_file_not_found_after_download(self):
        """
        Verifies that a `FileNotFoundError` is raised if the downloaded file
        does not exist at the expected path.

        Given:
            - `gdown.download` completes successfully (mocked not to raise an error).
            - `Path(output_path).exists()` returns `False`.

        When:
            - `load_model_from_drive` is called.

        Then:
            - `gdown.download` should be called once.
            - `Path(output_path).exists()` should be checked once.
            - A `FileNotFoundError` should be raised, indicating the absence of the file.
            - No further calls to `torch.load` or `CNF_UNet` should occur.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path` (str): Mock path for the downloaded weights file.
            - `image_size` (Tuple[int, int]): Mock image dimensions.
            - `device` (str): "cpu".

        Outputs:
            - Raises `FileNotFoundError`.

        Potential Exceptions:
            - `FileNotFoundError`: Expected exception.

        Relationships:
            - Tests error handling related to file system checks after download.

        Theory:
            - This test ensures that the function correctly validates the presence
              of the downloaded file before attempting to load it, preventing
              subsequent errors.
        """
        # Given
        drive_url = "http://mock.drive.com/non_existent.pth"
        output_path = "non_existent.pth"
        image_size = (64, 64)
        device = "cpu"

        self.mock_path_instance.exists.return_value = False # File does NOT exist after download

        # When / Then
        with self.assertRaises(FileNotFoundError) as cm:
            load_model_from_drive(drive_url, output_path, image_size, device)

        self.assertIn("Downloaded file not found", str(cm.exception))
        self.mock_gdown.download.assert_called_once()
        self.mock_path_instance.exists.assert_called_once()
        # Ensure no further torch calls are made if file not found
        self.mock_torch.load.assert_not_called()
        self.mock_CNF_UNet.assert_not_called()
        self.mock_logger.error.assert_not_called() # No loading error yet


    def test_download_failure(self):
        """
        Verifies that an exception during the download process is re-raised.

        Given:
            - `gdown.download` raises an arbitrary `Exception`.

        When:
            - `load_model_from_drive` is called.

        Then:
            - `gdown.download` should be called once.
            - The `Exception` raised by `gdown.download` should be re-raised by `load_model_from_drive`.
            - An error log message should be issued.
            - `Path(output_path).exists()` should not be called.
            - No further model loading steps should occur.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path` (str): Mock path for the downloaded weights file.
            - `image_size` (Tuple[int, int]): Mock image dimensions.
            - `device` (str): "cpu".

        Outputs:
            - Raises the original `Exception` from `gdown.download`.

        Potential Exceptions:
            - `Exception`: The exception raised by `gdown.download`.

        Relationships:
            - Tests robust error handling for network/download failures.

        Theory:
            - Ensures that critical failures during the download phase are propagated
              correctly, allowing upstream callers to handle them.
        """
        # Given
        drive_url = "http://mock.drive.com/bad_link.pth"
        output_path = "bad_download.pth"
        image_size = (64, 64)
        device = "cpu"

        self.mock_gdown.download.side_effect = Exception("Mock download error")

        # When / Then
        with self.assertRaisesRegex(Exception, "Mock download error"):
            load_model_from_drive(drive_url, output_path, image_size, device)

        self.mock_gdown.download.assert_called_once()
        self.mock_path_instance.exists.assert_not_called() # Should not check existence if download failed
        self.mock_logger.error.assert_called_once_with(
            f"Failed to download file from Google Drive: Mock download error"
        )


    def test_model_loading_runtime_error(self):
        """
        Verifies that a `RuntimeError` is raised if `torch.load` or `model.load_state_dict` fails.

        Given:
            - The model weights file exists.
            - `torch.load` (or `model.load_state_dict`) raises a `RuntimeError`.

        When:
            - `load_model_from_drive` is called.

        Then:
            - `gdown.download` should be called (assuming file exists, might be skipped if file pre-exists).
            - `Path(output_path).exists()` should be checked.
            - `CNF_UNet` should be instantiated and moved to device.
            - `torch.load` should be called.
            - A `RuntimeError` should be raised, indicating an issue during model weight loading.
            - An error log message should be issued.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path` (str): Mock path for the downloaded weights file.
            - `image_size` (Tuple[int, int]): Mock image dimensions.
            - `device` (str): "cpu".

        Outputs:
            - Raises `RuntimeError`.

        Potential Exceptions:
            - `RuntimeError`: Expected exception if model loading fails.

        Relationships:
            - Tests error handling during the critical phase of deserializing and applying model weights.

        Theory:
            - Ensures that internal PyTorch-related errors during model loading are caught
              and re-raised with a meaningful message, providing clear feedback on the failure.
        """
        # Given
        drive_url = "http://mock.drive.com/corrupt.pth"
        output_path = "corrupt.pth"
        image_size = (64, 64)
        device = "cpu"

        self.mock_torch.load.side_effect = RuntimeError("Mock torch load error")

        # When / Then
        with self.assertRaisesRegex(RuntimeError, "Failed to load model weights into CNF_UNet: Mock torch load error"):
            load_model_from_drive(drive_url, output_path, image_size, device)

        self.mock_gdown.download.assert_called_once() # Download attempt might happen first if file doesn't exist
        self.mock_path_instance.exists.assert_called_once()
        self.mock_torch.load.assert_called_once()
        self.mock_CNF_UNet.assert_called_once()
        self.mock_model_instance.to.assert_called_once()
        self.mock_logger.error.assert_called_once_with(
            f"Error loading model weights: Mock torch load error"
        )


    def test_default_image_size_from_config(self):
        """
        Verifies that `config.IMAGE_SIZE` is used when `image_size` is not provided.

        Given:
            - `load_model_from_drive` is called without the `image_size` argument.
            - `self.mock_config.IMAGE_SIZE` is set to a default value in `setUp`.

        When:
            - `load_model_from_drive` is called.

        Then:
            - `CNF_UNet` should be instantiated using `self.mock_config.IMAGE_SIZE`.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path` (str): Mock path for the downloaded weights file.
            - `device` (str): "cpu".
            - `image_size`: Not provided (defaults to `None`).

        Outputs:
            - `torch.nn.Module`: The mocked `CNF_UNet` model instance.

        Potential Exceptions:
            - `AssertionError`: If `CNF_UNet` is not called with the correct image size.

        Relationships:
            - Tests the default parameter handling and integration with the `config` module.

        Theory:
            - Ensures flexibility and adherence to configured defaults, reducing boilerplate
              when `image_size` is consistent across the project.
        """
        # Given
        drive_url = "http://mock.drive.com/default_size.pth"
        output_path = "default_size.pth"
        device = "cpu"

        # When
        load_model_from_drive(drive_url, output_path, device=device) # Omit image_size

        # Then
        self.mock_CNF_UNet.assert_called_once_with(image_size=self.mock_config.IMAGE_SIZE)


    def test_output_path_as_string(self):
        """
        Verifies that `output_path` provided as a string is correctly handled.

        Given:
            - `output_path` is provided as a string instead of a `pathlib.Path` object.

        When:
            - `load_model_from_drive` is called.

        Then:
            - `Path()` should be called with the string `output_path`.
            - `gdown.download` should receive the string representation of the Path object.
            - `torch.load` should receive the string `output_path`.

        Inputs:
            - `drive_url` (str): Mock Google Drive URL.
            - `output_path_str` (str): A string path for the downloaded weights file.
            - `image_size` (Tuple[int, int]): Mock image dimensions.
            - `device` (str): "cpu".

        Outputs:
            - `torch.nn.Module`: The mocked `CNF_UNet` model instance.

        Potential Exceptions:
            - `AssertionError`: If path handling is incorrect.

        Relationships:
            - Tests argument type flexibility for `output_path`.

        Theory:
            - Ensures the function is user-friendly and robust to common input types for paths.
        """
        # Given
        drive_url = "http://mock.drive.com/string_path.pth"
        output_path_str = "string_path_test.pth"
        image_size = (64, 64)
        device = "cpu"

        # When
        load_model_from_drive(drive_url, output_path_str, image_size, device)

        # Then
        # gdown.download expects a string path, so Path(output_path_str) is converted to str internally
        self.mock_gdown.download.assert_called_once_with(drive_url, str(self.mock_path_instance), quiet=False)
        self.mock_Path.assert_called_once_with(output_path_str) # Path() is called with the original string
        self.mock_torch.load.assert_called_once_with(output_path_str, map_location=self.mock_torch.device.return_value)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)