# load_model.py

import torch
import gdown
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

"""
Module for loading pre-trained Conditional Normalizing Flow (CNF) model weights
from Google Drive.

This module provides a robust function, `load_model_from_drive`, to facilitate
the retrieval and initialization of a `CNF_UNet` model with pre-trained weights.
It handles downloading the weights file from a specified Google Drive URL,
loading it onto the correct device (CPU or CUDA), and setting the model to
evaluation mode.

It is designed to seamlessly integrate with a PyTorch project structure where
the `CNF_UNet` model and a `config` module (defining parameters like `IMAGE_SIZE`)
are available. For environments where these imports might fail (e.g., standalone
script execution without the full project structure), dummy classes are provided
as fallbacks to allow the script to run for demonstration purposes,
though they will not provide functional model behavior.
"""

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import CNF_UNet model and config from your project structure.
# This assumes 'Synthetic_Image_Generator' is a package in your Python path.
try:
    from Synthetic_Image_Generator.model import CNF_UNet
    from Synthetic_Image_Generator import config
except ImportError:
    logger.warning("Could not import CNF_UNet or config from Synthetic_Image_Generator. "
                   "Please ensure your project structure is correct or adjust the import paths. "
                   "Using dummy classes for demonstration.")
    # Define dummy classes/configs if imports fail, to allow the script to run for demonstration.
    # In a real scenario, you'd fix the import or ensure the necessary files are present.
    class DummyConfig:
        """
        A dummy configuration class used when the actual 'config' module
        cannot be imported. It provides a default `IMAGE_SIZE` for
        demonstration purposes.
        """
        IMAGE_SIZE: Tuple[int, int] = (64, 64) # Default image size, adjust if your model uses a different one
    config = DummyConfig()

    class CNF_UNet(torch.nn.Module):
        """
        A dummy CNF_UNet class used when the actual model definition
        cannot be imported. It serves as a placeholder to allow the
        `load_model_from_drive` function to instantiate an object
        and call `load_state_dict` without errors, but does not
        provide actual model functionality.
        """
        def __init__(self, image_size: Tuple[int, int] = (64, 64)):
            super().__init__()
            logger.info("Using a dummy CNF_UNet class. Please ensure the real model is imported.")
            # Simple dummy model for demonstration if the real one isn't found
            self.linear = torch.nn.Linear(image_size[0] * image_size[1], 1)
            # Mock state_dict for `load_state_dict` to interact with
            self.state_dict_mock: Dict[str, torch.Tensor] = {
                "linear.weight": torch.randn(1, image_size[0] * image_size[1]),
                "linear.bias": torch.randn(1)
            }

        def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
            """
            Mocks the loading of a state dictionary for the dummy CNF_UNet.
            In a real model, this method would populate the model's parameters
            with the provided weights. For this dummy, it only prints a message
            and optionally updates mocked values.

            Args:
                state_dict (Dict[str, torch.Tensor]): A dictionary containing
                                                      model weights.
            """
            logger.info("Dummy CNF_UNet: Attempting to load state_dict.")
            for key in state_dict:
                if key in self.state_dict_mock:
                    self.state_dict_mock[key] = state_dict[key]
                else:
                    logger.warning(f"Warning: Key '{key}' not found in dummy model's state_dict.")
            logger.info("Dummy CNF_UNet: State_dict loaded (mocked).")

        def parameters(self) -> list:
            """
            Returns an empty list of parameters for the dummy model.
            In a real model, this would return an iterator over model parameters.
            """
            return []

        def eval(self) -> None:
            """
            Mocks setting the model to evaluation mode for the dummy CNF_UNet.
            In a real model, this method would set the module to evaluation mode.
            """
            pass # Dummy eval mode

def load_model_from_drive(
    drive_url: str,
    output_path: Union[str, Path] = "model_weights.pth",
    image_size: Tuple[int, int] = config.IMAGE_SIZE,
    device: Union[str, torch.device] = "cpu"
) -> CNF_UNet:
    """
    Downloads model weights from a Google Drive URL and loads them into a CNF_UNet model.

    This function streamlines the process of acquiring and initializing a PyTorch model
    from a remote Google Drive location. It first uses `gdown` to download the specified
    file, then verifies its existence, and finally loads the PyTorch state dictionary
    into an instantiated `CNF_UNet` model. The model is then moved to the specified
    device (CPU or CUDA) and set to evaluation mode (`.eval()`) for inference.
    This is particularly useful for deploying pre-trained models without needing
    to include large weight files directly in a repository.

    Args:
        drive_url (str): The Google Drive shareable link for the model weights file.
                         This should be a direct link, typically obtained by sharing
                         the file and copying the link.
        output_path (Union[str, Path]): The local path where the downloaded weights file
                                        will be saved. Can be a string or a `pathlib.Path` object.
                                        Defaults to "model_weights.pth" in the current directory.
        image_size (Tuple[int, int]): A tuple `(height, width)` representing the input
                                      image dimensions that the `CNF_UNet` model expects.
                                      This is crucial for correctly initializing the model
                                      architecture. Defaults to `config.IMAGE_SIZE` from
                                      the project's configuration.
        device (Union[str, torch.device]): The computing device on which to load the model.
                                           Accepts "cpu", "cuda", or a `torch.device` object.
                                           If "cuda" is specified but not available, it
                                           will fall back to "cpu" and log a warning.
                                           Defaults to "cpu".

    Returns:
        CNF_UNet: The instantiated `CNF_UNet` model with the downloaded weights loaded.
                  The model will be on the specified `device` and in evaluation mode.

    Raises:
        FileNotFoundError: If the downloaded file does not exist at `output_path`
                           after the download attempt, indicating a potential download failure.
        RuntimeError: If there is an issue loading the state dictionary into the model,
                      which could be due to a corrupted file, an incompatible model
                      architecture, or mismatched keys.
        Exception: Catches any other unexpected errors that might occur during the
                   download process (e.g., network issues, invalid URL) or model
                   instantiation.

    Example:
        ```python
        # Assuming you have a Google Drive link to your model weights
        my_drive_link = "[https://drive.google.com/file/d/1BWrRqSEY2KSE-u3TI8c2JYucm69g6oo7/view?usp=sharing](https://drive.google.com/file/d/1BWrRqSEY2KSE-u3TI8c2JYucm69g6oo7/view?usp=sharing)"
        
        # Specify the output filename and desired device
        weights_file = "my_model_weights.pth"
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            # Load the model
            loaded_model = load_model_from_drive(
                drive_url=my_drive_link,
                output_path=weights_file,
                image_size=(64, 64), # Ensure this matches your model's expected input
                device=target_device
            )
            print(f"Model successfully loaded on {target_device}.")

            # You can now use the loaded_model for inference
            # Example: Generate a dummy noise tensor and pass it through the model
            # dummy_noise = torch.randn(1, 1, 64, 64).to(target_device)
            # with torch.no_grad():
            #     generated_image = loaded_model(dummy_noise, torch.zeros(1).to(target_device))
            # print(f"Generated image shape: {generated_image.shape}")

        except (FileNotFoundError, RuntimeError, Exception) as e:
            print(f"An error occurred during model loading: {e}")
        finally:
            # Optional: Clean up the downloaded file
            if os.path.exists(weights_file):
                # os.remove(weights_file)
                # print(f"Cleaned up downloaded weights file: {weights_file}")
                pass # Keeping the file for demonstration; uncomment os.remove to delete
        ```

    Relationships with other functions:
        - This function relies on `gdown.download` for downloading files from Google Drive.
        - It instantiates `CNF_UNet`, expecting it to be a `torch.nn.Module` subclass
          with an `__init__` method that accepts `image_size` and a `load_state_dict` method.
        - It uses `torch.load` to deserialize the weight file and `torch.device` for device management.

    Explanation of the theory:
        Generative models, such as Conditional Normalizing Flows (CNFs) implemented with a U-Net
        architecture, are often trained to learn a complex data distribution (e.g., images).
        The training process results in a set of optimized parameters (weights) that capture
        this distribution. These weights are saved into a 'state dictionary' (typically a `.pth` file
        in PyTorch) which is a Python dictionary mapping parameter names to their tensor values.
        To use the trained model for inference (e.g., generating new images), these weights must
        be loaded back into an identical model architecture. The process involves:
        1.  **Download:** Retrieving the binary weights file from a storage location (e.g., Google Drive).
        2.  **Instantiation:** Creating an empty instance of the model's neural network architecture.
        3.  **Loading State Dictionary:** Populating the instantiated model's parameters with the
            values from the downloaded state dictionary. `torch.load` handles deserialization,
            and `model.load_state_dict` maps the weights to the correct layers.
        4.  **Device Placement:** Moving the model and its parameters to the appropriate computing
            device (CPU for general use, CUDA for GPU acceleration) for efficient computation.
        5.  **Evaluation Mode:** Setting the model to `.eval()` mode, which disables specific
            layers like Dropout and BatchNorm that behave differently during training vs. inference.

    References for the theory:
        - PyTorch Documentation on Saving and Loading Models: https://pytorch.org/docs/stable/notes/serialization.html
        - Original CNF paper (referenced conceptually for CNF models): "Neural Ordinary Differential Equations" by Chen et al. (NeurIPS 2018)
        - U-Net Architecture: "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al. (MICCAI 2015)
    """
    logger.info(f"Attempting to download model weights from: {drive_url}")
    try:
        # gdown automatically handles Google Drive's confirmation page and file IDs
        gdown.download(drive_url, str(output_path), quiet=False)
        logger.info(f"Model weights downloaded to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to download file from Google Drive: {e}")
        raise

    if not Path(output_path).exists():
        raise FileNotFoundError(f"Downloaded file not found at {output_path}. Download might have failed.")

    logger.info(f"Loading model weights from {output_path}...")
    try:
        # Determine device
        if isinstance(device, str):
            actual_device = torch.device(device)
        else:
            actual_device = device

        if actual_device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA is not available, falling back to CPU for model loading.")
            actual_device = torch.device("cpu")
        
        logger.info(f"Loading model on {actual_device}.")

        # Instantiate your model
        # Ensure image_size matches what your CNF_UNet expects during its __init__
        model = CNF_UNet(image_size=image_size).to(actual_device)

        # Load the state dictionary
        state_dict = torch.load(output_path, map_location=actual_device)

        # Load weights into the model
        model.load_state_dict(state_dict)
        model.eval() # Set the model to evaluation mode
        logger.info("Model weights loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model weights: {e}")
        raise RuntimeError(f"Failed to load model weights into CNF_UNet: {e}")

if __name__ == "__main__":
    """
    Main execution block for demonstrating the `load_model_from_drive` function.
    This block is executed when the script is run directly. It defines a sample
    Google Drive link for model weights, specifies the output filename, and
    attempts to load the model. It also includes optional commented-out code
    for basic inference and file cleanup.
    """
    # Your Google Drive link to the model weights. Replace with your actual link.
    # This example link is purely illustrative and may not point to a valid CNF_UNet model.
    google_drive_link = "https://drive.google.com/file/d/1BWrRqSEY2KSE-u3TI8c2JYucm69g6oo7/view?usp=sharing"
    
    # Define where to save the weights file locally
    weights_filename = "downloaded_cnf_unet_weights.pth"

    # Determine the image size expected by the model.
    # It attempts to use config.IMAGE_SIZE if available, otherwise defaults to (64, 64).
    model_image_size: Tuple[int, int] = getattr(config, 'IMAGE_SIZE', (64, 64))

    # Determine the target device for loading the model. Prioritizes CUDA if available.
    target_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        logger.info(f"Starting model loading demonstration on {target_device}...")
        loaded_model = load_model_from_drive(
            drive_url=google_drive_link,
            output_path=weights_filename,
            image_size=model_image_size,
            device=target_device
        )
        logger.info(f"Model loaded successfully on {target_device}!")
        
        # Example of how to use the loaded model for inference (uncomment to run):
        # logger.info("Generating a dummy image using the loaded model (if not dummy model)...")
        # dummy_noise = torch.randn(1, 1, *model_image_size).to(target_device) # Batch size 1, 1 channel
        # dummy_time = torch.zeros(1).to(target_device) # Time t=0 for the flow
        # with torch.no_grad():
        #     generated_image = loaded_model(dummy_noise, dummy_time)
        # logger.info(f"Generated image shape: {generated_image.shape}")

    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.error(f"An error occurred during model loading demonstration: {e}")

    finally:
        # Optional: Clean up the downloaded weights file if you don't need it after execution.
        # Uncomment the os.remove line below to delete the file.
        if Path(weights_filename).exists():
            # os.remove(weights_filename)
            # logger.info(f"Cleaned up downloaded weights file: {weights_filename}")
            pass # File is kept by default for user inspection; uncomment os.remove to delete.