# tests/test_generate.py
import torch

# Adjust import path
from generate import generate

class MockGenerator(torch.nn.Module):
    """A simple mock model that returns a tensor of the correct shape."""
    def forward(self, x, t):
        # The model's output (velocity) should have the same shape as the input noise
        assert x.dim() == 4
        assert t.dim() == 1
        return torch.randn_like(x)

def test_generate_output_shape_and_type():
    """
    GIVEN: a mock generator model, desired number of samples, and generation steps,
           along with mocked 'config.image_size' and 'config.device'
    WHEN: the `generate` function is called to produce images
    THEN: the output should be a PyTorch tensor with the correct shape (num_samples, 1, image_size[0], image_size[1]),
          and it should be on the 'cpu' device
    """
    mock_model = MockGenerator()
    num_samples = 8
    steps = 10
    image_size = (96, 96)  # Define for the test context

    # Mock the config dependencies for the generate function
    import config
    config.image_size = image_size
    config.device = "cpu"

    generated_images = generate(mock_model, num_samples, steps=steps)

    assert isinstance(generated_images, torch.Tensor)
    assert generated_images.shape == (num_samples, 1, *image_size)
    assert str(generated_images.device) == 'cpu' # Ensure it's moved back to CPU

def test_generate_with_different_batch_sizes(mocker):
    """
    GIVEN: a mock generator model, a non-standard number of samples (e.g., not divisible by default batch size),
           generation steps, and mocked 'config' values, with tqdm progress bar mocked
    WHEN: the `generate` function is called
    THEN: the function should successfully generate the specified number of samples,
          and the first dimension of the output tensor should match 'num_samples'
    """
    mock_model = MockGenerator()
    num_samples = 7 # A number not divisible by the default batch_size_gen
    steps = 5

    import config
    config.image_size = (96, 96)
    config.device = "cpu"
    
    # Mock tqdm to speed up test execution
    mocker.patch('generate.tqdm', side_effect=lambda x, **kwargs: x)

    generated_images = generate(mock_model, num_samples, steps=steps)
    assert generated_images.shape[0] == num_samples