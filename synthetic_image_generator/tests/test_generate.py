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
    """Test if the generate function produces output of the correct shape and type."""
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
    """Test that generation works with different batch sizes for generation."""
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