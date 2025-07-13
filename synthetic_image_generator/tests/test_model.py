# tests/test_model.py
import torch
import pytest

# Adjust import path
from model import (
    TimeEmbedding, ResidualBlock, SelfAttention2d, UNetBlock, CNF_UNet
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_EMBED_DIM = 256
BATCH_SIZE = 4

@pytest.mark.parametrize("embed_dim", [128, 256])
def test_time_embedding(embed_dim):
    """
    GIVEN: a TimeEmbedding module with a specified embedding dimension
    WHEN: a batch of random time values is passed through the module
    THEN: the output tensor should have the expected batch size and embedding dimension
    """
    t = torch.randn(BATCH_SIZE).to(DEVICE)
    model = TimeEmbedding(embed_dim).to(DEVICE)
    emb = model(t)
    assert emb.shape == (BATCH_SIZE, embed_dim)

@pytest.mark.parametrize("in_channels, out_channels", [(64, 128), (128, 128)])
def test_residual_block(in_channels, out_channels):
    """
    GIVEN: a ResidualBlock with specified input and output channels, and a time embedding
    WHEN: a batch of random input data and time embeddings are passed through the block
    THEN: the output tensor should have the expected batch size, output channels, and spatial dimensions
    """
    x = torch.randn(BATCH_SIZE, in_channels, 32, 32).to(DEVICE)
    t_emb = torch.randn(BATCH_SIZE, TIME_EMBED_DIM).to(DEVICE)
    model = ResidualBlock(in_channels, out_channels, TIME_EMBED_DIM).to(DEVICE)
    output = model(x, t_emb)
    assert output.shape == (BATCH_SIZE, out_channels, 32, 32)

def test_self_attention_2d():
    """
    GIVEN: a SelfAttention2d module with a specified number of channels
    WHEN: a batch of random input data is passed through the module
    THEN: the output tensor should have the same shape as the input tensor
    """
    x = torch.randn(BATCH_SIZE, 64, 16, 16).to(DEVICE)
    model = SelfAttention2d(64).to(DEVICE)
    output = model(x)
    assert output.shape == x.shape

def test_unet_block():
    """
    GIVEN: a UNetBlock with specified input/output channels and a time embedding dimension
    WHEN: a batch of random input data and time embeddings are passed through the block
    THEN: the output tensor should have the expected batch size, output channels, and spatial dimensions
    """
    x = torch.randn(BATCH_SIZE, 64, 32, 32).to(DEVICE)
    t_emb = torch.randn(BATCH_SIZE, TIME_EMBED_DIM).to(DEVICE)
    model = UNetBlock(64, 128, TIME_EMBED_DIM).to(DEVICE)
    output = model(x, t_emb)
    assert output.shape == (BATCH_SIZE, 128, 32, 32)

def test_cnf_unet_forward_pass():
    """
    GIVEN: a CNF_UNet model initialized with a time embedding dimension
    WHEN: a batch of random input images and time values are passed through the model
    THEN: the output tensor should have the same shape as the input image tensor
    """
    model = CNF_UNet(time_embed_dim=TIME_EMBED_DIM).to(DEVICE)
    x = torch.randn(BATCH_SIZE, 1, 96, 96).to(DEVICE)
    t = torch.rand(BATCH_SIZE).to(DEVICE) # t is a float from 0 to 1
    
    output = model(x, t)
    
    assert output.shape == (BATCH_SIZE, 1, 96, 96)