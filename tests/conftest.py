import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


@pytest.fixture
def simple_model():
    """Creates a simple 3-layer network for testing"""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )
    return model


@pytest.fixture
def dummy_dataloader():
    """Creates a small dummy dataloader"""
    X = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    return loader


@pytest.fixture
def device():
    """Returns CPU device for testing (to avoid GPU requirements)"""
    return 'cpu'


@pytest.fixture
def known_weights_model():
    """Creates a model with known weights for deterministic testing"""
    model = nn.Linear(3, 2, bias=False)
    # Set specific weights
    model.weight.data = torch.tensor([[1.0, 2.0, 3.0],
                                      [4.0, 5.0, 6.0]])
    return model
