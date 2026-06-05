"""
Tests for PolicyNetwork in engines/rl/policy_network.py
"""
import torch as th
import numpy as np
import pytest
from gymnasium import spaces
from engines.rl.policy_network import PolicyNetwork


@pytest.fixture
def observation_space():
    """
    Observation space matching the Chess960 environment (8x8x12 float32)
    """
    return spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)


@pytest.fixture
def network(observation_space):
    """
    Shared PolicyNetwork instance
    """
    return PolicyNetwork(observation_space)


## Testing instantiation

def test_network_instantiates(observation_space):
    """
    PolicyNetwork should instantiate without errors
    """
    network = PolicyNetwork(observation_space)
    assert network is not None


def test_network_instantiates_custom_features_dim(observation_space):
    """
    PolicyNetwork should accept a custom features_dim
    """
    network = PolicyNetwork(observation_space, features_dim=128)
    assert network is not None


## Testing forward pass output shape

def test_forward_output_shape_single(network):
    """
    Forward pass with batch size 1 should return tensor of shape (1, 256)
    """
    obs = th.zeros(1, 8, 8, 12)
    output = network(obs)
    assert output.shape == (1, 256)


def test_forward_output_shape_batch(network):
    """
    Forward pass with batch size 8 should return tensor of shape (8, 256)
    """
    obs = th.zeros(8, 8, 8, 12)
    output = network(obs)
    assert output.shape == (8, 256)


## Testing output is a tensor

def test_forward_returns_tensor(network):
    """
    Forward pass should return a torch.Tensor
    """
    obs = th.zeros(1, 8, 8, 12)
    output = network(obs)
    assert isinstance(output, th.Tensor)


## Testing forward pass with realistic input

def test_forward_with_random_input(network):
    """
    Forward pass should work with random board-like input without errors
    """
    obs = th.randint(0, 2, (4, 8, 8, 12)).float()
    output = network(obs)
    assert output.shape == (4, 256)