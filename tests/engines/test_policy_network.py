"""
Tests for PolicyNetwork and SpatialPolicyHead in engines/rl/policy_network.py
"""
import torch as th
import numpy as np
import pytest
from gymnasium import spaces
from engines.rl.policy_network import PolicyNetwork, SpatialPolicyHead, SpatialValueHead
from utils.action_masks import ACTION_SPACE_SIZE


@pytest.fixture
def observation_space():
    """
    Observation space matching the Chess960 environment (8x8x20)
    """
    return spaces.Box(low=0, high=1, shape=(8, 8, 20), dtype=np.float32)


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


def test_features_dim_matches_spatial_output(network):
    """
    features_dim should be channels * 8 * 8 (spatial features, no bottleneck)
    """
    assert network.features_dim == network.channels * 8 * 8


## Testing forward pass output shape

def test_forward_output_shape_single(network):
    """
    Forward pass with batch size 1 should return the flattened spatial features
    """
    obs = th.zeros(1, 8, 8, 20)
    output = network(obs)
    assert output.shape == (1, network.features_dim)


def test_forward_output_shape_batch(network):
    """
    Forward pass with batch size 8 should return tensor of shape (8, features_dim)
    """
    obs = th.zeros(8, 8, 8, 20)
    output = network(obs)
    assert output.shape == (8, network.features_dim)


## Testing output is a tensor

def test_forward_returns_tensor(network):
    """
    Forward pass should return a torch.Tensor
    """
    obs = th.zeros(1, 8, 8, 20)
    output = network(obs)
    assert isinstance(output, th.Tensor)


## Testing forward pass with realistic input

def test_forward_with_random_input(network):
    """
    Forward pass should work with random board-like input without errors
    """
    obs = th.randint(0, 2, (4, 8, 8, 20)).float()
    output = network(obs)
    assert output.shape == (4, network.features_dim)


## Testing the spatial policy head

def test_policy_head_output_shape(network):
    """
    SpatialPolicyHead should map trunk features to one logit per action (4672)
    """
    head = SpatialPolicyHead(channels=network.channels)
    obs = th.randint(0, 2, (4, 8, 8, 20)).float()
    logits = head(network(obs))
    assert logits.shape == (4, ACTION_SPACE_SIZE)


def test_policy_head_logit_ordering():
    """
    Logits must be ordered as action = square * 73 + plane: the logit for
    (square, plane) must equal the conv output at (plane, rank, file).
    """
    head = SpatialPolicyHead(channels=64)
    latent = th.randn(1, 64 * 8 * 8)
    logits = head(latent)

    x = latent.view(-1, 64, 8, 8)
    conv_out = head.conv(x)  # (1, planes, rank, file)
    for square, plane in [(0, 0), (9, 14), (63, 72), (28, 56)]:
        rank, file = square // 8, square % 8
        expected = conv_out[0, plane, rank, file]
        assert th.isclose(logits[0, square * 73 + plane], expected)


## Testing the spatial value head

def test_value_head_output_shape(network):
    """
    SpatialValueHead should map trunk features to a single value per board
    """
    head = SpatialValueHead(channels=network.channels)
    obs = th.randint(0, 2, (4, 8, 8, 20)).float()
    values = head(network(obs))
    assert values.shape == (4, 1)
