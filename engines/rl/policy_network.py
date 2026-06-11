"""
Contains the NN architecture.
"""
from functools import partial

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from utils.action_masks import N_PLANES


class ResBlock(nn.Module):
    """
    A single residual block: two 3x3 convs with batch normalisation and a skip
    connection (AlphaZero-style Conv -> BN -> ReLU). BN keeps activations
    well-scaled through depth, stabilising training.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyNetwork(BaseFeaturesExtractor):
    """
    ResNet trunk for the chess board representation.
    Outputs the final channels x 8 x 8 feature map flattened, preserving spatial
    structure so the policy/value heads can reshape it back to the board grid
    instead of working through a dense bottleneck.
    """
    def __init__(self, observation_space, num_blocks=6, channels=64):
        super().__init__(observation_space, features_dim=channels * 8 * 8)
        self.channels = channels

        n_input_channels = observation_space.shape[2]

        # Initial conv to lift input channels to working width
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

        self.flatten = nn.Flatten()

    def forward(self, observations):
        """
        Process board observations through the trunk.
        Permute to (batch, channels, H, W) format expected by Conv2d.
        """
        x = self._transpose(observations)
        x = self.initial(x)
        x = self.res_blocks(x)
        return self.flatten(x)

    def _transpose(self, observations):
        """
        Transpose observations from (batch, H, W, channels) to (batch, channels, H, W).
        """
        return observations.permute(0, 3, 1, 2)


class SpatialPolicyHead(nn.Module):
    """
    AlphaZerp-style policy head: a 1x1 conv maps the trunks 8*8 feature map to 73 movement plans per square.
    """
    def __init__(self, channels=64, n_planes=N_PLANES):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, n_planes, kernel_size=1)

    def forward(self, latent_pi):
        x = latent_pi.view(-1, self.channels, 8, 8)
        x = self.conv(x)                  # (batch, planes, rank, file)
        x = x.permute(0, 2, 3, 1)         # (batch, rank, file, planes)
        # square = rank * 8 + file, so flattening gives square * n_planes + plane
        return x.reshape(x.shape[0], -1)


class SpatialValueHead(nn.Module):
    """
    Value head replacing SB3 default single linear.
    
    1x1 conv compresses each squares 64 channels to 8, then a hidden dense layer combines information across the whole board non-linearly.
    """
    def __init__(self, channels=64, compress_channels=8, hidden_dim=256):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, compress_channels, kernel_size=1)
        self.hidden = nn.Linear(compress_channels * 8 * 8, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, latent_vf):
        x = latent_vf.view(-1, self.channels, 8, 8)
        x = F.relu(self.conv(x))
        x = F.relu(self.hidden(x.reshape(x.shape[0], -1)))
        return self.out(x)


class ChessPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO policy with spatial policy and value heads replacing the
    default dense layers.
    """
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = SpatialPolicyHead(channels=self.features_extractor.channels)
        self.value_net = SpatialValueHead(channels=self.features_extractor.channels)
        if self.ortho_init:
            self.action_net.apply(partial(self.init_weights, gain=0.01))
            self.value_net.conv.apply(partial(self.init_weights, gain=2 ** 0.5))
            self.value_net.hidden.apply(partial(self.init_weights, gain=2 ** 0.5))
            self.value_net.out.apply(partial(self.init_weights, gain=1))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
