"""
Contains the NN architecture.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResBlock(nn.Module):
    """
    A single residual block: two 3x3 convs with a skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


class PolicyNetwork(BaseFeaturesExtractor):
    """
    ResNet-style feature extractor for chess board representation.
    """
    def __init__(self, observation_space, features_dim=256, num_blocks=6, channels=64):
        super(PolicyNetwork, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]

        # Initial conv to lift input channels to working width
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Stack of residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)]
        )

        # Final conv to compress before flattening
        self.final_conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flatten size with a dummy forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.final_conv(
                self.res_blocks(self.initial(self._transpose(sample)))
            ).shape[1]

        # Final linear projection to features_dim
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        """
        Process board observations through the network.
        Permute to (batch, channels, H, W) format expected by Conv2d.
        """
        x = self._transpose(observations)
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return self.linear(x)

    def _transpose(self, observations):
        """
        Transpose observations from (batch, H, W, channels) to (batch, channels, H, W).
        """
        return observations.permute(0, 3, 1, 2)