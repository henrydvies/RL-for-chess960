"""
Contains the NN architecture.
"""
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PolicyNetwork(BaseFeaturesExtractor):
    """
    Represents a custom network architecture for PPO
    """
    def __init__(self, observation_space, features_dim=256):
        super(PolicyNetwork, self).__init__(observation_space, features_dim)
        
        # Board as layers
        n_input_channels = observation_space.shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            # Take sample and transpose
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(self._transpose(sample)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        
    def forward(self, observations):
        """
        Returns output of the linear layer, and transpose to be (12, 8, 8)
        """
        return self.linear(self.cnn(self._transpose(observations)))
    
    def _transpose(self, observations):
        """
        Transpose observations to match expected shape.
        """
        return observations.permute(0, 3, 1, 2)