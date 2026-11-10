"""The observation encoder of navigation-mappo-rl (networks/actor_critic_network.py).

The flat observation is history_length frames of [state | lidar]. It is split into two parts:
- the state values of all frames go through an MLP;
- the lidar becomes history_length * 3 channels over n_rays for a 1-D CNN.

Both feature sets are joined and mixed by two dense layers. LidarFeatures wraps the encoder
as a Stable-Baselines3 features extractor for single-agent PPO. The multi-agent policy uses
two encoders, one for the actor and one for the critic (multi_agent/Training/policy.py).
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Shared.observation import N_CHANNELS, STATE_DIM


class ObservationEncoder(nn.Module):

    def __init__(self, n_rays, history_length, features_dim=256):
        super().__init__()
        self.n_rays = n_rays
        self.history_length = history_length

        self.cnn = nn.Sequential(
            nn.Conv1d(history_length * N_CHANNELS, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(STATE_DIM * history_length, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        with torch.no_grad():
            n_cnn = self.cnn(torch.zeros(1, history_length * N_CHANNELS, n_rays)).shape[1]
        self.head = nn.Sequential(
            nn.Linear(n_cnn + 128, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        frames = observations.reshape(observations.shape[0], self.history_length, -1)
        states = frames[:, :, :STATE_DIM].flatten(start_dim=1)
        lidar = frames[:, :, STATE_DIM:].reshape(observations.shape[0], -1, self.n_rays)
        return self.head(torch.cat([self.cnn(lidar), self.state_mlp(states)], dim=1))


class LidarFeatures(BaseFeaturesExtractor):

    def __init__(self, observation_space, n_rays, history_length, features_dim=256):
        super().__init__(observation_space, features_dim)
        self.encoder = ObservationEncoder(n_rays, history_length, features_dim)

    def forward(self, observations):
        return self.encoder(observations)
