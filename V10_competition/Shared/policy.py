"""The network of both trainers in Stable-Baselines3: a decentralised actor and a centralised
critic, as navigation-mappo-rl's MAPPO. In the multi-agent env all 10 aircraft share one actor
and one critic (parameter sharing). In the single-agent env the critic attends over the agent
and its 10 intruders.

    actor   acts on the aircraft's own observation only: navigation-mappo-rl's encoder
            (Shared/encoder.py), then a tanh MLP (net_arch['pi'], 128-128 by default).
            This is all that runs at execution.
    critic  values the aircraft from its own observation (a second encoder) plus every
            aircraft's true state (Observer.observe_joint). The aircraft's own token attends
            over all tokens, as the repo's attention critic attends over all agents. Then a tanh MLP
            (net_arch['vf']). It is used in training only.

SB3 builds one features extractor and one MLP extractor. CTDEFeatures therefore returns
[actor features | critic features] side by side, and SplitMlpExtractor gives each head its
own part, so the policy loss only trains the actor's half and the value loss only the
critic's half.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from Shared.encoder import ObservationEncoder
from Shared.observation import JOINT_DIM

FEATURES_DIM = 256
TOKEN_DIM = 64


class CTDEFeatures(BaseFeaturesExtractor):

    def __init__(self, observation_space, n_rays, history_length):
        super().__init__(observation_space, 2 * FEATURES_DIM + TOKEN_DIM)
        self.actor_dim = FEATURES_DIM
        self.actor_encoder = ObservationEncoder(n_rays, history_length, FEATURES_DIM)
        self.critic_encoder = ObservationEncoder(n_rays, history_length, FEATURES_DIM)
        self.token = nn.Sequential(nn.Linear(JOINT_DIM, TOKEN_DIM), nn.ReLU())
        self.attention = nn.MultiheadAttention(TOKEN_DIM, num_heads=4, batch_first=True)

    def forward(self, observations):
        own, joint = observations['own'], observations['joint']
        tokens = self.token(joint)
        alive, is_self = joint[..., 0] > 0.5, joint[..., 1] > 0.5
        query = (tokens * is_self.unsqueeze(-1)).sum(dim=1, keepdim=True)
        padding = ~alive
        padding[padding.all(dim=1)] = False
        context, _ = self.attention(query, tokens, tokens, key_padding_mask=padding)
        critic = torch.cat([self.critic_encoder(own), context.squeeze(1)], dim=1)
        return torch.cat([self.actor_encoder(own), critic], dim=1)


class SplitMlpExtractor(nn.Module):

    def __init__(self, actor_dim, critic_dim, net_arch):
        super().__init__()
        self.actor_dim = actor_dim
        self.policy_net = _tanh_mlp(actor_dim, net_arch['pi'])
        self.value_net = _tanh_mlp(critic_dim, net_arch['vf'])
        self.latent_dim_pi, self.latent_dim_vf = net_arch['pi'][-1], net_arch['vf'][-1]

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features[:, :self.actor_dim])

    def forward_critic(self, features):
        return self.value_net(features[:, self.actor_dim:])


class CTDEPolicy(ActorCriticPolicy):

    def _build_mlp_extractor(self):
        actor_dim = self.features_extractor.actor_dim
        self.mlp_extractor = SplitMlpExtractor(actor_dim, self.features_dim - actor_dim, self.net_arch)


def _tanh_mlp(n_in, sizes):
    layers = []
    for n_out in sizes:
        layers += [nn.Linear(n_in, n_out), nn.Tanh()]
        n_in = n_out
    return nn.Sequential(*layers)
