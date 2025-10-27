"""
model.py

Defines the neural network architecture for the Actor and Critic.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Weight initialization (standard technique for PPO)."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # --- Critic Network (Value Network) ---
        # Estimates V(s), the value of a state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

        # --- Actor Network (Policy Network) ---
        # Predicts the mean of the actions
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

        # The standard deviation (std) is a learned parameter, independent of the state.
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, obs):
        """Returns the state value (V(s))."""
        return self.critic(obs)

    def get_action_and_value(self, obs, action=None):
        """
        Calculates an action, its log_prob, the distribution's entropy
        and the state value (V(s)).
        """
        action_mean = self.actor_mean(obs)

        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value.view(-1)

    def act(self, obs, deterministic=False):
        """
        Simplified action method for evaluation.
        If deterministic, takes the mean. Otherwise, samples.
        """
        action_mean = self.actor_mean(obs)

        if deterministic:
            return action_mean

        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        dist = Normal(action_mean, action_std)
        action = dist.sample()

        return action
