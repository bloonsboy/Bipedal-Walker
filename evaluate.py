"""
model.py (Parallel Version)

Defines the Actor-Critic network architecture for PPO.
This version is compatible with the parallel training script.
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initializes weights and biases for a linear layer.
    This helps stabilize training.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        """
        Initializes the Actor (policy) and Critic (value) networks.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Size of the hidden layers.
        """
        super().__init__()

        # --- Critic Network ---
        # Estimates the value of a state (V(s))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),  # Output is a single value
        )

        # --- Actor Network ---
        # Outputs the parameters for the action distribution (Policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_dim, action_dim), std=0.01
            ),  # Output is the mean for each action
        )

        # We learn the log of the standard deviation (log_std)
        # This is a common trick to ensure std is always positive
        # We use nn.Parameter so that this tensor is part of the model's parameters to be optimized
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        """
        Runs the critic network to get the state value.
        Args:
            x (torch.Tensor): Observations (state)
        Returns:
            torch.Tensor: The estimated value of the state(s).
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Runs the actor and critic network.

        Args:
            x (torch.Tensor): Observations (state).
            action (torch.Tensor, optional): If provided, computes log_prob for this action.
            deterministic (bool, optional):
                If True, returns the mean action (used for eval).
                If False, samples from the distribution (used for training).

        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        # Get action distribution parameters
        action_mean = self.actor_mean(x)
        action_log_std = self.actor_log_std.expand_as(
            action_mean
        )  # Make sure shape matches batch size
        action_std = torch.exp(action_log_std)

        # Create the probability distribution
        probs = Normal(action_mean, action_std)

        if deterministic:
            # For evaluation, we don't sample, we take the best action (the mean)
            action = action_mean
        elif action is None:
            # For training, we sample from the distribution
            action = probs.sample()

        # Get log probability of the action and entropy of the distribution
        log_prob = probs.log_prob(action).sum(dim=1)
        entropy = probs.entropy().sum(dim=1)

        # Get state value from critic
        value = self.critic(x)

        return action, log_prob, entropy, value
