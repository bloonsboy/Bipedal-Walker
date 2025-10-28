import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        # Actor network predicts the mean of the action distribution
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(
                nn.Linear(hidden_dim, action_dim), std=0.01
            ),  # Small std for output
        )

        # Log standard deviation learned as a parameter (state-independent)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Critic network predicts the state value V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),  # Larger std for output
        )

    def get_value(self, x):
        """Returns the state-value V(s)."""
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Returns an action, its log probability, entropy, and the state-value.
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Create Normal distribution
        probs = Normal(action_mean, action_std)

        if deterministic:
            # Use the mean action during evaluation
            action = action_mean
        elif action is None:
            # Sample action during training
            action = probs.sample()

        # Calculate log probability and entropy
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)

        # Get value from critic
        value = self.critic(x).squeeze(-1)

        return action, log_prob, entropy, value
