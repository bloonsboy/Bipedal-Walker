"""
storage.py

Defines the RolloutBuffer, an essential class for PPO that stores
trajectories (experiences) and calculates advantages (GAE).
"""

import torch
import numpy as np


class RolloutBuffer:
    def __init__(self, n_steps, obs_dim, action_dim, device, gamma, gae_lambda):
        self.n_steps = n_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Buffers to store data
        self.observations = torch.zeros(
            (n_steps,) + (obs_dim,), dtype=torch.float32
        ).to(device)
        self.actions = torch.zeros((n_steps,) + (action_dim,), dtype=torch.float32).to(
            device
        )
        self.log_probs = torch.zeros((n_steps,), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((n_steps,), dtype=torch.float32).to(device)
        self.dones = torch.zeros((n_steps,), dtype=torch.float32).to(device)
        self.values = torch.zeros((n_steps,), dtype=torch.float32).to(device)

        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        """Adds a transition to the buffer."""
        if self.step >= self.n_steps:
            print("Error: Buffer full")
            return

        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        """Calculates advantages (GAE) and "returns" (target value)."""
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        self.returns = torch.zeros_like(self.rewards).to(self.device)

        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            self.advantages[t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

        self.returns = self.advantages + self.values

    def get_batch(self, batch_size):
        """Generator for random mini-batches."""
        indices = np.arange(self.n_steps)
        np.random.shuffle(indices)

        for start in range(0, self.n_steps, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                self.observations[batch_indices],
                self.actions[batch_indices],
                self.log_probs[batch_indices],
                self.advantages[batch_indices],
                self.returns[batch_indices],
            )

    def reset(self):
        """Resets the buffer pointer."""
        self.step = 0
