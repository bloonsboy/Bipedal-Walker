import torch
import numpy as np


class RolloutBuffer:
    """Stores transitions and computes advantages/returns."""

    def __init__(
        self, n_steps, num_envs, obs_shape, action_shape, device, gamma, gae_lambda
    ):
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.observations = torch.zeros(
            (self.n_steps, self.num_envs) + self.obs_shape
        ).to(device)
        self.actions = torch.zeros(
            (self.n_steps, self.num_envs) + self.action_shape
        ).to(device)
        self.log_probs = torch.zeros((self.n_steps, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.n_steps, self.num_envs)).to(device)
        self.dones = torch.zeros((self.n_steps, self.num_envs)).to(device)
        self.values = torch.zeros((self.n_steps, self.num_envs)).to(device)

        self.advantages = torch.zeros((self.n_steps, self.num_envs)).to(device)
        self.returns = torch.zeros((self.n_steps, self.num_envs)).to(device)

        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        """Adds a transition to the buffer."""
        if self.step >= self.n_steps:
            raise ValueError("Buffer full")

        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value

        self.step += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        """Computes GAE and returns."""
        last_value = last_value.clone().to(self.device).squeeze()
        last_done = last_done.clone().to(self.device).squeeze()

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

    def get_batch(self, minibatch_size):
        """Yields random minibatches from the buffer."""
        total_batch_size = self.n_steps * self.num_envs
        indices = np.arange(total_batch_size)
        np.random.shuffle(indices)

        flat_obs = self.observations.reshape((total_batch_size,) + self.obs_shape)
        flat_actions = self.actions.reshape((total_batch_size,) + self.action_shape)
        flat_log_probs = self.log_probs.reshape(total_batch_size)
        flat_advantages = self.advantages.reshape(total_batch_size)
        flat_returns = self.returns.reshape(total_batch_size)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        for start in range(0, total_batch_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]

            yield (
                flat_obs[batch_indices],
                flat_actions[batch_indices],
                flat_log_probs[batch_indices],
                flat_advantages[batch_indices],
                flat_returns[batch_indices],
            )

    def reset(self):
        """Resets the buffer."""
        self.step = 0
