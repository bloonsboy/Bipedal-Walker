import torch
import numpy as np


class RolloutBuffer:
    def __init__(
        self, n_steps, num_envs, obs_shape, action_shape, device, gamma, gae_lambda
    ):
        """
        Initializes the buffer to store data from parallel environments.

        Args:
            n_steps (int): Number of steps to collect from EACH environment.
            num_envs (int): The number of parallel environments.
            obs_shape (tuple): Shape of a single observation.
            action_shape (tuple): Shape of a single action.
            device (torch.device): CPU or CUDA.
            gamma (float): Discount factor.
            gae_lambda (float): Lambda for GAE.
        """
        self.n_steps = n_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Initialize buffers. Note the shape: (n_steps, num_envs, *data_shape)
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

        # Buffer for advantages and returns
        self.advantages = torch.zeros((self.n_steps, self.num_envs)).to(device)
        self.returns = torch.zeros((self.n_steps, self.num_envs)).to(device)

        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        """
        Adds a transition from ALL parallel environments at the current step.
        """
        if self.step >= self.n_steps:
            raise ValueError("Buffer is full. Call reset() before adding more data.")

        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value

        self.step += 1

    def compute_returns_and_advantages(self, last_value, last_done):
        """
        Computes the advantages (GAE) and returns (target for value function)
        for all stored transitions.

        Args:
            last_value (torch.Tensor): Value estimation of the last obs (shape: [num_envs, 1]).
            last_done (torch.Tensor): Done flags from the last step (shape: [num_envs, 1]).
        """
        last_value = last_value.clone().to(self.device).squeeze()  # Shape: [num_envs]
        last_done = last_done.clone().to(self.device).squeeze()  # Shape: [num_envs]

        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            # Handle the last step of the rollout
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            # GAE Calculation
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            self.advantages[t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

        # Calculate returns (target for value function)
        self.returns = self.advantages + self.values

    def get_batch(self, minibatch_size):
        """
        Returns a generator that yields random minibatches from the collected data.

        The data is first flattened from [n_steps, num_envs, ...] to
        [n_steps * num_envs, ...] and then shuffled.

        Args:
            minibatch_size (int): The size of each minibatch.
        """
        # Calculate total batch size
        total_batch_size = self.n_steps * self.num_envs

        # Ensure total batch size is divisible by minibatch size
        if total_batch_size % minibatch_size != 0:
            raise ValueError(
                f"Total batch size ({total_batch_size}) must be divisible by minibatch size ({minibatch_size})"
            )

        # Create random indices for shuffling
        indices = np.arange(total_batch_size)
        np.random.shuffle(indices)

        # Flatten the data before batching
        # Shape changes from [n_steps, num_envs, *data_shape] to [total_batch_size, *data_shape]
        flat_obs = self.observations.reshape((total_batch_size,) + self.obs_shape)
        flat_actions = self.actions.reshape((total_batch_size,) + self.action_shape)
        flat_log_probs = self.log_probs.reshape(total_batch_size)
        flat_advantages = self.advantages.reshape(total_batch_size)
        flat_returns = self.returns.reshape(total_batch_size)

        # Normalize advantages (very important for stability)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        # Yield minibatches
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
        """Resets the buffer step pointer."""
        self.step = 0
