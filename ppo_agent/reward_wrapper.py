import gymnasium as gym
import numpy as np


class NaturalGaitRewardWrapper(gym.Wrapper):
    """
    Applies minimal reward shaping penalties primarily focused on hull angle.
    Version 2: Removed knee penalty which caused stiff legs.
    """

    def __init__(self, env):
        super().__init__(env)
        # VERY small penalty for leaning
        self.hull_angle_penalty_weight = 0.005
        # --- KNEE PENALTY REMOVED ---
        # self.knee_angle_penalty_weight = 0.005 # Removed
        # self.min_knee_angle = 0.8 # Removed

    def step(self, action):
        # Execute action in the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping_penalty = 0.0

        # Observation indices: obs[0]: hull angle
        hull_angle = obs[0]

        # 1. Hull angle penalty: Penalize absolute deviation from upright (0)
        # Kept very small to only slightly discourage extreme leaning
        shaping_penalty += abs(hull_angle) * self.hull_angle_penalty_weight

        # --- KNEE PENALTY REMOVED ---

        # Subtract the total penalty from the original reward
        reward -= shaping_penalty

        return obs, reward, terminated, truncated, info
