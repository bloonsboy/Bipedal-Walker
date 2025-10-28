import gymnasium as gym
import numpy as np


class NaturalGaitRewardWrapper(gym.Wrapper):
    """
    Applies reward shaping penalties to encourage a more natural gait.
    Penalties are applied for:
    - Excessive hull (torso) angle deviation from upright.
    - Knees bent too much (crouching).
    - High motor torque (jerky movements).
    """

    def __init__(self, env):
        super().__init__(env)
        # Weights for penalties
        self.hull_angle_penalty_weight = 0.1  # Penalize leaning
        self.knee_angle_penalty_weight = (
            0.05  # Penalize crouching (slightly lower weight)
        )
        self.action_penalty_weight = 0.001  # Penalize high torque

        # Thresholds
        self.min_knee_angle = 0.8  # approx 45 degrees, encourage straighter legs

    def step(self, action):
        # Execute action in the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping_penalty = 0.0

        # Observation indices:
        # obs[0]: hull angle
        # obs[4]: knee 1 angle
        # obs[8]: knee 2 angle

        hull_angle = obs[0]
        knee_angle_1 = obs[4]
        knee_angle_2 = obs[8]

        # 1. Hull angle penalty: Penalize absolute deviation from upright (0)
        shaping_penalty += abs(hull_angle) * self.hull_angle_penalty_weight

        # 2. Knee angle penalty: Penalize if knees are bent beyond the threshold
        if knee_angle_1 < self.min_knee_angle:
            shaping_penalty += (
                self.min_knee_angle - knee_angle_1
            ) * self.knee_angle_penalty_weight
        if knee_angle_2 < self.min_knee_angle:
            shaping_penalty += (
                self.min_knee_angle - knee_angle_2
            ) * self.knee_angle_penalty_weight

        # 3. Action penalty: Penalize large action magnitudes (high torque)
        shaping_penalty += np.sum(np.square(action)) * self.action_penalty_weight

        # Subtract the total penalty from the original reward
        reward -= shaping_penalty

        return obs, reward, terminated, truncated, info
