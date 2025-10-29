import gymnasium as gym
import numpy as np

# Observation indices based on BipedalWalker-v3 documentation:
# obs[8]: leg_0_contact (bool converted to 0.0 or 1.0)
# obs[13]: leg_1_contact (bool converted to 0.0 or 1.0)
# obs[14-23]: 10 lidar readings range [0, 1] (1 = far, 0 = touching)
LIDAR_FRONT_INDEX = 14  # Index of the lidar sensor pointing straight ahead
LEG0_CONTACT_INDEX = 8
LEG1_CONTACT_INDEX = 13


class HardcoreClimbingRewardWrapper(gym.Wrapper):
    """
    Applies reward shaping specifically to encourage climbing obstacles in Hardcore.
    It adds a bonus for lifting feet when an obstacle is detected nearby.
    """

    def __init__(self, env):
        super().__init__(env)
        # Bonus multiplier for lifting a foot when obstacle is near
        self.lift_foot_bonus = 0.1
        # How close an obstacle must be detected by lidar (0=touching, 1=far)
        self.obstacle_lidar_threshold = 0.2

    def step(self, action):
        # Execute action in the base environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        shaping_bonus = 0.0

        # Read relevant observations
        lidar_front = obs[LIDAR_FRONT_INDEX]
        leg0_contact = obs[LEG0_CONTACT_INDEX]
        leg1_contact = obs[LEG1_CONTACT_INDEX]

        # Check if an obstacle is detected directly in front
        if lidar_front < self.obstacle_lidar_threshold:
            # If obstacle is close, grant bonus if EITHER foot is off the ground
            if leg0_contact == 0.0 or leg1_contact == 0.0:
                shaping_bonus += self.lift_foot_bonus

        # Add the shaping bonus to the original reward
        reward += shaping_bonus

        # Log the shaping bonus for analysis (optional)
        # if "shaping_bonus" not in info:
        #     info["shaping_bonus"] = 0.0
        # info["shaping_bonus"] += shaping_bonus # Accumulate if needed

        return obs, reward, terminated, truncated, info
