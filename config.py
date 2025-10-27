"""
config.py

This file centralizes all hyperparameters for training and evaluation.
Modifying these values is the primary way to "fine-tune" the agent.
"""

import torch


class Config:
    def __init__(self):
        self.ENV_NAME = "BipedalWalker-v3"
        self.SEED = 42  # For reproducibility
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training parameters
        self.TOTAL_TIMESTEPS = 1_000_000  # Total environment steps
        self.LEARNING_RATE = 3e-4  # Learning rate
        self.N_STEPS = 2048  # Steps per rollout (data collection)
        self.BATCH_SIZE = 64  # Mini-batch size for update
        self.N_EPOCHS = 10  # Number of update epochs per rollout

        # PPO algorithm parameters
        self.GAMMA = 0.99  # Discount factor
        self.GAE_LAMBDA = 0.95  # Lambda for GAE
        self.CLIP_EPSILON = 0.2  # Epsilon for PPO clipping
        self.ENT_COEF = 0.02   # Entropy loss coefficient (exploration)
        self.VF_COEF = 0.5  # Value function loss coefficient
        self.MAX_GRAD_NORM = 0.5  # Gradient clipping

        # Neural network parameters
        self.HIDDEN_DIM = 64  # Hidden layer size

        # Logging and saving parameters
        self.LOG_DIR = "logs"
        self.MODEL_SAVE_DIR = "models"
        self.VIDEO_SAVE_DIR = "videos"
        self.MODEL_SAVE_PATH = f"{self.MODEL_SAVE_DIR}/ppo_bipedal_walker.pth"
