"""
config_improved.py (Version Hardcore)

Configuration updated for the "BipedalWalkerHardcore-v3" environment.

Main changes:
1. ENV_NAME: Switched to "BipedalWalkerHardcore-v3".
2. TOTAL_TIMESTEPS (2M -> 10M): This environment is much harder
   and requires significantly more training to solve.
3. MODEL_SAVE_PATH: Updated to reflect the new environment.

The "improved" parameters (larger network, larger rollouts)
are kept as they are well-suited for this harder task.
"""

import torch


class Config:
    def __init__(self):
        # --- Changements pour le mode Hardcore ---
        self.ENV_NAME = "BipedalWalkerHardcore-v3"
        self.TOTAL_TIMESTEPS = 10_000_000  # 10 Million steps

        # --- Paramètres de l'ancienne config "improved" (conservés) ---
        self.SEED = 42
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N_STEPS = 4096
        self.HIDDEN_DIM = 256
        self.LEARNING_RATE = 3e-4
        self.BATCH_SIZE = 64
        self.N_EPOCHS = 10
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIP_EPSILON = 0.2
        self.ENT_COEF = 0.01
        self.VF_COEF = 0.5
        self.MAX_GRAD_NORM = 0.5

        # Logging and saving
        self.LOG_DIR = "logs"
        self.MODEL_SAVE_DIR = "models"
        self.VIDEO_SAVE_DIR = "videos"
        # On change le nom du modèle
        self.MODEL_SAVE_PATH = (
            f"{self.MODEL_SAVE_DIR}/ppo_bipedal_walker_hardcore_v1.pth"
        )
