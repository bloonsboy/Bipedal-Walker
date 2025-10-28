import torch


class Config:
    def __init__(self):
        # --- Environment ---
        # Standard BipedalWalker environment
        self.ENV_NAME = "BipedalWalker-v3"
        # Activate our ADJUSTED custom reward shaping
        self.USE_REWARD_SHAPING = True

        # --- Parallel Training ---
        self.NUM_ENVS = 16
        self.SEED = 42

        # --- PPO Hyperparameters ---
        self.TOTAL_TIMESTEPS = 5_000_000
        self.N_STEPS = 1024
        self.MINIBATCH_SIZE = 512
        self.NUM_EPOCHS = 10

        # --- Algorithm Coefficients ---
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIP_COEF = 0.2
        self.VF_COEF = 0.5
        # Keep a little exploration
        self.ENT_COEF = 0.001
        self.MAX_GRAD_NORM = 0.5

        # --- Learning Rate ---
        self.LEARNING_RATE = 1e-4
        self.ANNEAL_LR = True

        # --- Network Architecture ---
        self.HIDDEN_DIM = 256

        # --- Saving & Logs ---
        # New model name for this version
        self.SAVE_PATH = "models/ppo_bipedal_walker_v3_natural_v2.pth"
        self.LOG_DIR = "logs"
        self.SAVE_FREQ = 100_000

        # --- Runtime ---
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Calculated Values (Do not modify) ---
        self.BATCH_SIZE = int(self.NUM_ENVS * self.N_STEPS)
        self.NUM_MINIBATCHES = int(self.BATCH_SIZE // self.MINIBATCH_SIZE)
        assert self.BATCH_SIZE % self.MINIBATCH_SIZE == 0
