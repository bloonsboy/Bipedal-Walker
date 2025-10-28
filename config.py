import torch


class Config:
    def __init__(self):
        # --- Environment ---
        # Target the simple BipedalWalker environment
        self.ENV_NAME = "BipedalWalker-v3"
        # Activate our custom reward shaping for a natural gait
        self.USE_REWARD_SHAPING = True

        # --- Parallel Training ---
        # Number of parallel environments (agents) to run
        self.NUM_ENVS = 16
        # Seed for reproducibility
        self.SEED = 42

        # --- PPO Hyperparameters ---
        # Total number of steps (actions) for the entire training (5M is often enough for v3)
        self.TOTAL_TIMESTEPS = 5_000_000
        # (Rollout size) Number of steps to collect PER agent BEFORE an update
        self.N_STEPS = 1024
        # Size of mini-batches for the SGD update
        self.MINIBATCH_SIZE = 512
        # Number of update epochs to run on the collected data
        self.NUM_EPOCHS = 10

        # --- Algorithm Coefficients ---
        # Discount factor for future rewards
        self.GAMMA = 0.99
        # Lambda parameter for Generalized Advantage Estimation (GAE)
        self.GAE_LAMBDA = 0.95
        # (Epsilon) PPO clipping limit to stabilize the update
        self.CLIP_COEF = 0.2
        # Weight of the "value loss" (Critic) in the total loss
        self.VF_COEF = 0.5
        # Weight of the "entropy loss" (Curiosity/exploration bonus)
        self.ENT_COEF = 0.001
        # Maximum limit for "gradient clipping" (avoids explosive updates)
        self.MAX_GRAD_NORM = 0.5

        # --- Learning Rate ---
        # Stable learning rate
        self.LEARNING_RATE = 1e-4
        # Linearly decrease the LR from 1e-4 to 0 over the entire training
        self.ANNEAL_LR = True

        # --- Network Architecture ---
        # Size of the hidden layers for the Actor and Critic
        self.HIDDEN_DIM = 256

        # --- Saving & Logs ---
        # Filename for the final model
        self.SAVE_PATH = "models/ppo_bipedal_walker_v3_natural.pth"
        # Folder for TensorBoard logs
        self.LOG_DIR = "logs"
        # Save a checkpoint every X steps
        self.SAVE_FREQ = 100_000

        # --- Runtime ---
        # Use the GPU (cuda) if available, otherwise the CPU
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Calculated Values (Do not modify) ---
        # Total batch size collected before each update
        self.BATCH_SIZE = int(self.NUM_ENVS * self.N_STEPS)
        # Total number of mini-batches per epoch
        self.NUM_MINIBATCHES = int(self.BATCH_SIZE // self.MINIBATCH_SIZE)
        assert self.BATCH_SIZE % self.MINIBATCH_SIZE == 0
