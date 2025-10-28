import torch

class Config:
    def __init__(self):
        """
        Configuration class for PPO training.
        V3: Tuned for breaking the "complex obstacle" plateau (-70 score).
        
        Changes:
        - N_STEPS: 512 -> 1024 (More stable advantage estimation for complex tasks)
        - ENT_COEF: 0.001 -> 0.005 (More exploration to find new jump techniques)
        - MINIBATCH_SIZE: 256 -> 512 (Keep ratios consistent)
        - TOTAL_TIMESTEPS: 10M -> 15M (More time to master difficult obstacles)
        - SAVE_PATH: v2 -> v3
        """
        
        # --- Environment Hyperparameters ---
        self.ENV_NAME = "BipedalWalkerHardcore-v3"
        self.NUM_ENVS = 8  # Number of parallel environments
        self.SEED = 42

        # --- PPO Hyperparameters ---
        # --- TUNING CHANGES ---
        self.TOTAL_TIMESTEPS = 15_000_000  # More training time
        self.N_STEPS = 1024                # Longer rollouts for stable gradients
        self.MINIBATCH_SIZE = 512          # Larger minibatches
        self.ENT_COEF = 0.005              # More exploration (curiosity)
        # --- END TUNING ---
        
        self.NUM_EPOCHS = 10               # Number of epochs to train on the collected data
        self.GAMMA = 0.99                  # Discount factor
        self.GAE_LAMBDA = 0.95             # Lambda for GAE
        self.CLIP_COEF = 0.2               # PPO clipping coefficient
        self.VF_COEF = 0.5                 # Value function coefficient
        self.MAX_GRAD_NORM = 0.5           # Max gradient norm for clipping

        # --- Learning Rate ---
        self.LEARNING_RATE = 3e-4          # Initial learning rate
        self.ANNEAL_LR = True              # Whether to anneal the learning rate (linear decay)

        # --- Network Architecture ---
        self.HIDDEN_DIM = 256              # Hidden dimension (kept from v2)

        # --- Logging and Saving ---
        self.SAVE_PATH = "models/ppo_bipedal_walker_hardcore_v2.pth" # New model file
        self.LOG_DIR = "logs"
        self.SAVE_FREQ = 100_000

        # --- Runtime ---
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Calculated values (Don't change these) ---
        self.BATCH_SIZE = int(self.NUM_ENVS * self.N_STEPS)
        self.NUM_MINIBATCHES = int(self.BATCH_SIZE // self.MINIBATCH_SIZE)
        assert (self.BATCH_SIZE % self.MINIBATCH_SIZE == 0), "BATCH_SIZE must be divisible by MINIBATCH_SIZE"
