import torch


class Config:
    def __init__(self):
        """
        Configuration class for PPO training.
        All hyperparameters and settings are defined here.
        """

        # --- Environment Hyperparameters ---
        self.ENV_NAME = "BipedalWalkerHardcore-v3"
        self.NUM_ENVS = 8  # Number of parallel environments
        self.SEED = 42

        # --- PPO Hyperparameters ---
        self.TOTAL_TIMESTEPS = 10_000_000  # Total steps for the entire training
        self.N_STEPS = 512  # Steps collected per environment FOR EACH update (batch_size = NUM_ENVS * N_STEPS)
        self.NUM_EPOCHS = 10  # Number of epochs to train on the collected data
        self.MINIBATCH_SIZE = 256  # Size of minibatches for SGD
        self.GAMMA = 0.99  # Discount factor
        self.GAE_LAMBDA = 0.95  # Lambda for GAE (Generalized Advantage Estimation)
        self.CLIP_COEF = 0.2  # PPO clipping coefficient
        self.ENT_COEF = (
            0.001  # Entropy coefficient (lower for hardcore, we want exploitation)
        )
        self.VF_COEF = 0.5  # Value function coefficient
        self.MAX_GRAD_NORM = 0.5  # Max gradient norm for clipping

        # --- Learning Rate ---
        # We use a linear learning rate decay
        self.LEARNING_RATE = 3e-4  # Initial learning rate
        self.ANNEAL_LR = True  # Whether to anneal the learning rate (linear decay)

        # --- Network Architecture ---
        self.HIDDEN_DIM = 256  # Hidden dimension for Actor and Critic networks

        # --- Logging and Saving ---
        self.SAVE_PATH = "models/ppo_bipedal_walker_hardcore_v2.pth"
        self.LOG_DIR = "logs"
        self.SAVE_FREQ = (
            100_000  # Save model every X global steps (relative to one env)
        )

        # --- Runtime ---
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Calculated values (Don't change these) ---
        # BATCH_SIZE is the total number of steps collected from all environments per update
        self.BATCH_SIZE = int(self.NUM_ENVS * self.N_STEPS)

        # NUM_MINIBATCHES is the number of minibatches to split the BATCH_SIZE into
        self.NUM_MINIBATCHES = int(self.BATCH_SIZE // self.MINIBATCH_SIZE)

        # Ensure minibatch size is valid
        assert (
            self.BATCH_SIZE % self.MINIBATCH_SIZE == 0
        ), "BATCH_SIZE must be divisible by MINIBATCH_SIZE"
