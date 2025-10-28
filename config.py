import torch

class Config:
    def __init__(self):
        # --- Environment ---
        # TARGET THE HARDCORE ENVIRONMENT NOW
        self.ENV_NAME = "BipedalWalkerHardcore-v3" 
        # DEACTIVATE reward shaping - Hardcore needs full movement freedom
        self.USE_REWARD_SHAPING = False 
        
        # --- Parallel Training ---
        self.NUM_ENVS = 16 
        self.SEED = 42 
        
        # --- PPO Hyperparameters ---
        # TOTAL TIMESTEPS: Original 5M + 25M for fine-tuning on Hardcore
        self.TOTAL_TIMESTEPS = 30_000_000 
        self.N_STEPS = 1024 
        self.MINIBATCH_SIZE = 512 
        self.NUM_EPOCHS = 10 
        
        # --- Algorithm Coefficients ---
        self.GAMMA = 0.99 
        self.GAE_LAMBDA = 0.95 
        self.CLIP_COEF = 0.2 
        self.VF_COEF = 0.5 
        # Slightly increase exploration again for Hardcore obstacles
        self.ENT_COEF = 0.002 
        self.MAX_GRAD_NORM = 0.5 

        # --- Learning Rate ---
        # Keep the stable learning rate and annealing
        self.LEARNING_RATE = 1e-4 
        self.ANNEAL_LR = True 

        # --- Network Architecture ---
        self.HIDDEN_DIM = 256 
        
        # --- Saving & Logs ---
        # Path to START training FROM (Load the v3 natural model)
        self.LOAD_PATH = "models/ppo_bipedal_walker_v3_natural_v2.pth" 
        # Path to SAVE the fine-tuned Hardcore model
        self.SAVE_PATH = "models/ppo_bipedal_walker_hardcore_finetuned.pth" 
        
        self.LOG_DIR = "logs" 
        # Adjust save freq if needed, saving less often might speed up slightly
        self.SAVE_FREQ = 250_000 

        # --- Runtime ---
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        # --- Calculated Values (Do not modify) ---
        self.BATCH_SIZE = int(self.NUM_ENVS * self.N_STEPS) 
        self.NUM_MINIBATCHES = int(self.BATCH_SIZE // self.MINIBATCH_SIZE) 
        assert (self.BATCH_SIZE % self.MINIBATCH_SIZE == 0)
    