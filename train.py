"""
train.py

Main script to launch the PPO agent's training.
This is the file you run to start learning.
"""

import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn  # For nn.utils.clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import random
import os

# Import local modules
from config_improved import Config
# from config_simple import Config
from ppo_agent.model import ActorCritic
from ppo_agent.storage import RolloutBuffer


def seed_everything(seed):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    config = Config()

    # --- 1. Initialization ---
    run_name = f"{config.ENV_NAME}__{int(time.time())}"

    # Create save directories if they don't exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

    writer = SummaryWriter(os.path.join(config.LOG_DIR, run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % (
            "\n".join(
                [
                    f"|{key}|{getattr(config, key)}|"
                    for key in dir(config)
                    if not key.startswith("__")
                ]
            )
        ),
    )

    seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    # Create the environment
    env = gym.make(config.ENV_NAME, render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # Crucial for logging rewards

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Initialize Agent and Optimizer
    agent = ActorCritic(obs_dim, action_dim, config.HIDDEN_DIM).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

    # Initialize Buffer
    buffer = RolloutBuffer(
        config.N_STEPS, obs_dim, action_dim, device, config.GAMMA, config.GAE_LAMBDA
    )

    # --- 2. Training Loop ---
    print("Starting training...")
    start_time = time.time()

    obs, _ = env.reset(seed=config.SEED)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
    done = False

    global_step = 0
    num_updates = config.TOTAL_TIMESTEPS // config.N_STEPS

    for update in range(1, num_updates + 1):

        # --- A. Collection Phase (Rollout) ---
        buffer.reset()

        for step in range(config.N_STEPS):
            global_step += 1

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(
                    obs_tensor.unsqueeze(0)
                )

            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            buffer.add(
                obs_tensor,
                action.squeeze(),
                log_prob,
                torch.tensor(reward, dtype=torch.float32).to(device),
                torch.tensor(done, dtype=torch.float32).to(device),
                value,
            )

            obs = next_obs
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            if done:
                if "episode" in info:
                    print(
                        f"global_step={global_step}, episode_reward={info['episode']['r']:.2f}"
                    )
                    # This is the main performance metric
                    writer.add_scalar(
                        "charts/episodic_reward", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

                obs, _ = env.reset()
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                done = False

        # --- B. Update Phase (PPO Learning) ---

        with torch.no_grad():
            last_value = agent.get_value(obs_tensor.unsqueeze(0)).reshape(1, -1)
            buffer.compute_returns_and_advantages(
                last_value, torch.tensor(done, dtype=torch.float32).to(device)
            )

        advantages = buffer.advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )  # Normalization

        for epoch in range(config.N_EPOCHS):
            for batch in buffer.get_batch(config.BATCH_SIZE):
                obs_b, act_b, old_log_prob_b, adv_b, return_b = batch

                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    obs_b, act_b
                )

                # Actor Loss (Policy)
                log_ratio = new_log_prob - old_log_prob_b
                ratio = torch.exp(log_ratio)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1 - config.CLIP_EPSILON, 1 + config.CLIP_EPSILON
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic Loss (Value)
                v_loss = F.mse_loss(new_value, return_b)

                # Entropy Loss (Exploration)
                entropy_loss = -entropy.mean()

                loss = (
                    pg_loss + config.VF_COEF * v_loss + config.ENT_COEF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

        # Log stability metrics
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/pg_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

        # Regular model saving
        if update % 50 == 0:
            torch.save(agent.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")

    # --- 3. End of Training ---
    env.close()
    writer.close()

    torch.save(agent.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Training finished. Final model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
