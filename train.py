import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import time
import random
import os

# Import local modules
from config_improved import Config  # Use the "pro" config
from ppo_agent.model import ActorCritic
from ppo_agent.storage import RolloutBuffer


def seed_everything(seed):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_env(env_id, seed, capture_video=False, run_name=""):
    """
    Wrapper function to create a single environment instance.
    This is required by AsyncVectorEnv.
    """

    def thunk():
        if capture_video:
            # Note: Video recording is usually done in evaluate.py
            # This is just an example if you wanted to record training
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode=None)

        # We use RecordEpisodeStatistics to automatically log episodic returns
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def main():
    config = Config()

    # --- 1. Initialization ---
    run_name = f"{config.ENV_NAME}__{int(time.time())}"

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

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

    # --- 2. Environment Setup (Vectorized) ---
    print(f"Initializing {config.NUM_ENVS} parallel environments...")
    # Create NUM_ENVS parallel environments
    envs = gym.vector.AsyncVectorEnv(
        [make_env(config.ENV_NAME, config.SEED + i) for i in range(config.NUM_ENVS)]
    )

    # Check dimensions
    # .single_..._space is used to get the shape from one of the parallel envs
    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")

    # --- 3. Agent and Buffer Setup ---
    agent = ActorCritic(obs_shape[0], action_shape[0], config.HIDDEN_DIM).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

    buffer = RolloutBuffer(
        config.N_STEPS,
        config.NUM_ENVS,
        obs_shape,
        action_shape,
        device,
        config.GAMMA,
        config.GAE_LAMBDA,
    )

    # Attempt to load a checkpoint if it exists
    if os.path.exists(config.SAVE_PATH):
        print(f"Loading existing model from {config.SAVE_PATH}")
        agent.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
        # Note: We don't save/load optimizer state for simplicity,
        # but for perfect resumption, you would.

    # --- 4. Training Loop ---
    print("Starting training...")
    start_time = time.time()

    # Reset environments
    obs, _ = envs.reset(seed=config.SEED)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    # Calculate total number of updates
    # BATCH_SIZE = num_envs * n_steps
    num_updates = config.TOTAL_TIMESTEPS // config.BATCH_SIZE

    print(f"Total timesteps: {config.TOTAL_TIMESTEPS}")
    print(f"Batch size (N_ENVS * N_STEPS): {config.BATCH_SIZE}")
    print(f"Number of updates: {num_updates}")

    last_save_step = 0

    for update in range(1, num_updates + 1):

        # --- A. Learning Rate Annealing ---
        if config.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * config.LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lr_now

        # --- B. Collection Phase (Rollout) ---
        buffer.reset()

        for step in range(config.N_STEPS):
            # Calculate global step (total steps across all envs)
            global_step = (update - 1) * config.BATCH_SIZE + step * config.NUM_ENVS

            with torch.no_grad():
                # Get action from the agent
                # obs_tensor shape: [num_envs, obs_dim]
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()

            # Execute action in all parallel environments
            next_obs, reward, done, truncated, info = envs.step(action_np)

            # Convert to tensors
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

            # Store data in buffer
            buffer.add(
                obs_tensor,
                action,
                log_prob,
                reward_tensor,
                done_tensor,
                value.squeeze(),
            )

            # Prepare for next step
            obs_tensor = next_obs_tensor

            # --- C. Logging Episodic Info ---
            # The 'info' dict from vectorized envs is different
            # We must check `info["_episode"]`
            if "_episode" in info:
                # This mask will be True for envs that just finished
                finished_envs_mask = info["_episode"]

                # Get rewards and lengths for all finished envs
                episode_rewards = info["episode"]["r"][finished_envs_mask]
                episode_lengths = info["episode"]["l"][finished_envs_mask]

                for i in range(len(episode_rewards)):
                    # We log one data point per finished episode
                    # The global_step is an approximation of when the episode finished
                    current_global_step = global_step + i
                    print(
                        f"global_step={current_global_step}, episode_reward={episode_rewards[i]:.2f}"
                    )
                    writer.add_scalar(
                        "charts/episodic_reward",
                        episode_rewards[i],
                        current_global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        episode_lengths[i],
                        current_global_step,
                    )

            # --- D. Saving Checkpoint ---
            # Save based on global_step
            if (global_step - last_save_step) >= config.SAVE_FREQ:
                print(f"Saving model checkpoint at step {global_step}...")
                torch.save(agent.state_dict(), config.SAVE_PATH)
                last_save_step = global_step

        # --- E. Update Phase (PPO Learning) ---

        # 1. Calculate advantages and returns
        with torch.no_grad():
            # Get value of the last observation
            last_value = agent.get_value(obs_tensor)  # Shape: [num_envs, 1]
            # Get done flag of the last observation
            last_done = (
                torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(-1)
            )  # Shape: [num_envs, 1]

            buffer.compute_returns_and_advantages(last_value, last_done)

        # 2. Train for N_EPOCHS
        for epoch in range(config.NUM_EPOCHS):
            # Get minibatches from buffer
            for batch in buffer.get_batch(config.MINIBATCH_SIZE):
                obs_b, act_b, old_log_prob_b, adv_b, return_b = batch

                # Recalculate values with current policy
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    obs_b, act_b
                )

                # --- Loss Calculation ---

                # Policy Loss (Actor)
                log_ratio = new_log_prob - old_log_prob_b
                ratio = torch.exp(log_ratio)

                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1 - config.CLIP_COEF, 1 + config.CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (Critic)
                v_loss = F.mse_loss(new_value, return_b)

                # Entropy Loss (Exploration)
                entropy_loss = -entropy.mean()

                # Total Loss
                loss = (
                    pg_loss + config.VF_COEF * v_loss + config.ENT_COEF * entropy_loss
                )

                # --- Optimization ---
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

        # Log losses and SPS
        global_step_final = update * config.BATCH_SIZE
        writer.add_scalar("losses/total_loss", loss.item(), global_step_final)
        writer.add_scalar("losses/pg_loss", pg_loss.item(), global_step_final)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step_final)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step_final)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step_final
        )

        # Calculate Steps Per Second
        sps = int(config.BATCH_SIZE / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step_final)
        print(f"Update {update}/{num_updates}, SPS: {sps}")

        # Reset start time for next SPS calculation
        start_time = time.time()

    # --- 5. End of Training ---
    envs.close()
    writer.close()

    # Save final model
    print(f"Training finished. Final model saved to {config.SAVE_PATH}")
    torch.save(agent.state_dict(), config.SAVE_PATH)


if __name__ == "__main__":
    main()
