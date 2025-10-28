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

try:
    from config import Config  # Use the updated config file
except ImportError:
    print("Error: Could not import Config from config.py")
    print("Please ensure config.py exists.")
    exit()

try:
    from ppo_agent.model import ActorCritic
except ImportError:
    print("Error: Could not import ActorCritic from ppo_agent/model.py")
    exit()

try:
    from ppo_agent.storage import RolloutBuffer
except ImportError:
    print("Error: Could not import RolloutBuffer from ppo_agent/storage.py")
    exit()

try:
    # Try importing the reward wrapper, but don't fail if it's not used
    from ppo_agent.reward_wrapper import NaturalGaitRewardWrapper
except ImportError:
    NaturalGaitRewardWrapper = None  # Define as None if not found


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_env(env_id, seed, use_reward_shaping, idx=0, capture_video=False, run_name=""):
    def thunk():
        # Create base env
        if capture_video and idx == 0:  # Only record the first env
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode=None)

        # Apply custom reward wrapper if enabled and available
        if use_reward_shaping and NaturalGaitRewardWrapper is not None:
            print(f"Applying NaturalGaitRewardWrapper to env {idx}")
            env = NaturalGaitRewardWrapper(env)
        elif use_reward_shaping and NaturalGaitRewardWrapper is None:
            print(
                f"Warning: USE_REWARD_SHAPING is True but wrapper not found/imported."
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)  # Logs episodic returns
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


def main():
    config = Config()
    run_name = f"{config.ENV_NAME}__{os.path.basename(config.SAVE_PATH).replace('.pth','')}__{int(time.time())}"

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

    print(
        f"Initializing {config.NUM_ENVS} parallel environments for {config.ENV_NAME}..."
    )
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(config.ENV_NAME, config.SEED, config.USE_REWARD_SHAPING, i)
            for i in range(config.NUM_ENVS)
        ]
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")

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

    # --- MODIFICATION: Load model logic ---
    start_step = 0  # Initialize start_step
    # Check if a specific LOAD_PATH is defined in the config
    if (
        hasattr(config, "LOAD_PATH")
        and config.LOAD_PATH
        and os.path.exists(config.LOAD_PATH)
    ):
        print(f"Loading model for fine-tuning from: {config.LOAD_PATH}")
        try:
            agent.load_state_dict(torch.load(config.LOAD_PATH, map_location=device))
            # Optional: Extract step count if saved in filename or elsewhere
            # start_step = extract_step_from_filename(config.LOAD_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from LOAD_PATH: {e}. Starting from scratch.")
            start_step = 0
    # Fallback: Check if the SAVE_PATH exists (for resuming interrupted training)
    elif os.path.exists(config.SAVE_PATH):
        print(f"Resuming training by loading model from: {config.SAVE_PATH}")
        try:
            agent.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
            # Optional: Extract step count if resuming
            # start_step = extract_step_from_filename(config.SAVE_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from SAVE_PATH: {e}. Starting from scratch.")
            start_step = 0
    else:
        print("No existing model found. Starting training from scratch.")
        start_step = 0
    # --- END MODIFICATION ---

    print("Starting training...")
    start_time = time.time()
    obs, _ = envs.reset(seed=config.SEED)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    # Calculate total steps and updates needed, considering the start_step if resuming/fine-tuning
    total_steps_needed = config.TOTAL_TIMESTEPS
    remaining_steps = total_steps_needed - start_step
    num_updates = remaining_steps // config.BATCH_SIZE
    start_update = (
        start_step // config.BATCH_SIZE
    ) + 1  # Calculate starting update number

    print(f"Total target timesteps: {config.TOTAL_TIMESTEPS}")
    print(f"Starting from step: {start_step}")
    print(f"Remaining steps: {remaining_steps}")
    print(f"Batch size (N_ENVS * N_STEPS): {config.BATCH_SIZE}")
    print(f"Total number of updates to run: {num_updates}")
    print(f"Starting from update: {start_update}")

    last_save_step = start_step

    # Adjust the loop range
    for update in range(start_update, start_update + num_updates):

        if config.ANNEAL_LR:
            # Adjust annealing based on the total number of updates planned FROM THE START
            total_updates_from_start = config.TOTAL_TIMESTEPS // config.BATCH_SIZE
            frac = 1.0 - (update - 1.0) / total_updates_from_start
            lr_now = max(
                frac * config.LEARNING_RATE, 1e-6
            )  # Ensure lr doesn't go to zero
            optimizer.param_groups[0]["lr"] = lr_now

        buffer.reset()
        current_rollout_start_time = time.time()  # Timer for rollout phase

        for step in range(config.N_STEPS):
            global_step = (update - 1) * config.BATCH_SIZE + step * config.NUM_ENVS
            # Ensure global_step starts correctly if resuming
            global_step = max(global_step, start_step + step * config.NUM_ENVS)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(obs_tensor)

            action_np = action.cpu().numpy()
            next_obs, reward, done, truncated, info = envs.step(action_np)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(device)

            buffer.add(
                obs_tensor,
                action,
                log_prob,
                reward_tensor,
                done_tensor,
                value.squeeze(),
            )
            obs_tensor = next_obs_tensor

            if "_episode" in info:
                finished_envs_mask = info["_episode"]
                episode_rewards = info["episode"]["r"][finished_envs_mask]
                episode_lengths = info["episode"]["l"][finished_envs_mask]

                for i in range(len(episode_rewards)):
                    # More accurate global step logging for episodes
                    actual_global_step = (
                        global_step + i
                    )  # Approximate step when episode ended
                    print(
                        f"global_step={actual_global_step}, episode_reward={episode_rewards[i]:.2f}"
                    )
                    writer.add_scalar(
                        "charts/episodic_reward", episode_rewards[i], actual_global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", episode_lengths[i], actual_global_step
                    )

            # Checkpoint saving logic
            if (global_step - last_save_step) >= config.SAVE_FREQ:
                print(f"Saving model checkpoint at step {global_step}...")
                torch.save(agent.state_dict(), config.SAVE_PATH)
                last_save_step = global_step

        # --- Update Phase ---
        update_start_time = time.time()  # Timer for update phase
        with torch.no_grad():
            last_value = agent.get_value(obs_tensor)
            last_done = torch.tensor(done, dtype=torch.float32).to(device).unsqueeze(-1)
            buffer.compute_returns_and_advantages(last_value, last_done)

        for epoch in range(config.NUM_EPOCHS):
            for batch in buffer.get_batch(config.MINIBATCH_SIZE):
                obs_b, act_b, old_log_prob_b, adv_b, return_b = batch
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    obs_b, act_b
                )

                log_ratio = new_log_prob - old_log_prob_b
                ratio = torch.exp(log_ratio)

                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1 - config.CLIP_COEF, 1 + config.CLIP_COEF
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_value = new_value.view(-1)  # Ensure shape matches return_b
                v_loss = F.mse_loss(new_value, return_b)

                entropy_loss = -entropy.mean()
                loss = (
                    pg_loss + config.VF_COEF * v_loss + config.ENT_COEF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

        # --- Logging after update ---
        update_end_time = time.time()
        rollout_time = update_start_time - current_rollout_start_time
        update_time = update_end_time - update_start_time
        total_update_cycle_time = update_end_time - current_rollout_start_time

        global_step_final = update * config.BATCH_SIZE
        # Ensure final global step calculation is correct if resuming
        global_step_final = max(
            global_step_final,
            start_step + config.BATCH_SIZE * (update - start_update + 1),
        )

        writer.add_scalar("losses/total_loss", loss.item(), global_step_final)
        writer.add_scalar("losses/pg_loss", pg_loss.item(), global_step_final)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step_final)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step_final)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step_final
        )

        # Calculate SPS based on the total time for one update cycle
        sps = int(config.BATCH_SIZE / total_update_cycle_time)
        writer.add_scalar("charts/SPS", sps, global_step_final)
        print(
            f"Update {update}/{start_update + num_updates -1}, Global Step ~{global_step_final}, SPS: {sps}, Rollout: {rollout_time:.2f}s, Update: {update_time:.2f}s"
        )

    envs.close()
    writer.close()

    print(f"Training finished. Final model saved to {config.SAVE_PATH}")
    torch.save(agent.state_dict(), config.SAVE_PATH)


if __name__ == "__main__":
    main()
