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
    from config import Config
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

# --- MODIFICATION: Correctly import the specific wrapper ---
try:
    # Attempt to import the specific class needed
    from ppo_agent.reward_wrapper import HardcoreClimbingRewardWrapper

    # Set a flag or variable to indicate success
    RewardWrapperClass = HardcoreClimbingRewardWrapper
    print("Successfully imported HardcoreClimbingRewardWrapper.")
except ImportError:
    # If import fails, set the flag/variable to None
    RewardWrapperClass = None
    print(
        "Could not import HardcoreClimbingRewardWrapper. Reward shaping might not be applied if enabled."
    )
# --- END MODIFICATION ---


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# --- MODIFICATION: Use the imported wrapper class ---
def make_env(env_id, seed, use_reward_shaping, idx=0, capture_video=False, run_name=""):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode=None)

        # Apply custom reward wrapper ONLY if enabled AND the class was imported successfully
        if use_reward_shaping and RewardWrapperClass is not None:
            print(f"Applying {RewardWrapperClass.__name__} to env {idx}")
            env = RewardWrapperClass(env)  # Use the imported class
        elif use_reward_shaping and RewardWrapperClass is None:
            # This warning should now only appear if the import truly failed
            print(
                f"Warning: USE_REWARD_SHAPING is True but wrapper class import failed."
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk


# --- END MODIFICATION ---


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
    print(f"Using device: {device}")  # Still check this output!

    print(
        f"Initializing {config.NUM_ENVS} parallel environments for {config.ENV_NAME}..."
    )
    # Pass the config flag directly to make_env
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

    start_step = 0
    if (
        hasattr(config, "LOAD_PATH")
        and config.LOAD_PATH
        and os.path.exists(config.LOAD_PATH)
    ):
        print(f"Loading model for fine-tuning from: {config.LOAD_PATH}")
        try:
            agent.load_state_dict(torch.load(config.LOAD_PATH, map_location=device))
            print("Model loaded successfully.")
            # Basic step count assumption: if loading, assume previous training completed fully for LR schedule
            # A more robust method would save/load step count with the model
            try:
                # Try to infer start step based on previous total timesteps if fine-tuning
                # This logic assumes the LOAD_PATH model finished a run of TOTAL_TIMESTEPS defined previously
                # It's an approximation for the LR scheduler.
                # Example: If previous config had 5M steps, start_step = 5_000_000
                # NOTE: This requires knowledge of the previous run's config or saving step count explicitly.
                # For simplicity, we'll keep start_step = 0 for now, affecting only LR schedule start point.
                # If you know the previous run finished at N steps, set start_step = N manually here.
                print(
                    f"Approximating start_step as 0 for LR scheduling. Adjust manually if needed."
                )
            except Exception:
                print("Could not infer start step. Using 0.")
                start_step = 0

        except Exception as e:
            print(f"Error loading model from LOAD_PATH: {e}. Starting from scratch.")
            start_step = 0
    elif os.path.exists(config.SAVE_PATH):
        print(f"Resuming training by loading model from: {config.SAVE_PATH}")
        try:
            agent.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
            print("Model loaded successfully.")
            # Same approximation issue for start_step if resuming
            start_step = 0  # Approximation
        except Exception as e:
            print(f"Error loading model from SAVE_PATH: {e}. Starting from scratch.")
            start_step = 0
    else:
        print("No existing model found. Starting training from scratch.")
        start_step = 0

    print("Starting training...")
    start_time = time.time()
    obs, _ = envs.reset(seed=config.SEED)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    total_steps_needed = config.TOTAL_TIMESTEPS
    # Adjust remaining steps based on the (approximate) start_step
    # Note: If start_step is 0, this runs for the full TOTAL_TIMESTEPS again
    remaining_steps = total_steps_needed - start_step
    if remaining_steps <= 0:
        print(
            f"Target TOTAL_TIMESTEPS ({total_steps_needed}) already reached or exceeded by start_step ({start_step}). No training needed."
        )
        envs.close()
        writer.close()
        return

    num_updates = remaining_steps // config.BATCH_SIZE
    start_update = (start_step // config.BATCH_SIZE) + 1

    print(f"Total target timesteps: {config.TOTAL_TIMESTEPS}")
    print(f"Starting from step (approx): {start_step}")
    print(f"Remaining steps to train: {remaining_steps}")
    print(f"Batch size (N_ENVS * N_STEPS): {config.BATCH_SIZE}")
    print(f"Total number of updates to run: {num_updates}")
    print(f"Starting from update: {start_update}")

    last_save_step = start_step

    for update in range(start_update, start_update + num_updates):

        if config.ANNEAL_LR:
            # Annealing should consider the TOTAL intended training duration (TOTAL_TIMESTEPS)
            total_intended_updates = config.TOTAL_TIMESTEPS // config.BATCH_SIZE
            # Calculate current progress fraction relative to the total intended training duration
            frac = 1.0 - (update - 1.0) / total_intended_updates
            # Ensure fraction doesn't go below zero if we run longer than initially planned
            frac = max(frac, 0.0)
            lr_now = frac * config.LEARNING_RATE
            # Prevent LR from becoming exactly zero, set a minimum
            lr_now = max(lr_now, 1e-6)
            optimizer.param_groups[0]["lr"] = lr_now

        buffer.reset()
        current_rollout_start_time = time.time()

        for step in range(config.N_STEPS):
            # Calculate accurate global_step based on update number and step within rollout
            global_step = (
                start_step
                + (update - start_update) * config.BATCH_SIZE
                + step * config.NUM_ENVS
            )

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

                # Log stats for each finished episode
                for i in range(len(episode_rewards)):
                    # Try to get a more accurate step count for episode end
                    actual_global_step = global_step + i  # Approximation
                    print(
                        f"global_step={actual_global_step}, episode_reward={episode_rewards[i]:.2f}"
                    )
                    writer.add_scalar(
                        "charts/episodic_reward", episode_rewards[i], actual_global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", episode_lengths[i], actual_global_step
                    )

            # Checkpoint saving logic based on accumulated steps
            effective_steps_done = (
                global_step + config.NUM_ENVS
            )  # Steps done *after* this loop iteration
            if (effective_steps_done - last_save_step) >= config.SAVE_FREQ:
                current_save_step = (
                    effective_steps_done // config.SAVE_FREQ
                ) * config.SAVE_FREQ  # Align save step
                print(f"Saving model checkpoint at step {current_save_step}...")
                torch.save(agent.state_dict(), config.SAVE_PATH)
                last_save_step = current_save_step

        update_start_time = time.time()
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

                new_value = new_value.view(-1)
                v_loss = F.mse_loss(new_value, return_b)

                entropy_loss = -entropy.mean()
                loss = (
                    pg_loss + config.VF_COEF * v_loss + config.ENT_COEF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

        update_end_time = time.time()
        rollout_time = update_start_time - current_rollout_start_time
        update_time = update_end_time - update_start_time
        total_update_cycle_time = update_end_time - current_rollout_start_time

        # Final global step for this update cycle
        global_step_final = start_step + (update - start_update + 1) * config.BATCH_SIZE

        writer.add_scalar("losses/total_loss", loss.item(), global_step_final)
        writer.add_scalar("losses/pg_loss", pg_loss.item(), global_step_final)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step_final)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step_final)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step_final
        )

        sps = (
            int(config.BATCH_SIZE / total_update_cycle_time)
            if total_update_cycle_time > 0
            else 0
        )
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
