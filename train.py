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

from config import Config
from ppo_agent.model import ActorCritic
from ppo_agent.storage import RolloutBuffer

# Import the new reward wrapper
from ppo_agent.reward_wrapper import NaturalGaitRewardWrapper


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_env(env_id, seed, use_shaping=False):
    def thunk():
        env = gym.make(env_id, render_mode=None)

        # Apply the natural gait reward wrapper if enabled
        if use_shaping:
            env = NaturalGaitRewardWrapper(env)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def main():
    config = Config()

    run_name = f"{config.ENV_NAME}__natural_gait__{int(time.time())}"

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    writer = SummaryWriter(os.path.join(config.LOG_DIR, run_name))

    seed_everything(config.SEED)
    device = config.DEVICE
    print(f"Using device: {device}")

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                config.ENV_NAME, config.SEED + i, use_shaping=config.USE_REWARD_SHAPING
            )
            for i in range(config.NUM_ENVS)
        ]
    )

    obs_shape = envs.single_observation_space.shape
    action_shape = envs.single_action_space.shape

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

    if os.path.exists(config.SAVE_PATH):
        print(f"Loading existing model from {config.SAVE_PATH}")
        agent.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))

    print("Starting training...")
    start_time = time.time()

    obs, _ = envs.reset(seed=config.SEED)
    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

    num_updates = config.TOTAL_TIMESTEPS // config.BATCH_SIZE
    last_save_step = 0

    for update in range(1, num_updates + 1):

        if config.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * config.LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lr_now

        buffer.reset()

        for step in range(config.N_STEPS):
            global_step = (update - 1) * config.BATCH_SIZE + step * config.NUM_ENVS

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

            if (global_step - last_save_step) >= config.SAVE_FREQ:
                print(f"Saving model checkpoint at step {global_step}...")
                torch.save(agent.state_dict(), config.SAVE_PATH)
                last_save_step = global_step

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

                v_loss = F.mse_loss(new_value, return_b)
                entropy_loss = -entropy.mean()

                loss = (
                    pg_loss + config.VF_COEF * v_loss + config.ENT_COEF * entropy_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

        global_step_final = update * config.BATCH_SIZE
        writer.add_scalar("losses/total_loss", loss.item(), global_step_final)
        writer.add_scalar("losses/pg_loss", pg_loss.item(), global_step_final)
        writer.add_scalar("losses/v_loss", v_loss.item(), global_step_final)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step_final)
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step_final
        )

        sps = int(config.BATCH_SIZE / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step_final)
        print(f"Update {update}/{num_updates}, SPS: {sps}")

        start_time = time.time()

    envs.close()
    writer.close()

    print(f"Training finished. Final model saved to {config.SAVE_PATH}")
    torch.save(agent.state_dict(), config.SAVE_PATH)


if __name__ == "__main__":
    main()
