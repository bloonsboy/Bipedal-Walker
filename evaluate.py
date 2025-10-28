import gymnasium as gym
import torch
import argparse
import time
import os
import numpy as np

from config import Config
from ppo_agent.model import ActorCritic

# Import the new reward wrapper
from ppo_agent.reward_wrapper import NaturalGaitRewardWrapper


def main(args):
    config = Config()
    device = config.DEVICE
    print(f"Using device: {device}")

    render_mode = "human"
    if args.record:
        print("Recording video...")
        render_mode = "rgb_array"

    try:
        env = gym.make(config.ENV_NAME, render_mode=render_mode)
        # Apply the wrapper if it was used during training
        if config.USE_REWARD_SHAPING:
            env = NaturalGaitRewardWrapper(env)

    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    video_folder = ""
    if args.record:
        video_folder = (
            f"videos/eval_run_{os.path.basename(config.SAVE_PATH)}_{int(time.time())}"
        )
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda e: True,
            name_prefix=f"eval-{os.path.basename(args.model_path)}",
        )

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    agent = ActorCritic(obs_shape[0], action_shape[0], config.HIDDEN_DIM).to(device)

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        env.close()
        return

    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        env.close()
        return

    total_rewards = []

    try:
        for episode in range(args.num_episodes):
            obs, info = env.reset(seed=config.SEED + episode)
            done = False
            episode_reward = 0

            while not done:
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                )

                with torch.no_grad():
                    action_tensor, _, _, _ = agent.get_action_and_value(
                        obs_tensor, deterministic=True
                    )

                action_np = action_tensor.squeeze(0).cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action_np)
                episode_reward += reward
                obs = next_obs
                done = terminated or truncated

            print(f"Episode {episode + 1}: Total Reward: {episode_reward:.2f}")
            total_rewards.append(episode_reward)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

    finally:
        env.close()

    if total_rewards:
        print("\n--- Evaluation Summary ---")
        print(
            f"Average reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}"
        )
        print(f"Std dev of reward: {np.std(total_rewards):.2f}")

    if args.record:
        print(f"\nVideo saved to the '{video_folder}' directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=Config().SAVE_PATH,
        help="Path to the saved model (.pth) file.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run for evaluation.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record a video of the evaluation episodes.",
    )

    try:
        args = parser.parse_args()
        main(args)
    except Exception as e:
        print(f"A critical error occurred: {e}")
