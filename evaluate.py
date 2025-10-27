"""
evaluate.py

Script to load a trained model and watch it play.
You can also use this to record videos.

Usage:
    python evaluate.py
    python evaluate.py --model_path="models/your_model.pth" --record
"""

import gymnasium as gym
import torch
import time
import argparse
import os

from config import Config
from ppo_agent.model import ActorCritic


def main(args):
    config = Config()
    device = config.DEVICE
    print(f"Using device: {device}")

    # Determine render mode (visual or recording)
    render_mode = "human" if not args.record else "rgb_array"

    env = gym.make(config.ENV_NAME, render_mode=render_mode)

    # Wrapper for video recording
    if args.record:
        os.makedirs(config.VIDEO_SAVE_DIR, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            config.VIDEO_SAVE_DIR,
            episode_trigger=lambda e: e % 1 == 0,  # Record all episodes
            name_prefix=f"eval-{args.model_path.split('/')[-1]}",
        )

    # Initialize agent
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = ActorCritic(obs_dim, action_dim, config.HIDDEN_DIM).to(device)

    # Load the model
    try:
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model loaded from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train a model first with train.py")
        env.close()
        return

    agent.eval()  # Evaluation mode (no dropout, etc.)

    print("Starting evaluation...")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                # Use 'act' method in deterministic mode
                action_tensor = agent.act(obs_tensor.unsqueeze(0), deterministic=True)

            action_np = action_tensor.cpu().numpy().squeeze()
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            total_reward += reward

            if render_mode == "human":
                time.sleep(1 / 60)  # Slow down for visualization

        print(f"Episode {episode + 1}/{args.episodes}, Reward: {total_reward:.2f}")

    env.close()
    print("Evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=Config().MODEL_SAVE_PATH,
        help="Path to the saved model .pth file.",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record a video of the evaluation in the 'videos' folder.",
    )
    args = parser.parse_args()
    main(args)
