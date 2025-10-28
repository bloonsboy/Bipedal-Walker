"""
evaluate.py (Simplified Version)

Script to visualize a trained agent.
This version is simplified to run a single environment
and render it directly to the screen (human mode).
"""

import gymnasium as gym
import torch
import argparse
import time
import os
import numpy as np

# Import local modules
# We try to import the config, handle error if it fails
try:
    from config_improved import Config
except ImportError:
    print("Error: Could not import Config from config_improved.py")
    print("Please ensure config_improved.py exists and is in the same directory.")
    exit()

try:
    from ppo_agent.model import ActorCritic
except ImportError:
    print("Error: Could not import ActorCritic from ppo_agent/model.py")
    print("Please ensure ppo_agent/model.py exists and is correct.")
    exit()


def main(args):
    """
    Main function to run the evaluation.
    """

    # --- 1. Load Configuration ---
    config = Config()

    # Set device
    device = config.DEVICE
    print(f"Using device: {device}")

    # --- 2. Create Environment ---
    # Default to "human" mode to watch the agent
    render_mode = "human"
    if args.record:
        # If --record is used, switch to "rgb_array" for video capture
        print("Recording video...")
        render_mode = "rgb_array"

    # Create a single environment
    try:
        env = gym.make(config.ENV_NAME, render_mode=render_mode)
    except Exception as e:
        print(f"Error creating environment '{config.ENV_NAME}': {e}")
        print("Make sure you have run 'pip install gymnasium[box2d]'")
        return

    # Apply video wrapper ONLY if recording
    video_folder = ""
    if args.record:
        video_folder = f"videos/eval_run_{int(time.time())}"
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder,
            episode_trigger=lambda e: e == 0,  # Record only the first episode
            name_prefix=f"eval-{os.path.basename(args.model_path)}",
        )

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape

    # --- 3. Load Agent ---
    agent = ActorCritic(obs_shape[0], action_shape[0], config.HIDDEN_DIM).to(device)

    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("This script is looking for the 'hardcore_v2' model.")
        print("Please make sure you have a trained model saved at that location.")
        print("You may need to run train.py first.")
        env.close()
        return

    # Load the saved model state
    try:
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()  # Set agent to evaluation mode
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("This might be because the model file is corrupt or")
        print(
            "the model architecture in ppo_agent/model.py doesn't match the saved file."
        )
        env.close()
        return

    # --- 4. Run Evaluation ---
    total_rewards = []

    try:
        for episode in range(args.num_episodes):
            obs, info = env.reset(seed=config.SEED + episode)
            done = False
            episode_reward = 0

            while not done:
                # Convert observation to tensor and add batch dimension (from [24] to [1, 24])
                obs_tensor = (
                    torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                )

                with torch.no_grad():
                    # Get deterministic action from agent
                    # This is the line that caused the 'deterministic' error before
                    action_tensor, _, _, _ = agent.get_action_and_value(
                        obs_tensor, deterministic=True
                    )

                # Remove batch dimension and convert to numpy (from [1, 4] to [4])
                action_np = action_tensor.squeeze(0).cpu().numpy()

                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action_np)

                episode_reward += reward

                # Update observation
                obs = next_obs

                # Check if episode is finished
                done = terminated or truncated

            print(f"Episode {episode + 1}: Total Reward: {episode_reward:.2f}")
            total_rewards.append(episode_reward)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")

    finally:
        env.close()  # Always close the environment

    # --- 5. Print Results ---
    if total_rewards:
        print("\n--- Evaluation Summary ---")
        print(
            f"Average reward over {len(total_rewards)} episodes: {np.mean(total_rewards):.2f}"
        )
        print(f"Std dev of reward: {np.std(total_rewards):.2f}")

    if args.record:
        print(f"\nVideo saved to the '{video_folder}' directory.")


if __name__ == "__main__":
    # We must ensure the Config object is created to get the default path
    # This must be done carefully in case config_improved.py doesn't exist
    try:
        default_model_path = Config().SAVE_PATH
    except NameError:
        # Fallback if Config() failed
        default_model_path = "models/ppo_bipedal_walker_hardcore_v2.pth"

    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent for BipedalWalker"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=default_model_path,
        help="Path to the saved model (.pth) file.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,  # Reduced to 5 for quicker visualization
        help="Number of episodes to run for evaluation.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record a video of the first episode."
    )

    # Added simple check for args
    try:
        args = parser.parse_args()
        main(args)
    except SystemExit:
        pass  # argparse handles --help, etc.
    except Exception as e:
        print(f"A critical error occurred: {e}")
