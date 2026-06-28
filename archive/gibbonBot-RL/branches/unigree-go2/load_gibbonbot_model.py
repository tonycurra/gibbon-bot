from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from gibbonbot_env import GibbonbotEnv

# Create output directories
Path("models").mkdir(exist_ok=True)
Path("tensorboard").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)

xml_path = str(Path("scene.xml").resolve())


def main():
    """Run the final Go1 environment setup."""
    # Note: The original tutorial includes an image showing the Go1 robot in the environment.
    # The image is available at: https://github.com/Kallinteris-Andreas/Gymnasium-kalli/assets/30759571/bf1797a3-264d-47de-b14c-e3c16072f695

    env = GibbonbotEnv(
        xml_file=xml_path,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        main_body=0,  # Index of the torso body
        healthy_z_range=(0.3, 1.0),  # Adjusted for the robot's height
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.1,
        frame_skip=5,  # 25, assuming default timestep of 0.002, dt=0.05
        render_mode="rgb_array",  # Change to "human" to visualize, otherwise rgb_array
    )
    # Add TimeLimit wrapper to prevent infinitely long episodes
    env = TimeLimit(env, max_episode_steps=1000)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run: gb1_{run_name}")

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints/latest/",
        name_prefix=f"gb1_{run_name}",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/",
        device="cpu",
    )

    model.learn(
        total_timesteps=1_000_000,
        tb_log_name=f"gb1_{run_name}",
        callback=checkpoint_callback,
    )

    model.save(f"models/gibbonbot_{run_name}_final")

    env.close()


if __name__ == "__main__":
    main()
