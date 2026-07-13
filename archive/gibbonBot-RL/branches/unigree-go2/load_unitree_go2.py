from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from unitree_go2_config import MAX_EPISODE_STEPS, TRAINING_PARAMS, UNITREE_GO2_PARAMS
from unitree_go2_env import UnitreeGo2Env

# Create output directories
Path("models").mkdir(exist_ok=True)
Path("tensorboard").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)

xml_path = str(Path(UNITREE_GO2_PARAMS["xml_file"]).resolve())


def main():
    """Run the final Go1 environment setup."""
    # Note: The original tutorial includes an image showing the Go1 robot in the environment.
    # The image is available at: https://github.com/Kallinteris-Andreas/Gymnasium-kalli/assets/30759571/bf1797a3-264d-47de-b14c-e3c16072f695

    # Combine common and training-specific parameters
    env_params = {**UNITREE_GO2_PARAMS, **TRAINING_PARAMS, "xml_file": xml_path}

    env = UnitreeGo2Env(**env_params)
    # Add TimeLimit wrapper to prevent infinitely long episodes
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

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
