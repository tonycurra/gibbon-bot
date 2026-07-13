import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

from unitree_go2_config import (
    MAX_EPISODE_STEPS,
    UNITREE_GO2_PARAMS,
    VISUALIZATION_PARAMS,
)
from unitree_go2_env import UnitreeGo2Env

checkpoint_dir = Path("checkpoints/latest")  # checkpoints
# xml_path = str(Path("unitree_go1/scene.xml").resolve())
xml_path = str(Path(UNITREE_GO2_PARAMS["xml_file"]).resolve())

if len(sys.argv) > 1:
    # Use provided checkpoint
    checkpoint = Path(sys.argv[1])
else:
    # Use latest checkpoint
    checkpoints = list(checkpoint_dir.glob("*.zip"))

    if not checkpoints:
        raise FileNotFoundError("No checkpoints found in checkpoints/")

    checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)

print(f"Loading checkpoint: {checkpoint}")

# Combine common and visualization-specific parameters
env_params = {**UNITREE_GO2_PARAMS, **VISUALIZATION_PARAMS, "xml_file": xml_path}

env = UnitreeGo2Env(**env_params)
# Add TimeLimit wrapper to match the training environment
env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)

model = PPO.load(checkpoint, device="cpu")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
