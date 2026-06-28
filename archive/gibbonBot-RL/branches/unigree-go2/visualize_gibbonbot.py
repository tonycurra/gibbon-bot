import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

checkpoint_dir = Path("checkpoints/latest")  # checkpoints
# xml_path = str(Path("unitree_go1/scene.xml").resolve())
xml_path = str(Path("scene.xml").resolve())

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


env = gym.make(
    "Ant-v5",
    xml_file=xml_path,
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=5,  # 25
    max_episode_steps=1000,
    render_mode="human",  # Change to "human" to visualize
)

model = PPO.load(checkpoint, device="cpu")

obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
