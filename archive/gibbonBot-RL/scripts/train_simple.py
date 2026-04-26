#!/usr/bin/env python3
"""
SIMPLIFIED TRAINING: Focus on lifting all arm bases high

This script uses a simple reward function:
- Directly reward the SUM of all arm base heights
- No complex stability rewards
- Clear goal: get all arms as high as possible
"""

import argparse
import os
import sys
from pathlib import Path

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from gibbonbot.envs.stand_env import StandEnv

# Simple training configuration
TOTAL_STEPS = 1_000_000  # 1 million steps
N_ENVS = 8  # Fewer environments for clearer learning signal


class SimpleStandEnv(StandEnv):
    """Simplified version with direct arm height rewards"""

    def step(self, action):
        # Call parent step but modify rewards to be simpler
        obs, reward, terminated, truncated, info = super().step(action)

        # Simplify reward: just use height reward directly
        # Remove complex penalties for clearer learning
        simple_reward = info.get("height_reward", 0) * 10.0  # Scale up height reward

        # Add small bonus for surviving
        if not terminated:
            simple_reward += 0.1

        return obs, simple_reward, terminated, truncated, info


def main():
    parser = argparse.ArgumentParser(description="Simplified Stand Training")
    parser.add_argument(
        "--steps", type=int, default=TOTAL_STEPS, help="Total training steps"
    )
    parser.add_argument(
        "--envs", type=int, default=N_ENVS, help="Number of parallel environments"
    )
    parser.add_argument("--watch", action="store_true", help="Enable viewer")
    parser.add_argument(
        "--inherit", action="store_true", help="Inherit from past training"
    )
    parser.add_argument(
        "--no-inherit", action="store_true", help="Don't inherit from past training"
    )
    args = parser.parse_args()

    print(f"🚀 Starting SIMPLE training")
    print(f"   Total steps: {args.steps:,}")
    print(f"   Parallel envs: {args.envs}")
    print(f"   Goal: Lift all arm bases as high as possible")
    print(f"   Reward: Direct height reward (higher arms = higher reward)")
    print()

    # Create vectorized environment
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = make_vec_env(SimpleStandEnv, n_envs=args.envs, vec_env_cls=DummyVecEnv)

    # Create PPO agent with simple configuration
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        tensorboard_log=f"runs/simple_stand/",
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=f"runs/simple_stand/checkpoints/",
        name_prefix="simple_stand",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"runs/simple_stand/best/",
        log_path=f"runs/simple_stand/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train!
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")
    finally:
        # Save final model
        model.save(f"runs/simple_stand/final_model")
        print(f"\n✅ Training completed!")
        print(f"   Final model saved to: runs/simple_stand/final_model.zip")
        print(f"   Best model saved to: runs/simple_stand/best/")
        print(f"\n📊 To view results:")
        print(f"   tensorboard --logdir runs/simple_stand/")


if __name__ == "__main__":
    main()
