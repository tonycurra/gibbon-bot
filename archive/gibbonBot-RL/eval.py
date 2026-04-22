"""
Load a trained policy and watch it in the MuJoCo viewer.

Usage:
    python eval.py                          # load best model
    python eval.py --model runs/ape_reach/ape_reach_final.zip
    python eval.py --episodes 5
"""

import argparse
import pathlib
import time
import numpy as np
from stable_baselines3 import PPO
from ape_env import ApeEnv

DEFAULT_MODEL = "runs/ape_reach/best/best_model.zip"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        print(f"No model at {args.model}")
        print("Run  python train.py  first.")
        return

    model = PPO.load(args.model)
    env   = ApeEnv(render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done   = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            time.sleep(0.002)   # slow down to real-time (~500 Hz → 0.002 s/step * 5 = 10ms)

        status = "SUCCESS" if info["success"] else ("FALLEN" if info["fallen"] else "TIMEOUT")
        print(f"Episode {ep+1}: {status}  "
              f"dist={info['dist_to_tree']:.2f}m  "
              f"reward={total_reward:.1f}  steps={steps}")

    env.close()

if __name__ == "__main__":
    main()
