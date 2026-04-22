"""
Load a trained policy and print episode stats (no viewer required).

Usage:
    python eval.py                          # load best model
    python eval.py --model runs/ape_reach/ape_reach_final.zip
    python eval.py --episodes 5
"""

import argparse
import pathlib
import numpy as np
from stable_baselines3 import PPO
from ape_env import ApeEnv

DEFAULT_MODEL = "runs/ape_reach/best/best_model.zip"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if not pathlib.Path(args.model).exists():
        print(f"No model at {args.model}")
        print("Run  python train.py  first.")
        return

    model = PPO.load(args.model)
    env   = ApeEnv(render_mode="human")

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done   = False
        total_reward = 0.0
        steps = 0
        min_dist = float("inf")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            min_dist = min(min_dist, info["dist_to_tree"])
            done = terminated or truncated

        status = "SUCCESS" if info["success"] else ("FALLEN" if info["fallen"] else "TIMEOUT")
        results.append((status, min_dist, total_reward, steps))
        print(f"Ep {ep+1:2d}: {status:<8}  "
              f"closest={min_dist:.2f}m  "
              f"final_dist={info['dist_to_tree']:.2f}m  "
              f"reward={total_reward:.1f}  steps={steps}")

    env.close()

    successes = sum(1 for r in results if r[0] == "SUCCESS")
    avg_close = np.mean([r[1] for r in results])
    print(f"\nSuccess rate: {successes}/{args.episodes}")
    print(f"Avg closest approach: {avg_close:.2f}m  (tree is 1.5m away = success)")

if __name__ == "__main__":
    main()
