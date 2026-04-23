"""
Load a trained policy and run evaluation episodes with a live viewer.

Usage:
    python eval.py                              # reach task, best model
    python eval.py --task stand                 # stand task, best model
    python eval.py --task stand --model runs/ape_stand/ape_stand_final.zip
    python eval.py --episodes 10
"""

import argparse
import pathlib
import numpy as np
from stable_baselines3 import PPO

TASK_DEFAULTS = {
    "reach": "runs/ape_reach/best/best_model.zip",
    "stand": "runs/ape_stand/best/best_model.zip",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     default="reach", choices=list(TASK_DEFAULTS))
    parser.add_argument("--model",    default=None)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    model_path = args.model or TASK_DEFAULTS[args.task]
    if not pathlib.Path(model_path).exists():
        print(f"No model at {model_path}")
        print(f"Run  python train.py --task {args.task}  first.")
        return

    if args.task == "reach":
        from ape_env import ApeEnv as EnvClass
    else:
        from stand_env import StandEnv as EnvClass

    model = PPO.load(model_path)
    env   = EnvClass(render_mode="human")

    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        done   = False
        total_reward = 0.0
        steps  = 0
        best_z = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            best_z = max(best_z, info.get("torso_z", 0.0))
            done = terminated or truncated

        if args.task == "reach":
            status = "SUCCESS" if info["success"] else ("FALLEN" if info["fallen"] else "TIMEOUT")
            print(f"Ep {ep+1:2d}: {status:<8}  "
                  f"closest={info.get('dist_to_tree', 0):.2f}m  "
                  f"reward={total_reward:.1f}  steps={steps}")
        else:
            status = "FALLEN" if info["fallen"] else "ALIVE"
            print(f"Ep {ep+1:2d}: {status:<6}  "
                  f"best_z={best_z:.3f}m  "
                  f"final_z={info['torso_z']:.3f}m  "
                  f"reward={total_reward:.1f}  steps={steps}")

        results.append((status, total_reward, steps))

    env.close()

    print(f"\n{'='*50}")
    avg_reward = np.mean([r[1] for r in results])
    print(f"Avg reward: {avg_reward:.1f}  over {args.episodes} episodes")
    if args.task == "reach":
        successes = sum(1 for r in results if r[0] == "SUCCESS")
        print(f"Success rate: {successes}/{args.episodes}")


if __name__ == "__main__":
    main()
