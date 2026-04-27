#!/usr/bin/env python3
"""
Replay a trained checkpoint.

Usage:
    python analyze.py                        # latest checkpoint, continuous
    python analyze.py ape_stand_50000.zip    # specific checkpoint
    python analyze.py --episodes 3
    python analyze.py --speed 0.02           # faster playback
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from stable_baselines3 import PPO

from gibbonbot.envs.stand_env import StandEnv

RUNS = pathlib.Path("runs/ape_stand")


def best_checkpoint():
    best = RUNS / "best" / "best_model.zip"
    if best.exists():
        return best
    ckpts = sorted(RUNS.glob("ape_stand_*.zip"))
    return ckpts[-1] if ckpts else None


def run(ckpt_path, args):
    print(f"Loading: {ckpt_path}")
    policy = PPO.load(str(ckpt_path))
    env = StandEnv(render_mode="human")

    try:
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            obs, _ = env.reset()
            total_reward = 0.0
            step_count = 0
            done = False

            while not done and step_count < 500:
                action, _ = policy.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                done = terminated or truncated

                sys.stdout.write(
                    f"\r  Step: {step_count:4d} | Reward: {total_reward:7.2f} | "
                    f"Height: {info['above_ground']:.3f}m | "
                    f"{'FALLEN' if info['fallen'] else 'ALIVE'}"
                )
                sys.stdout.flush()

                if hasattr(env, "_viewer") and env._viewer and not env._viewer.is_running():
                    return

                time.sleep(args.speed)

            status = "FALLEN" if info.get("fallen", False) else "ALIVE"
            print(f"\n  {status} | Reward: {total_reward:.2f} | Steps: {step_count}")

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="latest")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--speed", type=float, default=0.05)
    args = parser.parse_args()

    if args.checkpoint == "latest":
        ckpt = best_checkpoint()
        if not ckpt:
            print("No checkpoints found.")
            sys.exit(1)
    else:
        ckpt = RUNS / args.checkpoint
        if not ckpt.exists():
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    run(ckpt, args)


if __name__ == "__main__":
    main()
