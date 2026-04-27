#!/usr/bin/env python3
"""
Replay and analyze a trained checkpoint.

Usage:
    python analyze.py                    # latest checkpoint, 10 episodes
    python analyze.py mjx_00010000000.npz
    python analyze.py --step             # frame-by-frame
    python analyze.py --speed 0.1        # slower playback
    python analyze.py --episodes 3
"""

import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from gibbonbot.config.train_mjx import OBS_DIM, ActorCritic
from gibbonbot.envs.stand_env import StandEnv

RUNS = pathlib.Path("runs/ape_stand_mjx")


def get_latest_checkpoint():
    ckpts = sorted(RUNS.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_params(path):
    data = np.load(str(path))
    flat = [
        jnp.array(data[k])
        for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))
    ]
    net = ActorCritic()
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, flat)


def run(checkpoint_path, args):
    print(f"Analyzing: {checkpoint_path}")
    params = load_params(checkpoint_path)
    net = ActorCritic()

    @jax.jit
    def act(params, obs):
        mean, log_std, _ = net.apply(params, obs)
        return jnp.tanh(mean)

    env = StandEnv(render_mode="human")
    _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)

    try:
        for episode in range(args.episodes):
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            obs, _ = env.reset()
            obs_padded = np.concatenate([obs, _TREE_PAD])
            total_reward = 0.0
            step_count = 0
            done = False

            while not done and step_count < 500:
                action = np.array(act(params, jnp.array(obs_padded)))
                obs, reward, terminated, truncated, info = env.step(action)
                obs_padded = np.concatenate([obs, _TREE_PAD])
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

                if args.step:
                    input(" <Enter>")
                else:
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
    parser.add_argument("--step", action="store_true")
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    if args.checkpoint == "latest":
        ckpt = get_latest_checkpoint()
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
