#!/usr/bin/env python3
"""
Single slow simulation viewer for detailed observation and feedback.

Runs one episode at a time with slow rendering so you can observe behavior.
Loads the latest checkpoint and allows manual control of simulation speed.

Usage:
    python scripts/watch_single.py              # Watch latest MJX checkpoint
    python scripts/watch_single.py --step        # Step through frame by frame
    python scripts/watch_single.py --speed 0.1  # Custom speed (default: 0.05)
"""

import argparse
import os
import pathlib
import sys
import time

# Set up the Python path to include the src directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
import numpy as np

from gibbonbot.config.train_mjx import OBS_DIM, ActorCritic
from gibbonbot.envs.stand_env import StandEnv


def latest_checkpoint():
    """Find the latest MJX checkpoint."""
    runs_dir = pathlib.Path("runs/ape_stand_mjx")
    ckpts = sorted(runs_dir.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def load_params(path):
    """Restore Flax params from the flat .npz saved by train_mjx._save()."""
    data = np.load(str(path))
    flat = [
        jnp.array(data[k])
        for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))
    ]
    # round-trip through init to get the pytree structure, then fill
    net = ActorCritic()
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, flat)


def main():
    parser = argparse.ArgumentParser(description="Single slow simulation viewer")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="Simulation speed (seconds per frame, default: 0.05)",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step through frame by frame (press Enter for each step)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Specific checkpoint to load (e.g., 'mjx_000001000000.npz')",
    )
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint == "latest":
        ckpt = latest_checkpoint()
        if ckpt is None:
            print("Error: No checkpoints found in runs/ape_stand_mjx/")
            sys.exit(1)
    else:
        ckpt = pathlib.Path("runs/ape_stand_mjx") / args.checkpoint
        if not ckpt.exists():
            print(f"Error: Checkpoint {args.checkpoint} not found")
            sys.exit(1)

    print(f"Loading checkpoint: {ckpt.name}")
    params = load_params(ckpt)

    # Create deterministic action function
    net = ActorCritic()

    @jax.jit
    def act(params, obs):
        mean, log_std, _ = net.apply(params, obs)
        return jnp.tanh(mean)  # deterministic: use mean action

    # Initialize environment
    env = StandEnv(render_mode="human")

    print("Controls:")
    print("  - Close the MuJoCo window to exit")
    print("  - Press Enter to advance one frame (if --step is used)")
    print(f"  - Simulation speed: {args.speed}s per frame")
    print("\nObservation tips:")
    print("  - Watch for upside-down flips (should be penalized)")
    print("  - Check if robot maintains upright posture")
    print("  - Observe joint coordination and stability")

    def run_episode():
        obs, _ = env.reset()

        # Pad observation to match MJX format (add dummy tree position)
        _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)
        obs_padded = np.concatenate([obs, _TREE_PAD])

        total_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < 1000:  # Limit episode length
            # Get action from policy
            action = np.array(act(params, jnp.array(obs_padded)))

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Re-pad observation for next step
            obs_padded = np.concatenate([obs, _TREE_PAD])

            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # Display info
            sys.stdout.write(
                f"\rStep: {step_count:4d}  "
                f"Reward: {total_reward:7.2f}  "
                f"Height: {info['above_ground']:.3f}m  "
                f"Upright: {info['upright_reward']:.3f}  "
                f"Status: {'FALLEN' if info['fallen'] else 'ALIVE'}"
            )
            sys.stdout.flush()

            # Control simulation speed
            if args.step:
                input(" <Press Enter for next frame>")
            else:
                time.sleep(args.speed)

            # Check if viewer is still open
            viewer_alive = (
                getattr(env, "_viewer", None) is None or env._viewer.is_running()
            )
            if not viewer_alive:
                break

        print()  # New line after episode
        status = "FALLEN" if info.get("fallen", False) else "ALIVE"
        print(
            f"Episode Summary: {status} | Total Reward: {total_reward:.2f} | Steps: {step_count}"
        )

        return not viewer_alive  # Return True if user closed viewer

    # Run episodes until user closes viewer
    while True:
        if run_episode():
            break  # User closed viewer

        # Small delay before next episode
        time.sleep(1)

        # Reset for next episode
        print("\n" + "=" * 60)
        print("Starting new episode...")

    env.close()
    print("\nViewer closed. Goodbye!")


if __name__ == "__main__":
    main()
