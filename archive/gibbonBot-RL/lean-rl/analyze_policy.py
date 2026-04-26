#!/usr/bin/env python3
"""
CPU-based policy analysis - slow, detailed observation
Load any checkpoint and analyze behavior in detail

Usage:
    python analyze_policy.py              # Analyze latest checkpoint
    python analyze_policy.py mjx_00010000000.npz  # Specific checkpoint
    python analyze_policy.py --step       # Frame-by-frame
    python analyze_policy.py --speed 0.1 # Adjust speed
"""

import argparse
import os
import pathlib
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "..", "src")
sys.path.insert(0, src_path)

import jax
import jax.numpy as jnp
import numpy as np

from gibbonbot.config.train_mjx import OBS_DIM, ActorCritic
from gibbonbot.envs.stand_env import StandEnv


def load_params(path):
    """Load parameters from checkpoint."""
    data = np.load(str(path))
    flat = [
        jnp.array(data[k])
        for k in sorted(data.files, key=lambda s: int(s.replace("arr_", "")))
    ]
    net = ActorCritic()
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    _, treedef = jax.tree_util.tree_flatten(template)
    return jax.tree_util.tree_unflatten(treedef, flat)


def get_latest_checkpoint():
    """Find latest checkpoint."""
    runs_dir = pathlib.Path("runs/ape_stand_mjx")
    ckpts = sorted(runs_dir.glob("mjx_*.npz"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def analyze_checkpoint(checkpoint_path, args):
    """Analyze a specific checkpoint in detail."""
    print(f"🔍 Analyzing: {checkpoint_path}")
    print("=" * 60)

    # Load policy
    params = load_params(checkpoint_path)

    # Create action function
    net = ActorCritic()

    @jax.jit
    def act(params, obs):
        mean, log_std, _ = net.apply(params, obs)
        return jnp.tanh(mean)  # Deterministic

    # Run episodes
    env = StandEnv(render_mode="human")

    try:
        for episode in range(10):  # Run multiple episodes
            print(f"\n🎬 Episode {episode + 1}/10")
            obs, _ = env.reset()

            # Pad observation for MJX compatibility
            _TREE_PAD = np.array([5.0, 0.0, 1.5], dtype=np.float32)
            obs_padded = np.concatenate([obs, _TREE_PAD])

            total_reward = 0.0
            step_count = 0
            done = False

            while not done and step_count < 500:
                # Get action from policy
                action = np.array(act(params, jnp.array(obs_padded)))

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)

                # Re-pad for next step
                obs_padded = np.concatenate([obs, _TREE_PAD])

                total_reward += reward
                step_count += 1
                done = terminated or truncated

                # Display metrics
                sys.stdout.write(
                    f"\r  Step: {step_count:4d} | "
                    f"Reward: {total_reward:7.2f} | "
                    f"Height: {info['above_ground']:.3f}m | "
                    f"Status: {'FALLEN' if info['fallen'] else 'ALIVE'}"
                )
                sys.stdout.flush()

                # Check viewer
                if not env._viewer.is_running():
                    return

                # Control speed
                if args.step:
                    input(" <Press Enter>")
                else:
                    import time

                    time.sleep(args.speed)

            # Episode summary
            status = "FALLEN" if info.get("fallen", False) else "ALIVE"
            print(
                f"\n  📊 Summary: {status} | Reward: {total_reward:.2f} | Steps: {step_count}"
            )

    except KeyboardInterrupt:
        print("\n⏹️  Analysis stopped by user")
    finally:
        env.close()
        print("\n👋 Analysis complete")


def main():
    parser = argparse.ArgumentParser(description="CPU policy analysis")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="latest",
        help="Checkpoint to analyze (default: latest)",
    )
    parser.add_argument("--step", action="store_true", help="Step-by-step analysis")
    parser.add_argument("--speed", type=float, default=0.05, help="Playback speed")
    args = parser.parse_args()

    # Find checkpoint
    if args.checkpoint == "latest":
        ckpt = get_latest_checkpoint()
        if not ckpt:
            print("❌ No checkpoints found!")
            sys.exit(1)
    else:
        ckpt = pathlib.Path("runs/ape_stand_mjx") / args.checkpoint
        if not ckpt.exists():
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            sys.exit(1)

    analyze_checkpoint(ckpt, args)

    analyze_checkpoint(ckpt, args)


if __name__ == "__main__":
    main()
