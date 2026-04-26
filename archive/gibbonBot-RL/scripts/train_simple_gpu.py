#!/usr/bin/env python3
"""
SIMPLIFIED GPU TRAINING: Focus on lifting all arm bases high

Uses MJX for GPU acceleration with simple reward function.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def main():
    parser = argparse.ArgumentParser(description="Simplified GPU Training")
    parser.add_argument(
        "--steps", type=int, default=1_000_000, help="Total training steps"
    )
    parser.add_argument(
        "--envs", type=int, default=None, help="Number of parallel envs (auto if None)"
    )
    args = parser.parse_args()

    print(f"🚀 Starting SIMPLE GPU training")
    print(f"   Total steps: {args.steps:,}")
    print(f"   Parallel envs: {'auto' if args.envs is None else args.envs}")
    print(f"   Goal: Lift all arm bases as high as possible")
    print(f"   Reward: Direct height reward (higher arms = higher reward)")
    print()

    # Run the GPU training script with simplified rewards
    from src.gibbonbot.config.train_mjx import main as gpu_main

    # Set steps if provided
    if args.steps:
        # We'll need to modify this in the actual training script
        print(f"⚠️  Note: Step count will be set in training script")

    # Run GPU training
    try:
        gpu_main()
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")


if __name__ == "__main__":
    main()
