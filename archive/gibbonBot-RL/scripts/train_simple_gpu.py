#!/usr/bin/env python3
"""
SIMPLE GPU TRAINING

Direct height rewards with GPU acceleration.
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
    parser = argparse.ArgumentParser(description="Simple GPU Training")
    parser.add_argument(
        "--steps", type=int, default=1000000, help="Total training steps"
    )
    parser.add_argument(
        "--envs", type=int, default=None, help="Number of parallel envs (auto if None)"
    )
    args = parser.parse_args()

    print(f"🚀 Simple GPU Training")
    print(f"   Steps: {args.steps:,}")
    print(f"   Envs: {'auto' if args.envs is None else args.envs}")
    print(f"   Goal: Lift all arm bases as high as possible")
    print()

    # Run GPU training with simple rewards
    from src.gibbonbot.config.train_mjx import main as gpu_main

    try:
        gpu_main()
    except KeyboardInterrupt:
        print("\n⏹️ Training stopped by user")


if __name__ == "__main__":
    main()
