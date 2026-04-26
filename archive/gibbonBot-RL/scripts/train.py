#!/usr/bin/env python3
"""
Unified training script for gibbonBot-RL

Usage:
    python train.py stand          # GPU training (MJX) with auto viewer
    python train.py stand --cpu     # CPU training (SB3) with auto viewer
    python train.py reach           # Train reach task
    python train.py --help          # Show all options

Features:
    - Automatic viewer launch
    - GPU/CPU detection
    - Task selection
    - Progress monitoring
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Unified gibbonBot-RL training script")
    parser.add_argument(
        "task",
        nargs="?",
        default="stand",
        choices=["stand", "reach"],
        help="Task to train (stand or reach)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (SB3)")
    parser.add_argument("--gpu", action="store_true", help="Force GPU training (MJX)")
    parser.add_argument(
        "--envs", type=int, default=None, help="Number of parallel environments"
    )
    parser.add_argument(
        "--no-watch", action="store_true", help="Disable automatic viewer"
    )
    parser.add_argument(
        "--inherit", action="store_true", help="Inherit from best checkpoint"
    )
    return parser.parse_args()


def detect_gpu():
    """Check if CUDA is available"""
    try:
        import jax

        devices = jax.devices()
        return any("cuda" in str(d).lower() for d in devices)
    except:
        return False


def run_viewer(task):
    """Run the viewer in a separate process"""
    viewer_cmd = [sys.executable, "src/gibbonbot/utils/watch.py", f"--task", task]
    print(f"\n🎬 Starting viewer: {' '.join(viewer_cmd)}\n")
    subprocess.run(viewer_cmd)


def main():
    args = parse_args()

    # Set up environment
    os.environ["PYTHONPATH"] = "."

    # Determine training mode
    has_gpu = detect_gpu()
    use_gpu = args.gpu or (has_gpu and not args.cpu)

    print(f"🤖 gibbonBot-RL Training")
    print(f"   Task: {args.task}")
    print(f"   Mode: {'GPU (MJX)' if use_gpu else 'CPU (SB3)'}")
    print(f"   GPU detected: {has_gpu}")
    print(f"   Viewer: {'Enabled' if not args.no_watch else 'Disabled'}")
    print(f"   Inherit: {args.inherit}")
    if args.envs:
        print(f"   Environments: {args.envs}")
    print()

    # Build training command
    if use_gpu:
        script = "src/gibbonbot/config/train_mjx.py"
        cmd = [sys.executable, script]
        if args.envs:
            cmd.extend(["--envs", str(args.envs)])
        if args.inherit:
            cmd.append("--inherit")
    else:
        # Use unified environment
        script = "scripts/train_fixed.py"
        cmd = [sys.executable, script, "--task", args.task]
        if args.envs:
            cmd.extend(["--envs", str(args.envs)])
        if args.inherit:
            cmd.append("--inherit")
        if not args.no_watch:
            cmd.append("--watch")

    print(f"🚀 Starting training: {' '.join(cmd)}")

    # Start training process
    train_process = subprocess.Popen(cmd)

    # Start viewer in background if requested
    if not args.no_watch and use_gpu:
        viewer_thread = threading.Thread(
            target=run_viewer, args=(args.task,), daemon=True
        )
        viewer_thread.start()
        time.sleep(2)  # Give viewer time to start

    # Wait for training to complete
    try:
        train_process.wait()
    except KeyboardInterrupt:
        print("\n⏹️  Training stopped by user")
        train_process.terminate()
        if not args.no_watch and use_gpu:
            viewer_thread.join(timeout=1)

    print(f"\n✅ Training completed!")


if __name__ == "__main__":
    main()
