"""
One command to start training + live viewer together.

Usage:
    python run.py                  # stand task (default)
    python run.py --task reach
    python run.py --task stand --resume

Training runs in the background. Viewer runs in this terminal.
Close the viewer window (or Ctrl+C) to stop both.
"""

import argparse
import subprocess
import sys
import time
import pathlib
import signal
import os

TASKS = {"reach", "stand"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",   default="stand", choices=list(TASKS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_cmd = [sys.executable, "train.py", "--task", args.task]
    if args.resume:
        train_cmd.append("--resume")

    print(f"[run] starting training:  {' '.join(train_cmd)}")
    trainer = subprocess.Popen(train_cmd)

    # give trainer a moment to start before viewer tries to find a checkpoint
    print(f"[run] waiting for first checkpoint...")
    time.sleep(3)

    print(f"[run] starting viewer...")
    try:
        # viewer runs in foreground — blocks until window is closed
        subprocess.run([sys.executable, "watch.py", "--task", args.task])
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[run] stopping trainer...")
        trainer.terminate()
        try:
            trainer.wait(timeout=5)
        except subprocess.TimeoutExpired:
            trainer.kill()
        print("[run] done.")


if __name__ == "__main__":
    main()
