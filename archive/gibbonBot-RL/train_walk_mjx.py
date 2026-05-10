#!/usr/bin/env python3
"""Back-compat shim — calls train_mjx.py with --task walk.

The actual trainer is train_mjx.py, which is task-parameterized. This file
preserves the old `train_walk_mjx.py` entry point so existing scripts and
muscle memory keep working.
"""
import os
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    # Strip any user-supplied --task to avoid conflict (this shim implies walk).
    cleaned = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--task":
            skip_next = True
            continue
        if a.startswith("--task="):
            continue
        cleaned.append(a)
    cmd = [sys.executable, "train_mjx.py", "--task", "walk", *cleaned]
    os.execv(cmd[0], cmd)
