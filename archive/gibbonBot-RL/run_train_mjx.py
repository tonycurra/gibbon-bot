#!/usr/bin/env python3
"""
Wrapper script to run train_mjx.py with correct Python path
"""

import os
import subprocess
import sys

# Set up the Python path to include the src directory
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")

# Set PYTHONPATH environment variable
env = os.environ.copy()
env["PYTHONPATH"] = src_path

# Run the actual training script
script_path = os.path.join(project_root, "scripts", "train_mjx.py")

try:
    result = subprocess.run(
        [sys.executable, script_path] + sys.argv[1:], env=env, check=True
    )
    sys.exit(result.returncode)
except subprocess.CalledProcessError as e:
    sys.exit(e.returncode)
except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    sys.exit(1)
